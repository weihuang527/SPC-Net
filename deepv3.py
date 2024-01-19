
import logging
import torch
import numpy as np
from torch import nn
from network import Resnet
from network import Mobilenet
from network import Shufflenet
from network.mynn import initialize_weights, Norm2d, Upsample, freeze_weights, unfreeze_weights
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from einops import rearrange, repeat
import torch.distributed as dist
from network.styleRepIN import StyleRepresentation

import torchvision.models as models
from utils.__print import __print, print_rank

def l2_normalize(x):
    return F.normalize(x, p=2, dim=-1)

def momentum_update(old_value, new_value, momentum):
    update = momentum * old_value + (1 - momentum) * new_value
    return update

def sigmoid_rampup(current, rampup_length=40000, anchor=-0.1):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(anchor * phase * phase))

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class _AtrousSpatialPyramidPoolingModule(nn.Module):
    """
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    """

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=(6, 12, 18), local_rank=0):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn
        print_rank("output_stride = {}".format(output_stride), local_rank)
        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 4:
            rates = [4 * r for r in rates]
        elif output_stride == 16:
            pass
        elif output_stride == 32:
            rates = [r // 2 for r in rates]
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          Norm2d(reduction_dim), nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                Norm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, 256, kernel_size=1, bias=False),
            Norm2d(256), nn.ReLU(inplace=True))

    def forward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = Upsample(img_features, x_size[2:])
        out = img_features

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out


class DeepV3Plus(nn.Module):
    """
    Implement DeepLab-V3 model
    A: stride8
    B: stride16
    with skip connections
    """

    def __init__(self,
                 num_classes,
                 trunk='resnet-50',
                 criterion=None,
                 criterion_aux=None,
                 variant='D',
                 skip='m1',
                 skip_num=48,
                 args=None):
        super(DeepV3Plus, self).__init__()
        self.criterion = criterion
        self.criterion_aux = criterion_aux
        self.variant = variant
        self.args = args
        self.trunk = trunk
        self.num_classes = num_classes
        self.batch_size = args.train_batch_size
        if len(args.dataset) == 1:
            self.num_datasets = 10
        else:
            self.num_datasets = len(args.dataset)
        self.gamma = args.gamma_prototype
        self.gamma_rampup = args.gamma_rampup
        self.temperature = args.temperature
        self.gamma_style = args.gamma_style
        self.dis_mode = args.dis_mode
        self.feadim_style1 = args.feadim_style1
        self.feadim_style2 = args.feadim_style2
        self.channel_wise = args.channel_wise
        self.feature_dim = 256

        if trunk == 'shufflenetv2':
            channel_1st = 3
            channel_2nd = 24
            channel_3rd = 116
            channel_4th = 232
            prev_final_channel = 464
            final_channel = 1024
            resnet = Shufflenet.shufflenet_v2_x1_0(pretrained=True, iw=self.args.wt_layer)

            class Layer0(nn.Module):
                def __init__(self, iw):
                    super(Layer0, self).__init__()
                    self.layer = nn.Sequential(resnet.conv1, resnet.maxpool)
                    self.instance_norm_layer = resnet.instance_norm_layer1
                    self.iw = iw

                def forward(self, x_tuple):
                    if len(x_tuple) == 2:
                        w_arr = x_tuple[1]
                        x = x_tuple[0]
                    else:
                        print("error in shufflnet layer 0 forward path")
                        return
                    x = self.layer[0][0](x)
                    if self.iw >= 1:
                        if self.iw == 1 or self.iw == 2:
                            x, w = self.instance_norm_layer(x)
                            w_arr.append(w)
                        else:
                            x = self.instance_norm_layer(x)
                    else:
                        x = self.layer[0][1](x)
                    x = self.layer[0][2](x)
                    x = self.layer[1](x)
                    return [x, w_arr]

            class Layer4(nn.Module):
                def __init__(self, iw):
                    super(Layer4, self).__init__()
                    self.layer = resnet.conv5
                    self.instance_norm_layer = resnet.instance_norm_layer2
                    self.iw = iw

                def forward(self, x_tuple):
                    if len(x_tuple) == 2:
                        w_arr = x_tuple[1]
                        x = x_tuple[0]
                    else:
                        print("error in shufflnet layer 4 forward path")
                        return
                    x = self.layer[0](x)
                    if self.iw >= 1:
                        if self.iw == 1 or self.iw == 2:
                            x, w = self.instance_norm_layer(x)
                            w_arr.append(w)
                        else:
                            x = self.instance_norm_layer(x)
                    else:
                        x = self.layer[1](x)
                    x = self.layer[2](x)
                    return [x, w_arr]

            self.layer0 = Layer0(iw=self.args.wt_layer[2])
            self.layer1 = resnet.stage2
            self.layer2 = resnet.stage3
            self.layer3 = resnet.stage4
            self.layer4 = Layer4(iw=self.args.wt_layer[6])

            if self.variant == 'D':
                for n, m in self.layer2.named_modules():
                    if isinstance(m, nn.Conv2d) and m.stride==(2,2):
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                for n, m in self.layer3.named_modules():
                    if isinstance(m, nn.Conv2d) and m.stride==(2,2):
                        m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif self.variant == 'D16':
                for n, m in self.layer3.named_modules():
                    if isinstance(m, nn.Conv2d) and m.stride==(2,2):
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            else:
                print("Not using Dilation ")
        elif trunk == 'mobilenetv2':
            channel_1st = 3
            channel_2nd = 16
            channel_3rd = 32
            channel_4th = 64

            prev_final_channel = 320

            final_channel = 1280
            resnet = Mobilenet.mobilenet_v2(pretrained=True, iw=self.args.wt_layer, local_rank=self.args.local_rank)
            self.layer0 = nn.Sequential(resnet.features[0],
                                        resnet.features[1])
            self.layer1 = nn.Sequential(resnet.features[2], resnet.features[3],
                                        resnet.features[4], resnet.features[5], resnet.features[6])
            self.layer2 = nn.Sequential(resnet.features[7], resnet.features[8], resnet.features[9], resnet.features[10])
            self.layer3 = nn.Sequential(resnet.features[11], resnet.features[12], resnet.features[13],
                                        resnet.features[14], resnet.features[15], resnet.features[16],
                                        resnet.features[17])
            self.layer4 = nn.Sequential(resnet.features[18])

            if self.variant == 'D':
                for n, m in self.layer2.named_modules():
                    if isinstance(m, nn.Conv2d) and m.stride==(2,2):
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                for n, m in self.layer3.named_modules():
                    if isinstance(m, nn.Conv2d) and m.stride==(2,2):
                        m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif self.variant == 'D16':
                for n, m in self.layer3.named_modules():
                    if isinstance(m, nn.Conv2d) and m.stride==(2,2):
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            else:
                print("Not using Dilation ")
        else:
            channel_1st = 3
            channel_2nd = 64
            channel_3rd = 256
            channel_4th = 512
            prev_final_channel = 1024
            final_channel = 2048
            
            if trunk == 'resnet-18':
                channel_1st = 3
                channel_2nd = 64
                channel_3rd = 64
                channel_4th = 128
                prev_final_channel = 256
                final_channel = 512
                resnet = Resnet.resnet18(wt_layer=self.args.wt_layer)
                resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            elif trunk == 'resnet-50':
                resnet = Resnet.resnet50(wt_layer=self.args.wt_layer, local_rank=self.args.local_rank)
                resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            elif trunk == 'resnet-101': # three 3 X 3
                resnet = Resnet.resnet101(pretrained=True, wt_layer=self.args.wt_layer, local_rank=self.args.local_rank)
                resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1,
                                            resnet.conv2, resnet.bn2, resnet.relu2,
                                            resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
            else:
                raise ValueError("Not a valid network arch")

            self.layer0 = resnet.layer0
            self.layer1, self.layer2, self.layer3, self.layer4 = \
                resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

            if self.variant == 'D':
                for n, m in self.layer3.named_modules():
                    if 'conv2' in n:
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                    elif 'downsample.0' in n:
                        m.stride = (1, 1)
                for n, m in self.layer4.named_modules():
                    if 'conv2' in n:
                        m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                    elif 'downsample.0' in n:
                        m.stride = (1, 1)
            elif self.variant == 'D4':
                for n, m in self.layer2.named_modules():
                    if 'conv2' in n:
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                    elif 'downsample.0' in n:
                        m.stride = (1, 1)
                for n, m in self.layer3.named_modules():
                    if 'conv2' in n:
                        m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                    elif 'downsample.0' in n:
                        m.stride = (1, 1)
                for n, m in self.layer4.named_modules():
                    if 'conv2' in n:
                        m.dilation, m.padding, m.stride = (8, 8), (8, 8), (1, 1)
                    elif 'downsample.0' in n:
                        m.stride = (1, 1)
            elif self.variant == 'D16':
                for n, m in self.layer4.named_modules():
                    if 'conv2' in n:
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                    elif 'downsample.0' in n:
                        m.stride = (1, 1)
            else:
                # raise 'unknown deepv3 variant: {}'.format(self.variant)
                print("Not using Dilation ")

        if self.variant == 'D':
            os = 8
        elif self.variant == 'D4':
            os = 4
        elif self.variant == 'D16':
            os = 16
        else:
            os = 32

        self.output_stride = os
        self.aspp = _AtrousSpatialPyramidPoolingModule(final_channel, 256,
                                                    output_stride=os, local_rank=self.args.local_rank)

        self.bot_fine = nn.Sequential(
            nn.Conv2d(channel_3rd, 48, kernel_size=1, bias=False),
            Norm2d(48),
            nn.ReLU(inplace=True))

        self.bot_aspp = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True))

        self.final1 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True))

        # self.final2 = nn.Sequential(
        #     nn.Conv2d(256, num_classes, kernel_size=1, bias=True))

        self.projection = nn.Sequential(
            nn.Conv2d(256, self.feature_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=1))

        self.dsn = nn.Sequential(
            nn.Conv2d(prev_final_channel, 512, kernel_size=3, stride=1, padding=1),
            Norm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )
        initialize_weights(self.dsn)

        initialize_weights(self.aspp)
        initialize_weights(self.bot_aspp)
        initialize_weights(self.bot_fine)
        initialize_weights(self.final1)
        # initialize_weights(self.final2)
        initialize_weights(self.projection)

        # Setting the flags
        self.eps = 1e-5
        self.whitening = False

        if trunk == 'resnet-101':
            self.three_input_layer = True
            in_channel_list = [64, 64, 128, 256, 512, 1024, 2048]   # 8128, 32640, 130816
            out_channel_list = [32, 32, 64, 128, 256,  512, 1024]
        elif trunk == 'resnet-18':
            self.three_input_layer = False
            in_channel_list = [0, 0, 64, 64, 128, 256, 512]   # 8128, 32640, 130816
            out_channel_list = [0, 0, 32, 32, 64,  128, 256]
        else: # ResNet-50
            self.three_input_layer = False
            in_channel_list = [0, 0, 64, 256, 512, 1024, 2048]   # 8128, 32640, 130816
            out_channel_list = [0, 0, 32, 128, 256,  512, 1024]

        # define prototype
        self.prototypes = nn.Parameter(torch.zeros(self.num_classes, self.num_datasets, self.feature_dim), requires_grad=True)
        self.feat_norm = nn.LayerNorm(self.feature_dim)
        self.mask_norm = nn.LayerNorm(self.num_classes)
        trunc_normal_(self.prototypes, std=0.02)

        self.styleAdaIN1 = StyleRepresentation(num_prototype=self.num_datasets, 
                                        channel_size=self.feadim_style1, 
                                        batch_size=self.batch_size, 
                                        gamma=self.gamma_style,
                                        dis_mode=self.dis_mode,
                                        channel_wise=self.channel_wise)
        self.styleAdaIN2 = StyleRepresentation(num_prototype=self.num_datasets, 
                                        channel_size=self.feadim_style2, 
                                        batch_size=self.batch_size, 
                                        gamma=self.gamma_style,
                                        dis_mode=self.dis_mode,
                                        channel_wise=self.channel_wise)


    def prototype_learning_batch(self, emb, gt_seg, curr_iter, max_iter):
        if self.gamma_rampup:
            momentum = sigmoid_rampup(curr_iter, max_iter)
        else:
            momentum = self.gamma
        var_loss = torch.tensor(0, dtype=emb.dtype, device=emb.device)
        dis_loss = torch.tensor(0, dtype=emb.dtype, device=emb.device)
        protos = self.prototypes.data.clone()  # [K, M, C], [19, 2, 256]
        protos_new = self.prototypes.data.clone()
        contrast_label = torch.zeros(1, dtype=torch.long).cuda()
        
        for dataset_id in range(self.num_datasets):
            emb_b = emb[dataset_id*self.batch_size:(dataset_id+1)*self.batch_size, ...]
            gt_b = gt_seg[dataset_id*self.batch_size:(dataset_id+1)*self.batch_size, ...]
            for k in range(self.num_classes):
                mask_k = gt_b == k
                emb_k = emb_b[mask_k, :]
                num = emb_k.shape[0]
                if num == 0: continue

                mean_k = l2_normalize(torch.mean(emb_k, dim=0))

                # central = mean_k
                central = protos[k, dataset_id, :]

                var_loss += (1 - torch.mm(emb_k, central.reshape(self.feature_dim, 1))).pow(2).mean() / (self.num_classes * self.num_datasets)

                mask_pos = torch.ones(self.num_classes, self.num_datasets).bool().cuda()
                mask_pos[k, :] = False
                logits_pos = torch.mean(torch.mm(protos[k, ...], mean_k.reshape(self.feature_dim, 1)), dim=0, keepdim=True)
                logits_neg = torch.mm(protos[mask_pos, ...], mean_k.reshape(self.feature_dim, 1)).transpose(1, 0)
                logits = torch.cat([logits_pos, logits_neg], dim=1)
                logits /= self.temperature
                dis_loss += F.cross_entropy(logits, contrast_label) / (self.num_classes * self.num_datasets)

                # update prototype
                protos_new[k, dataset_id, :] = momentum_update(old_value=protos[k, dataset_id, :], new_value=mean_k, momentum=momentum)
        self.prototypes = nn.Parameter(l2_normalize(protos_new), requires_grad=True)

        if dist.is_available() and dist.is_initialized():
            protos = self.prototypes.data.clone()
            dist.all_reduce(protos.div_(dist.get_world_size()))
            self.prototypes = nn.Parameter(protos, requires_grad=True)

        return var_loss, dis_loss

    def encoder(self, x):
        w_arr = []

        # ResNet
        # encoder
        if self.trunk == 'mobilenetv2' or self.trunk == 'shufflenetv2':
            x_tuple = self.layer0([x, w_arr])
            x = x_tuple[0]        # [N, 16, 384, 384]
            w_arr = x_tuple[1]
        else:
            if self.three_input_layer:
                x = self.layer0[0](x)
                if self.args.wt_layer[0] == 1 or self.args.wt_layer[0] == 2:
                    x, w = self.layer0[1](x)
                    w_arr.append(w)
                else:
                    x = self.layer0[1](x)
                x = self.layer0[2](x)
                x = self.layer0[3](x)
                if self.args.wt_layer[1] == 1 or self.args.wt_layer[1] == 2:
                    x, w = self.layer0[4](x)
                    w_arr.append(w)
                else:
                    x = self.layer0[4](x)
                x = self.layer0[5](x)
                x = self.layer0[6](x)
                if self.args.wt_layer[2] == 1 or self.args.wt_layer[2] == 2:
                    x, w = self.layer0[7](x)
                    w_arr.append(w)
                else:
                    x = self.layer0[7](x)
                x = self.layer0[8](x)
                x = self.layer0[9](x)  # [N, 128, 192, 192]
            else:   # Single Input Layer
                x = self.layer0[0](x)  # [N, 64, 384, 384]
                if self.args.wt_layer[2] == 1 or self.args.wt_layer[2] == 2:
                    x, w = self.layer0[1](x)
                    w_arr.append(w)
                else:
                    x = self.layer0[1](x)  # [N, 64, 384, 384]
                x = self.layer0[2](x)  # [N, 64, 384, 384]
                x = self.layer0[3](x)  # [N, 64, 192, 192]
        # import pdb; pdb.set_trace()
        x = self.styleAdaIN1(x)

        x_tuple = self.layer1([x, w_arr])  # [N, 256, 192, 192]
        low_level = x_tuple[0]  # [N, 256, 192, 192]
        x_tuple[0] = self.styleAdaIN2(x_tuple[0])

        x_tuple = self.layer2(x_tuple)  # [N, 512, 96, 96]
        x_tuple = self.layer3(x_tuple)  # [N, 1024, 48, 48]
        aux_out = x_tuple[0]  # [N, 1024, 48, 48]
        x_tuple = self.layer4(x_tuple)  # [N, 2048, 48, 48]
        x = x_tuple[0]  # [N, 2048, 48, 48]
        w_arr = x_tuple[1]  # []
        
        return x, aux_out, low_level

    def decoder(self, x, low_level, x_size):
        # decoder
        x = self.aspp(x)  # [N, 1280, 48, 48]
        dec0_up = self.bot_aspp(x)  # [N, 256, 48, 48]

        dec0_fine = self.bot_fine(low_level)  # [N, 48, 192, 192]
        dec0_up = Upsample(dec0_up, low_level.size()[2:])  # [N, 256, 192, 192]
        dec0 = [dec0_fine, dec0_up]
        dec0 = torch.cat(dec0, 1)  # [N, 304, 192, 192]
        dec1 = self.final1(dec0)  # [N, 256, 192, 192]

        main_out = self.projection(dec1)
        main_out = F.normalize(main_out, p=2, dim=1)
        return main_out

    def forward(self, x, gts=None, aux_gts=None, class_gts=None, img_gt=None, curr_iter=0, max_iter=40000):
        x_size = x.size()  # [N, 3, 768, 768]

        # encoder
        x, aux_out, low_level = self.encoder(x)

        # decoder
        _fea_out = self.decoder(x, low_level, x_size)   # [B, C, H, W], [8, 256, 192, 192]
        fea_out = rearrange(_fea_out, 'b c h w -> (b h w) c')    # [BxHxW, C]
        fea_out = self.feat_norm(fea_out)
        fea_out = l2_normalize(fea_out)

        self.prototypes.data.copy_(l2_normalize(self.prototypes))
        masks = torch.einsum('nd,kmd->nmk', fea_out, self.prototypes)   # [BxHxW, M, K]

        main_out = torch.amax(masks, dim=1)   # [BxHxW, K]
        main_out = self.mask_norm(main_out)
        main_out = rearrange(main_out, "(b h w) k -> b k h w", b=_fea_out.shape[0], h=_fea_out.shape[2])  # [B, K, H, W], [8, 19, 192, 192]
        main_out = F.interpolate(input=main_out, size=x_size[2:], mode='bilinear', align_corners=True)

        if self.training:
            seg_loss = self.criterion(main_out, gts)

            gt_seg = gts[:, None, ...]  # [B, 1, 4xH, 4xW], [8, 1, 768, 768]
            gt_seg = F.interpolate(gt_seg.float(), size=_fea_out.size()[2:], mode='nearest')
            gt_seg = gt_seg.squeeze(dim=1)
            fea_out = rearrange(fea_out, '(b h w) c -> b (h w) c', b=_fea_out.shape[0], h=_fea_out.shape[2])
            gt_seg = rearrange(gt_seg, 'b h w -> b (h w)')
            var_loss, dis_loss = self.prototype_learning_batch(fea_out, gt_seg, curr_iter, max_iter)

            # aux_out = self.dsn(aux_out)
            # if aux_gts.dim() == 1:
            #     aux_gts = gts
            # aux_gts = aux_gts.unsqueeze(1).float()
            # aux_gts = nn.functional.interpolate(aux_gts, size=aux_out.shape[2:], mode='nearest')
            # aux_gts = aux_gts.squeeze(1).long()
            # aux_loss = self.criterion_aux(aux_out, aux_gts)
            # main_loss = [seg_loss, aux_loss]
            # return_loss = [main_loss, var_loss, dis_loss]

            return_loss = [seg_loss, var_loss, dis_loss]            
            return return_loss
        else:
            return main_out


def DeepR18V3PlusD(args, num_classes, criterion, criterion_aux):
    """
    Resnet 18 Based Network
    """
    __print("Model : DeepLabv3+, Backbone : ResNet-18", args.local_rank)
    return DeepV3Plus(num_classes, trunk='resnet-18', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D32', skip='m1', args=args)

def DeepR50V3PlusD(args, num_classes, criterion, criterion_aux):
    """
    Resnet 50 Based Network
    """
    __print("Model : DeepLabv3+, Backbone : ResNet-50", args.local_rank)
    return DeepV3Plus(num_classes, trunk='resnet-50', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepR101V3PlusD(args, num_classes, criterion, criterion_aux):
    """
    Resnet 101 Based Network
    """
    __print("Model : DeepLabv3+, Backbone : ResNet-101", args.local_rank)
    return DeepV3Plus(num_classes, trunk='resnet-101', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepR101V3PlusD_OS8(args, num_classes, criterion, criterion_aux):
    """
    Resnet 101 Based Network
    """
    print("Model : DeepLabv3+, Backbone : ResNet-101")
    return DeepV3Plus(num_classes, trunk='resnet-101', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)

def DeepMobileNetV3PlusD(args, num_classes, criterion, criterion_aux):
    """
    MobileNet Based Network
    """
    __print("Model : DeepLabv3+, Backbone : mobilenetv2", args.local_rank)
    return DeepV3Plus(num_classes, trunk='mobilenetv2', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepShuffleNetV3PlusD(args, num_classes, criterion, criterion_aux):
    """
    ShuffleNet Based Network
    """
    __print("Model : DeepLabv3+, Backbone : shufflenetv2", args.local_rank)
    return DeepV3Plus(num_classes, trunk='shufflenetv2', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)
