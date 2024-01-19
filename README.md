# Style Projected Clustering for Domain Generalized Semantic Segmentation

[paper](https://openaccess.thecvf.com/content/CVPR2023/html/Huang_Style_Projected_Clustering_for_Domain_Generalized_Semantic_Segmentation_CVPR_2023_paper.html) | [code](https://gitee.com/mindspore/models/tree/master/research/cv/SPC-Net)

**非官方版**/**This is the unofficial version**

**申明：由于华为对代码的管控，原始训练代码无法带出公司，我手上也没有原始训练代码。这是根据记忆实现的主要核心代码，因此仅可作为借鉴，请辩证使用。给您带来的不便，敬请原谅。**

**Disclaimer: Due to Huawei's control over code, the original training code cannot be taken out of the company, and I do not have the original training code now. This is the core code implemented based on my memory, so it can only be used as a reference, please use it dialectically. We apologize for the inconvenience caused to you.**


## 训练框架/Training Framework
本文的训练框架是基于文章：[RobustNet: Improving Domain Generalization in Urban-Scene Segmentation via Instance Selective Whitening](https://openaccess.thecvf.com/content/CVPR2021/html/Choi_RobustNet_Improving_Domain_Generalization_in_Urban-Scene_Segmentation_via_Instance_Selective_CVPR_2021_paper.html)而实现的，其开源代码在：[https://github.com/shachoi/RobustNet](https://github.com/shachoi/RobustNet)。复现请引用该文章，谢谢！

The training framework of our paper is based on the paper: [RobustNet: Improving Domain Generalization in Urban-Scene Segmentation via Instance Selective Whitening](https://openaccess.thecvf.com/content/CVPR2021/html/Choi_RobustNet_Improving_Domain_Generalization_in_Urban-Scene_Segmentation_via_Instance_Selective_CVPR_2021_paper.html). The source code is at: [https://github.com/shachoi/RobustNet](https://github.com/shachoi/RobustNet). Please cite this paper if you want to reproduce our method, thank you!


我主要修改的是他们的[deepv3.py](https://github.com/shachoi/RobustNet/blob/main/network/deepv3.py)代码。可以将他们的这个代码替换为我实现的，同时添加上我实现的styleRepIN.py代码到“network”文件中。

What I mainly modified was their [deepv3.py](https://github.com/shachoi/RobustNet/blob/main/network/deepv3.py) script. You can replace this script with the one I implemented and add the styleRepIN.py script in the "network" folder.


## 核心代码/Core Code
### 风格表征/Style Representation
在styleRepIN.py中实现，调用是在deepv3.py的405和411行。

It is implemented in the styleRepIN.py script, and is called in lines 405 and 411 of the deepv3.py script.

### 损失函数/Loss Function
在deepv3.py的419行中实现（prototype_learning_batch），语义聚类即在这里实现。

It is implemented in line 419 of the deepv3.py script (prototype_learning_batch), semantic clustering is implemented here.
