import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
import torchvision
import torchvision.models as models

class ResnetEncoder(nn.Module):
    def __init__(self, num_layers=18, pretrained=True):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])  # 在不同的 ResNet 层中通道的数量

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        res_feat_chs = {18: 256,
                        34: 256,
                        50: 1024,
                        101: 2048}

        self.res_feat_chs = res_feat_chs[num_layers]  # 选择通道数

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        # Resnet101 only: 定义额外加的Conv1x1层，使Resnet101的输出tensor shape能与之后的loss factory对齐
        self.conv1x1 = nn.Conv2d(1024, self.res_feat_chs, kernel_size=1)

        # 定义全连接层
        # (输入层, 激活函数, 隐藏层, 激活函数, 输出层)
        # 输入层: 输入大小为 self.res_feat_chs * 7 * 7 (这是将 ResNet 的特征图展平后的大小)，输出大小为 256
        # 激活函数: 激活函数
        # 隐藏层: 输入大小为 256，输出大小为 256
        # 输出层: 输入大小为 256，输出大小为 2 (表示 x 和 y 坐标)
        self.location_xy = nn.Sequential(
                    nn.Linear(self.res_feat_chs * 7 * 7, 256),
                    nn.ReLU(True),
                    nn.Linear(256, 256),
                    nn.ReLU(True),
                    nn.Linear(256, 2),

                )
        self.location_z = nn.Sequential(
            nn.Linear(self.res_feat_chs * 7 * 7, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 1),
        )
        self.orientation_conf = nn.Sequential(
            nn.Linear(self.res_feat_chs * 7 * 7, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 2),  # 2 表示方向置信度的两个值 (方向,置信度)
        )


    def forward(self, input_image, bbox):
        ##############
        # 四层特征提取 #
        ##############
        self.features = []
        # 第一层
        x = self.encoder.conv1(input_image)  # 对输入图像进行卷积
        x = self.encoder.bn1(x)  # 批归一化
        self.features.append(self.encoder.relu(x))  # 使用 relu 激活函数增加非线性
        #print((self.features[-1]).shape)
        # 第二层
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))  # 最大池化操作减少特征图的尺寸
        #print((self.features[-1]).shape)
        # 第三层
        self.features.append(self.encoder.layer2(self.features[-1]))  # 直接通过 layer2 提取特征
        #print((self.features[-1]).shape)
        # 第四层
        self.features.append(self.encoder.layer3(self.features[-1]))  # 直接通过 layer3 提取特征
        #print((self.features[-1]).shape)

        # Resnet101 only: 定义额外加的Conv1x1层，使Resnet101的输出tensor shape能与之后的loss factory对齐
        # self.features.append(self.conv1x1(self.features[-1]))

        ################
        # bbox ROI 对齐 #
        ################
        last_feat = self.features[-1]  # 获取最后一层的特征
        if len(bbox.shape) == 3:  # 基于边界框进行 ROI 对齐
            f = torchvision.ops.roi_align(last_feat, [i/16 for i in bbox], (7, 7))
        else:
            f = torchvision.ops.roi_align(last_feat, [bbox/16], (7, 7))
        # torchvision.ops.roi_align 将输入特征图(last_feat)中对应 bbox 的区域重新采样为固定大小(这里是 7x7)的特征图。
        # 这个操作可以确保即使输入的边界框大小不同，输出的特征图大小也是一致的，从而方便后续的处理
        #print(f.shape)

        # 经过前面的 ROI 对齐操作，f 是一个形状为 (N, C, 7, 7) 的张量
        # N 是 ROI 的数量。-1 表示自动推断大小
        # C 是特征图的通道数(由 self.res_feat_chs 表示)
        # 7x7 是每个 ROI 的固定大小
        f = f.view(-1, self.res_feat_chs * 7 * 7)  # 将经过 ROI 对齐得到的固定大小的特征图转换成一维向量
        #print(f.shape)

        #################
        # 全连接层计算输出 #
        #################
        location_xy = self.location_xy(f)
        location_xy = location_xy.view(-1, 2)
        #print(location_xy.shape)

        location_z = self.location_z(f)
        # print(location_z.shape)
        orientation_conf = self.orientation_conf(f)
        # print(orientation_conf.shape)

        return location_xy, location_z, orientation_conf