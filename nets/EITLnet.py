# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .backbone_9_ import mit_b0_9,  mit_b2_9
from .backbone import mit_b0,  mit_b2
from .attention import CoordAtt
from .attention import Coord_MultiStrip_Att
from .attention import SEBlock
from .attention import CBAM
from .attention import GatedAttention2D


# MixVisionTransformer:
#   一整条MiT主干网络, 对应架构图中的Transformer Block1-4
#   backbone: RGB stream
#   backbone_9: noise stream

# CW-HPF + Noise stream:
#   RGB -> Noise

# 双流特征融合: Z1 ... Z4 + CAF
#   Segformer:
#       RGB stream:     x = self.backbone.forward(inputs)
#       Noise stream:   y = self.hpf_conv(new_inputs) 
#                       residual_features = self.backbone_3.forward(y)
#    融合:
#       c4 = torch.cat([c4, residual_features[3]], dim=1)
#       c3 = torch.cat([c3, residual_features[2]], dim=1)
#       c2 = torch.cat([c2, residual_features[1]], dim=1)
#       c1 = torch.cat([c1, residual_features[0]], dim=1)
#       c4 = self.caf4(c4)
#       c3 = self.caf3(c3)
#       c2 = self.caf2(c2)
#       c1 = self.caf1(c1)

# SegFormer_GA:
#   在 SegFormer 基础上, 增加 GatedAttention2D 融合模块
#   在 c3(G3), c4(G4) 上叠加 GatedAttention2D

# SegFormerHead: MLP Decoder
#   对融合后的四个尺度特征进行解码, 输出分割结果



# SRMLayer + DepthwiseConv2D: 高频噪声提取分支
def SRMLayer():
    q = [4.0, 12.0, 2.0]
    
    # 手工设计 SRM 高通滤波器
    filter1 = [[0, 0, 0, 0, 0],
               [0, -1, 2, -1, 0],
               [0, 2, -4, 2, 0],
               [0, -1, 2, -1, 0],
               [0, 0, 0, 0, 0]]
    filter2 = [[-1, 2, -2, 2, -1],
               [2, -6, 8, -6, 2],
               [-2, 8, -12, 8, -2],
               [2, -6, 8, -6, 2],
               [-1, 2, -2, 2, -1]]
    filter3 = [[0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 1, -2, 1, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]]
    filter1 = np.asarray(filter1, dtype=float) / q[0]
    filter2 = np.asarray(filter2, dtype=float) / q[1]
    filter3 = np.asarray(filter3, dtype=float) / q[2]
    filters = np.asarray(
        [[filter1, filter1, filter1], [filter2, filter2, filter2], [filter3, filter3, filter3]])
    # print(filters.shape) # shape=(3,3,5,5)
    filters = np.repeat(filters, repeats=3, axis=0)
    filters = torch.from_numpy(filters.astype(np.float32))
    # filters = torch.from_numpy(filters)
    # print(filters.shape) # shape=(9,3,5,5)
    return filters


# self.hpf_conv = DepthwiseConv2D(in_channels=3, out_channels=9, kernel_size=5, padding=2)
class DepthwiseConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(DepthwiseConv2D, self).__init__()

        # 不参与训练, filter 固定(SRM)
        # groups: 把输入通道和输出通道都拆成 groups 组, 每一组之间单独做一套卷积, 互相不看见
        # groups: 1 -> 普通卷积
        # groups: n -> 分组卷积
        # 第 i 组输出通道, 只和第 i 组输入通道做卷积(减少参数量和计算量)

        # groups: 2
        # 输入被分为 2 组: 
        # 第 0 组: 输入通道 0~15;
        # 第 1 组: 输入通道 16~31;

        # 输出也分为 2 组: 
        # 第 0 组输出通道 0~31 只看输入 0~15;
        # 第 1 组输出通道 32~63 只看输入 16~31;

        # groups=in_channels: 3
        # weight.shape: [9, 3, 5, 5]
        self.depthwise = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                   groups=3, bias=bias)
        filters = SRMLayer()
        self.depthwise.weight = nn.Parameter(filters)
        self.depthwise.weight.requires_grad = False

    def forward(self, x):
        out = self.depthwise(x)
        return out


class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, num_classes=2, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
        super(SegFormerHead, self).__init__()
        # c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = [64, 128, 320, 512]      #b0
        # c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = [128, 256, 640, 1024]    # b3

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            c1=embedding_dim * 4,
            c2=embedding_dim,
            k=1,
        )

        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x


class SegFormer(nn.Module):
    def __init__(self, num_classes=2, phi='b0', pretrained=False, dual=False):
        super(SegFormer, self).__init__()
        self.dual = dual
        self.in_channels = {
            'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
            'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
        }[phi]
        self.backbone = {
            'b0': mit_b0,  'b2': mit_b2,
        }[phi](pretrained)
        self.backbone_3 = {
            'b0': mit_b0_9, 'b2': mit_b2_9,
        }[phi](pretrained)
        self.embedding_dim = {
            'b0': 256, 'b1': 256, 'b2': 768,
            'b3': 768, 'b4': 768, 'b5': 768,
        }[phi]
        self.channels = [64, 128, 320, 512]
        # self.channels = [32, 64, 160, 256]

        if self.dual:
            print("----------use constrain-------------")
            self.hpf_conv = DepthwiseConv2D(in_channels=9, out_channels=9, kernel_size=5, padding=2)
            # self.caf4 = CBAM(self.channels[3]*2)  # 512*2 = 1024
            # self.caf3 = CBAM(self.channels[2]*2)  # 320*2 = 640
            # self.caf2 = CBAM(self.channels[1]*2)  # 128*2 = 256
            # self.caf1 = CBAM(self.channels[0]*2)  # 64*2  = 128
            # self.caf4 = SEBlock(self.channels[3]*2, self.channels[3]*2, reduction=32)
            # self.caf3 = SEBlock(self.channels[2]*2, self.channels[2]*2, reduction=32)
            # self.caf2 = SEBlock(self.channels[1]*2, self.channels[1]*2, reduction=32)
            # self.caf1 = SEBlock(self.channels[0]*2, self.channels[0]*2, reduction=32)
            self.caf4 = CoordAtt(self.channels[3], self.channels[3], reduction=32)
            self.caf3 = CoordAtt(self.channels[2], self.channels[2], reduction=32)
            self.caf2 = CoordAtt(self.channels[1], self.channels[1], reduction=32)
            self.caf1 = CoordAtt(self.channels[0], self.channels[0], reduction=32)

            # 解码头要吃的是“融合后的通道数”
            head_in_channels = [c * 2 for c in self.in_channels]
        else:
            # 单分支 baseline：直接用 backbone 的通道数
            head_in_channels = self.in_channels

        self.decode_head = SegFormerHead(num_classes, head_in_channels, self.embedding_dim)

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        x = self.backbone.forward(inputs)
        c1, c2, c3, c4 = x
        if self.dual:
            channels = torch.split(inputs, split_size_or_sections=1, dim=1)
            new_inputs = torch.cat(channels * 3, dim=1)
            y = self.hpf_conv(new_inputs) 
            residual_features = self.backbone_3.forward(y)  # hpf流
            c4 = torch.cat([c4, residual_features[3]], dim=1)
            c3 = torch.cat([c3, residual_features[2]], dim=1)
            c2 = torch.cat([c2, residual_features[1]], dim=1)
            c1 = torch.cat([c1, residual_features[0]], dim=1)
            c4 = self.caf4(c4)
            c3 = self.caf3(c3)
            c2 = self.caf2(c2)
            c1 = self.caf1(c1)
            x = [c1, c2, c3, c4]

        x = self.decode_head.forward(x)

        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

class SegFormer_GA(nn.Module):
    """
    双分支 + CBAM + GatedAttention + 图像级分类头

    输出：
      - seg_logits: 像素级分割 [B, num_classes, H, W]
      - cls_logits: 图像级分类 [B, num_img_classes](当 return_cls=True 时)
    """
    def __init__(self,
                 num_classes=2,
                 phi='b0',
                 pretrained=False,
                 dual=False,
                 num_img_classes=2):     # 图像级分类类别数, 默认二分类(真/假)
        super(SegFormer_GA, self).__init__()
        self.dual = dual
        self.num_img_classes = num_img_classes

        # ------------------------------
        #  Backbone 通道配置(保持不变)
        # ------------------------------
        self.in_channels = {
            'b0': [32, 64, 160, 256],
            'b1': [64, 128, 320, 512],
            'b2': [64, 128, 320, 512],
            'b3': [64, 128, 320, 512],
            'b4': [64, 128, 320, 512],
            'b5': [64, 128, 320, 512],
        }[phi]

        self.backbone = {
            'b0': mit_b0,
            'b2': mit_b2,
        }[phi](pretrained)

        # 噪声流 backbone(输入 9 通道)
        self.backbone_3 = {
            'b0': mit_b0_9,
            'b2': mit_b2_9,
        }[phi](pretrained)

        self.embedding_dim = {
            'b0': 256, 'b1': 256, 'b2': 768,
            'b3': 768, 'b4': 768, 'b5': 768,
        }[phi]

        # 为了可读性, 用一个变量记一下每个 stage 通道
        self.channels = self.in_channels

        # ------------------------------
        #  Dual 分支 + 注意力 + 图像级 head
        # ------------------------------
        if self.dual:
            print("----------use constrain-------------")
            # CW-HPF：把 RGB 3 通道 -> 9 通道, 再做 SRM 卷积
            self.hpf_conv = DepthwiseConv2D(in_channels=9, out_channels=9,
                                            kernel_size=5, padding=2)

            # CAF (简化版)：对拼接后的 Z1..Z4 做 CBAM
            # self.caf4 = CBAM(self.channels[3]*2)  # 512*2 = 1024
            # self.caf3 = CBAM(self.channels[2]*2)  # 320*2 = 640
            # self.caf2 = CBAM(self.channels[1]*2)  # 128*2 = 256
            # self.caf1 = CBAM(self.channels[0]*2)  # 64*2  = 128
            # self.caf4 = SEBlock(self.channels[3]*2, self.channels[3]*2, reduction=32)
            # self.caf3 = SEBlock(self.channels[2]*2, self.channels[2]*2, reduction=32)
            # self.caf2 = SEBlock(self.channels[1]*2, self.channels[1]*2, reduction=32)
            # self.caf1 = SEBlock(self.channels[0]*2, self.channels[0]*2, reduction=32)
            self.caf4 = Coord_MultiStrip_Att(self.channels[3], self.channels[3], reduction=32)
            self.caf3 = Coord_MultiStrip_Att(self.channels[2], self.channels[2], reduction=32)
            self.caf2 = Coord_MultiStrip_Att(self.channels[1], self.channels[1], reduction=32)
            self.caf1 = Coord_MultiStrip_Att(self.channels[0], self.channels[0], reduction=32)

            # GatedAttention2D：在高层语义上再做一次 token 级 gate(你的创新点1)
            self.gate4 = GatedAttention2D(self.channels[3] * 2, num_heads=4)
            self.gate3 = GatedAttention2D(self.channels[2] * 2, num_heads=4)
            # 如有需要可以继续开 gate2/gate1

            # --------------------------
            #  图像级分类 head(创新点2)
            #  使用融合 + 注意力后的 c4 做全局分类
            # --------------------------
            self.cls_pool = nn.AdaptiveAvgPool2d(1)  # [B,C,H,W] -> [B,C,1,1]
            self.cls_head = nn.Sequential(
                nn.Flatten(),                            # [B,C,1,1] -> [B,C]
                nn.Linear(self.channels[3] * 2, 128),    # 1024 -> 128
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, num_img_classes),         # 128 -> num_img_classes
            )

            # 解码头吃的是 Z1..Z4 通道翻倍后的尺寸
            head_in_channels = [c * 2 for c in self.in_channels]
        else:
            head_in_channels = self.in_channels

        # SegFormer MLP 解码头(像素级分割)
        self.decode_head = SegFormerHead(num_classes,
                                         in_channels=head_in_channels,
                                         embedding_dim=self.embedding_dim)

    def forward(self, inputs, return_cls: bool = False):
        """
        Args:
            inputs: [B,3,H,W]
            return_cls: 是否返回图像级分类 logits

        Returns:
            seg_logits                 (当 return_cls=False)
            (seg_logits, cls_logits)   (当 return_cls=True)
        """
        H, W = inputs.size(2), inputs.size(3)

        # 1. RGB 流 backbone
        x = self.backbone(inputs)     # [c1,c2,c3,c4]
        c1, c2, c3, c4 = x

        cls_logits = None  # 先占位

        if self.dual:
            # 2. 构建 9 通道输入：把每个通道复制 3 次 -> [B,9,H,W]
            channels = torch.split(inputs, split_size_or_sections=1, dim=1)  # 3 个 [B,1,H,W]
            new_inputs = torch.cat(channels * 3, dim=1)                      # [B,9,H,W]

            # 3. CW-HPF：SRM 高通滤波
            y = self.hpf_conv(new_inputs)                                    # [B,9,H,W]

            # 4. 噪声流 backbone
            residual_features = self.backbone_3(y)                           # [f1,f2,f3,f4]

            # 5. 双流特征拼接 -> Z1..Z4
            c4 = torch.cat([c4, residual_features[3]], dim=1)  # [B, C4*2, H/32, W/32]
            c3 = torch.cat([c3, residual_features[2]], dim=1)
            c2 = torch.cat([c2, residual_features[1]], dim=1)
            c1 = torch.cat([c1, residual_features[0]], dim=1)

            # 6. CAF (CBAM)：对每个尺度做通道+空间注意力
            c4 = self.caf4(c4)
            c3 = self.caf3(c3)
            c2 = self.caf2(c2)
            c1 = self.caf1(c1)

            # 7. GatedAttention2D：在高层语义上做 token 级门控
            # c4 = self.gate4(c4)
            # c3 = self.gate3(c3)
            # 如需可继续：
            # c2 = self.gate2(c2)
            # c1 = self.gate1(c1)

            # 8. 图像级分类分支(基于注意力后的 c4)
            if return_cls:
                cls_feat = self.cls_pool(c4)      # [B,C,1,1]
                cls_logits = self.cls_head(cls_feat)  # [B,num_img_classes]

            # 9. 多尺度特征给解码头
            x = [c1, c2, c3, c4]
        # else:
        #   单分支 baseline 的话就直接用 backbone 输出的 c1..c4

        # 10. SegFormer 多尺度解码 -> segmentation logits
        seg_logits = self.decode_head(x)                        # [B,num_classes,H/4,W/4]
        seg_logits = F.interpolate(seg_logits, size=(H, W),
                                   mode='bilinear', align_corners=True)

        if return_cls:
            return seg_logits, cls_logits
        else:
            return seg_logits
