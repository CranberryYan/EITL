import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import math
import torch.nn.functional as F


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup,reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp*2, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup*2, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup*2, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out
    

class SEBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, reduction=16):
        super(SEBlock, self).__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        mid_channels = max(1, in_channels // reduction)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, H, W)
        weight = self.pool(x)           # → (B, C, 1, 1)
        weight = self.fc(weight)        # → (B, C, 1, 1)
        return x * weight               # 输出仍是 (B, C, H, W)
    
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # (B, C, 1, 1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))  # (B, C, 1, 1)
        max_out = self.fc(self.max_pool(x))  # (B, C, 1, 1)
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2  # 保证输出大小一致
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 沿通道维度取最大值和平均值, 保持H, W维度
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        out = self.conv(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.channel_att(x)
        out = out * self.spatial_att(out)
        return out

class BaseAttention(nn.Module):
    def __init__(self, d_model=128, num_heads=4):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.h = num_heads
        self.dh = d_model // num_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, S, D = x.shape

        # Q,K,V projections
        q = self.Wq(x).reshape(B, S, self.h, self.dh).transpose(1, 2)
        k = self.Wk(x).reshape(B, S, self.h, self.dh).transpose(1, 2)
        v = self.Wv(x).reshape(B, S, self.h, self.dh).transpose(1, 2)

        attn = torch.softmax(
            (q @ k.transpose(-2, -1)) / (self.dh ** 0.5),
            dim=-1
        )
        y = attn @ v  # [B, H, S, dh]

        # combine heads
        y = y.transpose(1, 2).reshape(B, S, D)
        y = self.Wo(y)

        return y, attn


class GatedAttention(nn.Module):
    """
    G1 variant from the paper:
    Gate applied AFTER SDPA output, BEFORE output projection.
    """
    def __init__(self, d_model=128, num_heads=4):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.h = num_heads
        self.dh = d_model // num_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

        # per-head, per-feature gate
        self.Wg = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, S, D = x.shape

        q = self.Wq(x).reshape(B, S, self.h, self.dh).transpose(1, 2)
        k = self.Wk(x).reshape(B, S, self.h, self.dh).transpose(1, 2)
        v = self.Wv(x).reshape(B, S, self.h, self.dh).transpose(1, 2)

        attn = torch.softmax(
            (q @ k.transpose(-2, -1)) / (self.dh ** 0.5),
            dim=-1
        )
        y = attn @ v  # [B, H, S, dh]

        # reshape gate for broadcasting
        g = torch.sigmoid(self.Wg(x)).reshape(B, S, self.h, self.dh)
        g = g.transpose(1, 2)  # [B, H, S, dh]

        y = y * g  # ★ GATED ATTENTION — core of the paper

        y = y.transpose(1, 2).reshape(B, S, D)
        y = self.Wo(y)

        return y, attn, g

class GatedAttention2D(nn.Module):
    def __init__(self, channels, num_heads=4, max_tokens=4096):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.max_tokens = max_tokens
        self.attn = GatedAttention(d_model=channels, num_heads=num_heads)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape

        # 1. 形状检查 & max_tokens 保护, 只在非 tracing 模式下启用
        if not torch.jit.is_tracing():
            # 通道检查
            assert C == self.channels, \
                f"Channels mismatch in GatedAttention2D: expected {self.channels}, got {C}"

            # OOM 保护: 只在训练/正常运行时根据 H*W 做分支
            if (self.max_tokens is not None) and (H * W > self.max_tokens):
                # 直接 bypass, 返回原特征
                return x

        # 2. 真正的 gated attention 逻辑(这一部分是可 trace 的)
        S = H * W
        x_flat = x.flatten(2).transpose(1, 2)   # [B, S, C]
        y, attn, g = self.attn(x_flat)          # [B, S, C]
        y = y.transpose(1, 2).reshape(B, C, H, W)
        return y

class StripPooling(nn.Module):
    """
    Multi-Strip StripPooling 模块

    输入:  x ∈ R^{BxCxLx1}  (可以是 Hx1 的纵向条带, 也可以是 Wx1 的横向条带)
    输出:  x' 与输入同形状, 经过多尺度 depthwise strip conv 聚合后的特征

    参数:
        channels: 通道数 C
        kernel_sizes: 多个 strip 卷积核尺寸, 例如 (3, 7)
        use_bn: 是否在融合后做一次 BN
    """
    def __init__(self, channels, kernel_sizes=(3, 7), use_bn=True):
        super(StripPooling, self).__init__()
        self.kernel_sizes = kernel_sizes

        self.strips = nn.ModuleList()
        for k in kernel_sizes:
            assert k % 2 == 1, "strip kernel size 建议使用奇数, 方便 same padding"
            # x 形状为 [B, C, L, 1], 所以 kernel=(k,1) 沿条带长度 L 卷积
            self.strips.append(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=(k, 1),
                    padding=(k // 2, 0),
                    groups=channels,   # depthwise
                    bias=False
                )
            )

        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        # x: [B, C, L, 1]
        feats = [x]
        for conv in self.strips:
            feats.append(conv(x))
        # 多尺度结果 + 原始条带求平均融合
        out = torch.mean(torch.stack(feats, dim=0), dim=0)
        if self.use_bn:
            out = self.bn(out)
        return out

class h_swish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3., inplace=True) / 6.

class Coord_MultiStrip_Att(nn.Module):
    """
    Multi-Strip Coordinate Attention (与当前 CoordAtt 同型, 用于 cat 后的特征)

    约定: 
      - 传入的 inp / oup 是 cat 之前的通道数;
      - 输入张量 x 的通道数应为 2*inp;
      - 输出张量 out 的通道数为 2*oup(一般你这边就是 2*inp)。

    相比原 CoordAtt: 
      - 在 H / W 方向的条带特征上引入 StripPooling, 多尺度 depthwise strip conv;
      - 其它接口、输入输出形状保持一致, 可直接替换现有 caf1~caf4。
    """
    def __init__(self, inp, oup, reduction=32, strip_kernel_sizes=(3, 7)):
        super(Coord_MultiStrip_Att, self).__init__()

        self.inp = inp
        self.oup = oup
        self.in_channels = inp * 2     # 真实输入通道数(cat 之后)
        self.out_channels = oup * 2    # 真实输出通道数(注意力通道数量)

        # 方向池化: 保留 H / W 其中一个维度
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))   # -> [B, Cin, H, 1]
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))   # -> [B, Cin, 1, W]

        # 和原 CoordAtt 一样, 用 "cat 前通道数" 做 reduction
        mip = max(8, inp // reduction)

        # 多尺度 StripPooling, 在条带上做 depthwise 卷积(通道数用真实 Cin)
        self.strip_pool_h = StripPooling(self.in_channels,
                                         kernel_sizes=strip_kernel_sizes,
                                         use_bn=True)
        self.strip_pool_w = StripPooling(self.in_channels,
                                         kernel_sizes=strip_kernel_sizes,
                                         use_bn=True)

        # cat([x_h, x_w], dim=2), 通道数还是 Cin
        self.conv1 = nn.Conv2d(self.in_channels, mip,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        # 生成方向注意力, 通道数与 identity 对齐: 2*oup
        self.conv_h = nn.Conv2d(mip, self.out_channels,
                                kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_w = nn.Conv2d(mip, self.out_channels,
                                kernel_size=1, stride=1, padding=0, bias=False)

        # 如果 2*inp != 2*oup, 则对 identity 做一次 1x1 投影
        if self.in_channels != self.out_channels:
            self.project = nn.Conv2d(self.in_channels, self.out_channels,
                                     kernel_size=1, stride=1, padding=0, bias=False)
        else:
            self.project = None

    def forward(self, x):
        identity = x
        b, c, h, w = x.size()

        # 这里严格按照语义检查一下, 方便 debug(不想要可以去掉 assert)
        # 例如: inp=64 -> 期望 x 的通道数为 128
        assert c == self.in_channels, \
            f"Coord_MultiStrip_Att: expect input channels={self.in_channels}, but got {c}"

        # 1) H / W 方向池化
        x_h = self.pool_h(x)                 # [B, Cin, H, 1]
        x_w = self.pool_w(x)                 # [B, Cin, 1, W]
        x_w = x_w.permute(0, 1, 3, 2)        # -> [B, Cin, W, 1]

        # 2) StripPooling: 多尺度条带上下文
        x_h = self.strip_pool_h(x_h)         # [B, Cin, H, 1]
        x_w = self.strip_pool_w(x_w)         # [B, Cin, W, 1]

        # 3) H/W 合并 + 1x1 conv 融合(结构和原 CoordAtt 保持一致)
        y = torch.cat([x_h, x_w], dim=2)     # [B, Cin, H+W, 1]
        y = self.conv1(y)                    # -> [B, mip, H+W, 1]
        y = self.bn1(y)
        y = self.act(y)

        # 4) 拆回 H / W, 两路生成方向注意力
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)        # [B, mip, 1, W]

        a_h = self.conv_h(x_h).sigmoid()     # [B, Cout, H, 1]
        a_w = self.conv_w(x_w).sigmoid()     # [B, Cout, 1, W]

        # 5) 通道数对齐 & 重标定
        if self.project is not None:
            identity = self.project(identity)  # [B, Cout, H, W]

        out = identity * a_h * a_w           # 广播到 [B, Cout, H, W]
        return out


class ChannelSELayer(nn.Module):
    """
    cSE: Spatial Squeeze & Channel Excitation
    返回的是 [B, C, 1, 1] 的通道注意力 logits
    """
    def __init__(self, in_channels, reduction=16):
        super(ChannelSELayer, self).__init__()
        mid_channels = max(1, in_channels // reduction)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)          # [B, C, H, W] -> [B, C, 1, 1]
        self.fc1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=True)

    def forward(self, x):
        z = self.avg_pool(x)
        z = self.fc1(z)
        z = self.relu(z)
        z = self.fc2(z)     # 不在这里做 sigmoid，方便和 spatial 分支一起融合
        return z            # [B, C, 1, 1]
    
class SpatialSELayer(nn.Module):
    """
    sSE: Channel Squeeze & Spatial Excitation
    返回的是 [B, 1, H, W] 的空间注意力 logits
    """
    def __init__(self, in_channels):
        super(SpatialSELayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)

    def forward(self, x):
        z = self.conv(x)    # [B, 1, H, W]
        return z            # 不在这里做 sigmoid

class ChannelSpatialSELayer(nn.Module):
    def __init__(self, gate_channel, reduction=16):
        super(ChannelSpatialSELayer, self).__init__()
        self.channel_se = ChannelSELayer(gate_channel, reduction=reduction)
        self.spatial_se = SpatialSELayer(gate_channel)

    def forward(self, x):
        # 通道 logits: [B, C, 1, 1]
        c_logits = self.channel_se(x)
        # 空间 logits: [B, 1, H, W]
        s_logits = self.spatial_se(x)

        # 广播相乘 -> [B, C, H, W]
        att_logits = c_logits * s_logits
        att = 1.0 + torch.sigmoid(att_logits)

        return att * x


if __name__ == "__main__":
    torch.manual_seed(0)

    # 测试 CoordAtt 和 Coord_MultiStrip_Att
    # 模拟你网络里的一个 stage
    B, C4, H, W = 2, 64, 32, 32  # 比如 channels[3] = 64
    c4 = torch.randn(B, C4, H, W)
    r4 = torch.randn(B, C4, H, W)

    # 模拟 U-Net / FPN 的 skip 连接 cat
    c4_cat = torch.cat([c4, r4], dim=1)  # [B, 2*C4, H, W]
    print("c4_cat shape:", c4_cat.shape)

    # 原 CoordAtt(你的实现)
    base_caf4 = CoordAtt(C4, C4, reduction=32)
    y_base = base_caf4(c4_cat)
    print("CoordAtt output shape:", y_base.shape)

    # Multi-Strip 版本(改进 CAF)
    ms_caf4 = Coord_MultiStrip_Att(C4, C4, reduction=32, strip_kernel_sizes=(3, 7))
    y_ms = ms_caf4(c4_cat)
    print("Coord_MultiStrip_Att output shape:", y_ms.shape)

    # 简单对比一下两者输出差异和参数量
    diff = (y_base - y_ms).abs().mean().item()
    print("\nMean |difference| between CoordAtt and Coord_MultiStrip_Att:", diff)

    def count_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    print("\nCoordAtt params             :", count_params(base_caf4))
    print("Coord_MultiStrip_Att params:", count_params(ms_caf4))

    # 测试 cSE, sSE, scSE
    B, C, H, W = 2, 320, 32, 32
    x = torch.randn(B, C, H, W)

    scse = ChannelSpatialSELayer(C, reduction=16)
    y = scse(x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)

    # 看一下注意力的范围大致如何（是否在 ~[1,2] 这一段）
    with torch.no_grad():
        c_logits = scse.channel_se(x)
        s_logits = scse.spatial_se(x)
        att = 1.0 + torch.sigmoid(c_logits * s_logits)
        print("Att mean :", att.mean().item())
        print("Att min  :", att.min().item())
        print("Att max  :", att.max().item())