import math

import torch
import torch.nn.functional as F
from mamba_ssm import Mamba
from timm.models.layers import trunc_normal_
from torch import nn


# PVMLayer是基于mamba结构进行的修改
# 记住输入mamba的张量必须是(B,H*W,C)结构，输出也是(B,H*W,C)结构，mamba
# 不改变维度，只是拿到结果之后，人为进行nn.Linear(input_dim, output_dim)变换，使之通道发生变换
# 输入(B,input_dim,H,W)------输出(B,output_dim,H,W) 只改变维度
class PVMLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
            d_model=input_dim // 4,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))  # 可学习的参数，初始值为 1，用于控制跳跃连接（skip connection）的影响。

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()  # 返回H*W
        img_dims = x.shape[2:]  # 返回(H,W)

        # ----------------------- #
        # (B,C,H,W)--(B,H*W,C)
        # ----------------------- #
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        # --------------------------------------------------------------------- #
        # 沿着通道维度dim = 2,将x_norm切成4个张量，每个张量具有相同的维度，chunk的意思是块
        # (B,H*W,C)---4个(B,H*W,C/4)
        # --------------------------------------------------------------------- #
        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)
        # -------------------------------- #
        # 对每一个x 进行mamba 和 自适应残差连接
        # (B,H*W,C/4)---(B,H*W,C/4)
        # -------------------------------- #
        x_mamba1 = self.mamba(x1) + self.skip_scale * x1
        x_mamba2 = self.mamba(x2) + self.skip_scale * x2
        x_mamba3 = self.mamba(x3) + self.skip_scale * x3
        x_mamba4 = self.mamba(x4) + self.skip_scale * x4
        # -------------------------------- #
        # 4个(B,H*W,C/4)---(B,H*W,C)
        # -------------------------------- #
        x_mamba = torch.cat([x_mamba1, x_mamba2, x_mamba3, x_mamba4], dim=2)
        # --------------------------- #
        # (B,H*W,C)---(B,H*W,output_dim)
        # --------------------------- #
        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        # ------------------------------------------------------------ #
        # (B,H*W,output_dim)---(B,output_dim,H*W)---(B,output_dim,H,W)
        # ------------------------------------------------------------ #
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out


# ----------------------------------------- #
# 通道注意力桥接模块:获得不同张量的通道注意力权重
# 大小为:
# b, c0, H/2, W/2
# b, c1, H/4, W/4
# b, c2, H/8, W/8
# b, c3, H/16, W/16
# b, c4, H/32, W/32
# ----------------------------------------- #
class Channel_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        c_list_sum = sum(c_list) - c_list[-1]  # 计算除最后一个通道数之外的所有通道数之和
        self.split_att = split_att  # 参数用来选择注意力机制的类型：'fc' 表示使用全连接层，而非 'fc' 表示使用一维卷积层。
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 平均池化，输入通道的空间缩小到(1,1),每个通道只保留了一个平均值
        # 一维卷积层，用于汇总多个输入张量的注意力信息。
        self.get_all_att = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        # ------------------------------------------- #
        # att1 到 att5:这些层用于为每个输入生成通道注意力权重。
        # 根据 split_att 参数，选择使用全连接层或一维卷积层。
        # ------------------------------------------- #
        self.att1 = nn.Linear(c_list_sum, c_list[0]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[0], 1)
        self.att2 = nn.Linear(c_list_sum, c_list[1]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[1], 1)
        self.att3 = nn.Linear(c_list_sum, c_list[2]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[2], 1)
        self.att4 = nn.Linear(c_list_sum, c_list[3]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[3], 1)
        self.att5 = nn.Linear(c_list_sum, c_list[4]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[4], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, t1, t2, t3, t4, t5):
        # -----------------------------------
        # t1: b, c0, H/2, W/2 ---- b, c0, 1, 1
        # t2: b, c1, H/4, W/4 ---- b, c1, 1, 1
        # t3: b, c2, H/8, W/8  ---- b, c2, 1, 1   >>>>> b, c_list_sum, 1, 1
        # t4: b, c3, H/16, W/16 ---- b, c3, 1, 1
        # t5: b, c4, H/32, W/32 ---- b, c4, 1, 1
        # -----------------------------------
        att = torch.cat((self.avgpool(t1),
                         self.avgpool(t2),
                         self.avgpool(t3),
                         self.avgpool(t4),
                         self.avgpool(t5)), dim=1)
        # ------------------------------------------------------------------------------------
        # b, c_list_sum, 1, 1 ---- b, 1, c_list_sum ---- b, 1, c_list_sum 注意此时卷积不改变通道数 它是将第二个维度 1 看成通道数
        # 在论文中没有体现，没什么用
        # 汇总多个输入张量的注意力信息得到:得到综合注意力向量
        # ------------------------------------------------------------------------------------
        att = self.get_all_att(att.squeeze(-1).transpose(-1, -2))
        if self.split_att != 'fc':
            att = att.transpose(-1, -2)
        # -----------------------------------------------------------------
        # 将综合注意力向量 att 输入 att1 到 att5，分别生成每个输入张量的通道注意力权重
        # 注意 此时全连接会改变通道数 它是将 第三个维度 c_list_sum 看成通道数
        # att1: b, 1, c_list_sum ---- b, 1, c0
        # att2: b, 1, c_list_sum ---- b, 1, c1
        # att3: b, 1, c_list_sum ---- b, 1, c2
        # att4: b, 1, c_list_sum ---- b, 1, c3
        # att5: b, 1, c_list_sum ---- b, 1, c4
        # -----------------------------------------------------------------
        att1 = self.sigmoid(self.att1(att))
        att2 = self.sigmoid(self.att2(att))
        att3 = self.sigmoid(self.att3(att))
        att4 = self.sigmoid(self.att4(att))
        att5 = self.sigmoid(self.att5(att))
        # --------------------------------------
        # att1: b, 1, c0 ---- b, c0, 1, 1 ---- b, c0, H/2, W/2
        # att2: b, 1, c1 ---- b, c1, 1, 1 ---- b, c1, H/4, W/4
        # att3: b, 1, c2 ---- b, c2, 1, 1 ---- b, c2, H/8, W/8
        # att4: b, 1, c3 ---- b, c3, 1, 1 ---- b, c3, H/16, W/16
        # att5: b, 1, c4 ---- b, c4, 1, 1 ---- b, c4, H/32, W/32
        # -------------------------------------
        if self.split_att == 'fc':
            att1 = att1.transpose(-1, -2).unsqueeze(-1).expand_as(t1)
            att2 = att2.transpose(-1, -2).unsqueeze(-1).expand_as(t2)
            att3 = att3.transpose(-1, -2).unsqueeze(-1).expand_as(t3)
            att4 = att4.transpose(-1, -2).unsqueeze(-1).expand_as(t4)
            att5 = att5.transpose(-1, -2).unsqueeze(-1).expand_as(t5)
        else:
            att1 = att1.unsqueeze(-1).expand_as(t1)
            att2 = att2.unsqueeze(-1).expand_as(t2)
            att3 = att3.unsqueeze(-1).expand_as(t3)
            att4 = att4.unsqueeze(-1).expand_as(t4)
            att5 = att5.unsqueeze(-1).expand_as(t5)

        return att1, att2, att3, att4, att5


# -------------------------------------- #
# 空间注意力桥接模块:获得不同张量的空间注意力权重
# 大小为:
# (b, 1, H/4, W/4)
# (b, 1, H/8, W/8)
# (b, 1, H/16, W/16)
# (b, 1, H/32, W/32)
# -------------------------------------- #
class Spatial_Att_Bridge(nn.Module):
    def __init__(self):
        super().__init__()
        # -------------------------------------------- #
        # 该卷积层的作用是将输入的2个通道的特征图压缩成一个通道。
        # 由于膨胀率的使用，卷积核的感受野被扩大，用于捕获更大的空间信息
        # -------------------------------------------- #
        self.shared_conv2d = nn.Sequential(nn.Conv2d(2, 1, 7, stride=1, padding=9, dilation=3),
                                           nn.Sigmoid())

    def forward(self, t1, t2, t3, t4, t5):
        t_list = [t1, t2, t3, t4, t5]
        att_list = []
        # --------------------------------------------------------------------------------------------------------- #
        # t1: b, c0, H/2, W/2 ---- (b, 1, H/2, W/2) + (b, 1, H/2, W/2) ---- (b, 2, H/2, W/2) ---- (b, 1, H/2, W/2)
        # t2: b, c1, H/4, W/4 ---- b, c1, 1, 1 ......                                             (b, 1, H/4, W/4)
        # t3: b, c2, H/8, W/8  ---- b, c2, 1, 1 ......                                            (b, 1, H/8, W/8)
        # t4: b, c3, H/16, W/16 ---- b, c3, 1, 1 ......                                           (b, 1, H/16, W/16)
        # t5: b, c4, H/32, W/32 ---- b, c4, 1, 1 ......                                           (b, 1, H/32, W/32)
        # --------------------------------------------------------------------------------------------------------- #
        for t in t_list:
            avg_out = torch.mean(t, dim=1, keepdim=True)
            max_out, _ = torch.max(t, dim=1, keepdim=True)
            att = torch.cat([avg_out, max_out], dim=1)
            att = self.shared_conv2d(att)
            att_list.append(att)
        return att_list[0], att_list[1], att_list[2], att_list[3], att_list[4]


class SC_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()

        self.catt = Channel_Att_Bridge(c_list, split_att=split_att)
        self.satt = Spatial_Att_Bridge()

    def forward(self, t1, t2, t3, t4, t5):
        r1, r2, r3, r4, r5 = t1, t2, t3, t4, t5
        # --------------------------------------------------------- #
        # *的意思是逐元素乘法
        # 例如:t1   b, c0, H/2, W/2 ---- b, 1, H/2, W/2 (空间注意力权重) * t1(b, c0, H/2, W/2)(原张量) ---- t1(b, c0, H/2, W/2)
        # --------------------------------------------------------- #
        satt1, satt2, satt3, satt4, satt5 = self.satt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = satt1 * t1, satt2 * t2, satt3 * t3, satt4 * t4, satt5 * t5
        # --------------------------------------------------------- #
        # t1(b, c0, H/2, W/2) + 原始t1(b, c0, H/2, W/2) ---- t1(b, c0, H/2, W/2)
        # --------------------------------------------------------- #
        r1_, r2_, r3_, r4_, r5_ = t1, t2, t3, t4, t5
        t1, t2, t3, t4, t5 = t1 + r1, t2 + r2, t3 + r3, t4 + r4, t5 + r5
        # ------------------------------------------ #
        # t1(b, c0, H/2, W/2) ---- (b, c0, H/2, W/2) (通道注意力权重) * t1(b, c0, H/2, W/2) ---- t1(b, c0, H/2, W/2)
        # ------------------------------------------ #
        catt1, catt2, catt3, catt4, catt5 = self.catt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = catt1 * t1, catt2 * t2, catt3 * t3, catt4 * t4, catt5 * t5
        # -------------------------------------------------------- #
        # t1(b, c0, H/2, W/2) + 从空间注意力出来的t1(b, c0, H/2, W/2) ---- (b, c0, H/2, W/2)
        # -------------------------------------------------------- #
        return t1 + r1_, t2 + r2_, t3 + r3_, t4 + r4_, t5 + r5_


class UltraLight_VM_UNet(nn.Module):

    def __init__(self, num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64],
                 split_att='fc', bridge=True):
        super().__init__()

        self.bridge = bridge

        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
        )
        self.encoder4 = nn.Sequential(
            PVMLayer(input_dim=c_list[2], output_dim=c_list[3])
        )
        self.encoder5 = nn.Sequential(
            PVMLayer(input_dim=c_list[3], output_dim=c_list[4])
        )
        self.encoder6 = nn.Sequential(
            PVMLayer(input_dim=c_list[4], output_dim=c_list[5])
        )

        if bridge:
            self.scab = SC_Att_Bridge(c_list, split_att)
            print('SC_Att_Bridge was used')

        self.decoder1 = nn.Sequential(
            PVMLayer(input_dim=c_list[5], output_dim=c_list[4])
        )
        self.decoder2 = nn.Sequential(
            PVMLayer(input_dim=c_list[4], output_dim=c_list[3])
        )
        self.decoder3 = nn.Sequential(
            PVMLayer(input_dim=c_list[3], output_dim=c_list[2])
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1),
        )
        self.decoder5 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1),
        )
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # 1, 3, H, W --- 1, 8, H/2, W/2
        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out
        # 1, 8, H/2, W/2 ---  1, 16, H/4, W/4
        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out
        # 1, 16, H/4, W/4 ---  1, 24, H/8, W/8
        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out
        # 1, 24, H/8, W/8 --- 1, 32, H/16, W/16
        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        t4 = out
        # 1, 32, H/16, W/16 --- 1, 48, H/32, W/32
        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        t5 = out
        # ----------------------------------- #
        # 对  t1,t2,t3,t4,t5 进行 SAB和CAB 的 加权
        # ----------------------------------- #
        if self.bridge: t1, t2, t3, t4, t5 = self.scab(t1, t2, t3, t4, t5)
        # ------------------------------------------ #
        # 不再进行下采样，用PVM进行特征提取，只会 改变通道数
        # 1, 48, H/32, W/32 --- 1, 64, H/32, W/32
        # ------------------------------------------ #
        out = F.gelu(self.encoder6(out))

        # ------------------------------------------ #
        # 不进行上采样，用PVM进行特征提取，只会 改变通道数
        # 1, 64, H/32, W/32 --- 1, 48, H/32, W/32 + 1, 48, H/32, W/32 --- 1, 48, H/32, W/32
        # ------------------------------------------ #
        out5 = F.gelu(self.dbn1(self.decoder1(out)))
        out5 = torch.add(out5, t5)

        # 1, 48, H/32, W/32 --- 1, 32, H/16, W/16 + 1, 32, H/16, W/16 --- 1, 32, H/16, W/16
        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))
        out4 = torch.add(out4, t4)  # b, c3, H/16, W/16

        # 1, 32, H/16, W/16 --- 1, 24, H/8, W/8  + 1, 24, H/8, W/8 --- 1, 24, H/8, W/8
        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))
        out3 = torch.add(out3, t3)

        # 1, 24, H/8, W/8 --- 1, 16, H/4, W/4 + 1, 16, H/4, W/4 --- 1, 16, H/4, W/4
        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c1, H/4, W/4
        out2 = torch.add(out2, t2)  # b, c1, H/4, W/4

        # 1, 16, H/4, W/4 --- 1, 8, H/2, W/2 + 1, 8, H/2, W/2 --- 1, 8, H/2, W/2
        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c0, H/2, W/2
        out1 = torch.add(out1, t1)  # b, c0, H/2, W/2

        # 1, 8, H/2, W/2 --- 1, 3, H, W
        out0 = F.interpolate(self.final(out1), scale_factor=(2, 2), mode='bilinear',
                             align_corners=True)  # b, num_class, H, W

        return torch.sigmoid(out0)


if __name__ == '__main__':
    model = UltraLight_VM_UNet(num_classes=2).to('cuda')
    x = torch.rand(1, 3, 256, 256).to('cuda')
    print(model(x).shape)

    from torchsummary import summary
    from thop import clever_format, profile

    input_shape = [256, 256]
    num_classes = 3
    summary(model, (3, 256, 256))

    dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1]).to('cuda')
    flops, params = profile(model, (dummy_input,), verbose=False)
    # --------------------------------------------------------#
    #   flops * 2是因为profile没有将卷积作为两个operations
    #   有些论文将卷积算乘法、加法两个operations。此时乘2
    #   有些论文只考虑乘法的运算次数，忽略加法。此时不乘2
    #   本代码选择乘2，参考YOLOX。
    # --------------------------------------------------------#
    flops = flops * 2
    flops, params = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))