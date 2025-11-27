import torch
import torch.nn.functional as F
from mamba_ssm import Mamba
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


class PVMLayer3(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=8):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
            d_model=input_dim // 3,  # Model dimension d_model
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
        x1, x2, x3 = torch.chunk(x_norm, 3, dim=2)
        # -------------------------------- #
        # 对每一个x 进行mamba 和 自适应残差连接
        # (B,H*W,C/4)---(B,H*W,C/4)
        # -------------------------------- #
        x_mamba1 = self.mamba(x1) + self.skip_scale * x1
        x_mamba2 = self.mamba(x2) + self.skip_scale * x2
        x_mamba3 = self.mamba(x3) + self.skip_scale * x3
        # -------------------------------- #
        # 4个(B,H*W,C/4)---(B,H*W,C)
        # -------------------------------- #
        x_mamba = torch.cat([x_mamba1, x_mamba2, x_mamba3], dim=2)
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


class PVMLayer5(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
            d_model=input_dim // 5,  # Model dimension d_model
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
        x1, x2, x3, x4, x5 = torch.chunk(x_norm, 5, dim=2)
        # -------------------------------- #
        # 对每一个x 进行mamba 和 自适应残差连接
        # (B,H*W,C/4)---(B,H*W,C/4)
        # -------------------------------- #
        x_mamba1 = self.mamba(x1) + self.skip_scale * x1
        x_mamba2 = self.mamba(x2) + self.skip_scale * x2
        x_mamba3 = self.mamba(x3) + self.skip_scale * x3
        x_mamba4 = self.mamba(x4) + self.skip_scale * x4
        x_mamba5 = self.mamba(x5) + self.skip_scale * x5
        # -------------------------------- #
        # 4个(B,H*W,C/4)---(B,H*W,C)
        # -------------------------------- #
        x_mamba = torch.cat([x_mamba1, x_mamba2, x_mamba3, x_mamba4, x_mamba5], dim=2)
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


# --------------------------------------------------------
# 该模块对应论文中RVM Layer,输入(B,Cin,H,W)----输出(B,Cout,H,W)
# ssm状态扩展因子:d_state d_conv:局部卷积宽度 expand:块扩展因子
# --------------------------------------------------------
class RVMLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
            d_model=input_dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        # 定义一个线性层，将输入维度映射到输出维度。作用是改变通道
        self.proj = nn.Linear(input_dim, output_dim)
        # 创建一个可学习的参数 skip_scale，用于在跳跃连接中调整输入和 Mamba 输出的权重
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()  # H*W
        img_dims = x.shape[2:]  # (H,W)
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)  # (B,C,H,W)->(B,H*W,C)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm) + self.skip_scale * x_flat
        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)  # (B,H*W,Cin)->(B,H*W,Cout)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim,
                                                *img_dims)  # (B,H*W,Cout)->(B,Cout,H*W)->(B,Cout,H,W)
        return out


# -------------------------- #
# conv + BN + Relu
# -------------------------- #
class _ConvBNReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


# ------------------------------------------------------------------------- #
# 残差深度可分离卷积:逐通道卷积(可能改变分辨率k=5,,s=1,p=2,所以不变) + 逐点卷积(改变通道数)
# 该残差连接 是  x =  x + 逐通道(x) ,随后 x = 逐点卷积(x)
# 输入(B, Cin, H, W) --- (B, Cout, H, W)
# ------------------------------------------------------------------------- #
class _RDSConv(nn.Module):

    def __init__(self, dw_channels, out_channels, ksize=3, stride=1, padding=1, **kwargs):
        super(_RDSConv, self).__init__()
        self.dwconv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, ksize, stride, padding, groups=dw_channels, bias=False),
            nn.BatchNorm2d(dw_channels),
            nn.ReLU(True),
        )
        self.pwconv = nn.Sequential(
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = x + self.dwconv(x)  # 多了一个残差连接
        x = self.pwconv(x)
        return x


# -------------------------- #
# 只有 逐通道卷积(一般是改变分辨率)
# 改变 通道数 不改变分辨 s= 1
# -------------------------- #
class _DWConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DWConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, out_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output


class Classifer(nn.Module):  # (1, 129, H/8, W/8)

    def __init__(self, dw_channels, num_classes=2, stride=1, **kwargs):
        super(Classifer, self).__init__()
        self.conv1 = _ConvBNReLU(dw_channels, dw_channels, stride)
        self.conv2 = _ConvBNReLU(dw_channels, dw_channels, stride)
        self.conv3 = nn.Conv2d(dw_channels, dw_channels, stride)
        self.conv = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(130, num_classes, 1)
        )

        self.conv3 = PVMLayer3(129, 1)

    def forward(self, x):
        bd3 = self.conv3(x)
        x = torch.cat([x, bd3], 1)
        x = self.conv(x)
        return x


# -------------------------------------- #
# conv -- PVM -- RDS -- PVM -- RDS
# -------------------------------------- #
class LearningToDownsample(nn.Module):
    """Learning to downsample module"""

    def __init__(self, dw_channels1=32, dw_channels2=48, out_channels=64, **kwargs):
        super(LearningToDownsample, self).__init__()
        self.conv = _ConvBNReLU(3, dw_channels1, 3, 2, 1)

        self.pvm1 = PVMLayer(dw_channels1, dw_channels2)
        self.pvm1_1 = PVMLayer(dw_channels2, dw_channels2)

        self.pvm2 = PVMLayer(dw_channels2, out_channels)
        self.pvm2_1 = PVMLayer(out_channels, out_channels)

    def forward(self, x):
        x = self.conv(x)

        x = F.max_pool2d(self.pvm1(x), 2, 2)
        x = self.pvm1_1(x)

        x = F.max_pool2d(self.pvm2(x), 2, 2)
        x = self.pvm2_1(x)

        return x


class GlobalFeatureExtractor(nn.Module):
    """Global feature extractor module"""

    def __init__(self, in_channels=64, block_channels=(64, 96, 128),
                 out_channels=128, t=6, num_blocks=(3, 3, 3), **kwargs):
        super(GlobalFeatureExtractor, self).__init__()
        self.bottleneck1 = self._make_layer(LinearBottleneck, in_channels, block_channels[0], num_blocks[0], t, 2)
        self.bottleneck2 = self._make_layer(LinearBottleneck, block_channels[0], block_channels[1], num_blocks[1], t, 2)
        self.bottleneck3 = self._make_layer(LinearBottleneck, block_channels[1], block_channels[2], num_blocks[2], t, 1)
        self.SE0 = SE(block_channels[0])
        self.SE1 = SE(block_channels[1])
        self.SE2 = SE(block_channels[2])

    def _make_layer(self, block, inplanes, planes, blocks, t=6, stride=1):
        layers = []
        layers.append(block(inplanes, planes, t, stride))
        for i in range(1, blocks):
            layers.append(block(planes, planes, t, 1))
            layers.append(PVMLayer(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.SE0(x)

        x = self.bottleneck2(x)
        x = self.SE1(x)

        x = self.bottleneck3(x)
        x = self.SE2(x)

        return x


class LinearBottleneck(nn.Module):
    """LinearBottleneck used in MobileNetV2"""

    def __init__(self, in_channels, out_channels, t=6, stride=2, **kwargs):
        super(LinearBottleneck, self).__init__()
        # 只有在s == 1 且 输入输出通道数相同的情况下 才能使用 短接
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            # pw
            _ConvBNReLU(in_channels, in_channels * t, 1),
            # dw
            _DWConv(in_channels * t, in_channels * t, stride),
            # pw-linear
            nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out


class SE(nn.Module):
    def __init__(self, channel, ratio=16):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid()
        )

    # -------------------------------------------------- #
    # (B, C, H, W) --- (B, C, 1, 1) --- (B, C)
    # (B, C) --- (B, C) --- y(B, C, 1, 1) 此张量是注意力权重
    # 最终返回 x*y(B, C, H, W)
    # -------------------------------------------------- #
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class FastSCVM(nn.Module):
    def __init__(self, num_classes=2, **kwargs):
        super(FastSCVM, self).__init__()

        self.learning_to_downsample = LearningToDownsample(16, 32, 64)
        self.maxpool = nn.MaxPool2d(3, 1, 1)
        self.p1 = PVMLayer(64, 1)
        self.p2 = PVMLayer(64, 64)
        self.p3 = PVMLayer(128, 1)
        self.p4 = PVMLayer(128, 128)
        self.p5 = PVMLayer5(65, 129)

        self.global_feature_extractor = GlobalFeatureExtractor(65, [32, 64, 128], 64, 6, [2, 2, 2])
        self.classifier = Classifer(129, num_classes)

        self.bn_prelu_1 = BNPReLU(64)
        self.bn_prelu_2 = BNPReLU(128)
        self.relu = nn.ReLU(True)

    def forward(self, x):  # (1 ,3 ,256 ,256)
        size = x.size()[2:]
        # 模块1 下采样提取特征 8倍下采样
        higher_res_features = self.learning_to_downsample(x)  # torch.Size([1, 64, 32, 32])

        # 模块2 进行 边界特征提取   只改变通道数
        h1 = self.maxpool(higher_res_features)  # torch.Size([1, 64, 32, 32])
        bd1 = self.p1(higher_res_features) + self.p1(h1)  # torch.Size([1, 1, 32, 32])

        # 模块2 进行全局特征提取    不改变通道数
        output1 = self.p2(higher_res_features)
        output1 = self.bn_prelu_1(output1)  # torch.Size([1, 64, 32, 32])
        output1 = torch.cat([output1, bd1], 1)  # torch.Size([1, 65, 32, 32])

        # 右侧
        # 倒残差结构 特征提取
        x = self.global_feature_extractor(output1)  # torch.Size([1, 128, 8, 8])
        # 模块3 进行 边界特征提取 ，只改变通道数
        x1 = self.maxpool(x)  # torch.Size([1, 128, 8, 8])
        bd2 = self.p3(x) + self.p3(x1)  # torch.Size([1, 1, 8, 8])  yes
        # 模块3 进行 全局特征提取， 只改变通道数
        output2 = self.p4(x)
        output2 = self.bn_prelu_2(output2)  # torch.Size([1, 128, 8, 8]) yes
        output2 = torch.cat([output2, bd2], 1)  # torch.Size([1, 129, 8, 8])
        output2 = F.interpolate(output2, scale_factor=4, mode='bilinear',
                                align_corners=True)  # torch.Size([1, 129, 32, 32]) 右侧结果 进行4倍上采样

        # 左侧
        output3 = self.p5(output1)  # torch.Size([1, 129, 32, 32])


        # 左右侧相加
        output4 = output2 + output3
        output4 = self.relu(output4)  # torch.Size([1, 129, 32, 32])

        # 还原
        output5 = self.classifier(output4)
        output5 = F.interpolate(output5, size, mode='bilinear', align_corners=True)  # 8倍上采样

        return output5


if __name__ == '__main__':
    print('## --------------------------------------------------------------- ##')
    print('## ----------------------对应论文中的params和flops------------------- ##')
    print('## ---------------------------------------------------------------  ##')
    model = FastSCVM(num_classes=2).to('cuda')  ###########此处需要修改

    from thop import profile  ## 导入thop模块
    from torchsummary import summary

    input = torch.rand(1, 3, 800, 800).cuda()
    flops, params = profile(model, inputs=(input,))
    summary(model, (3, 512, 512))
    print('flops', str(flops / 1e9) + 'G')  ## 打印计算量 单位G
    print('params', str(params / 1e6) + 'M')  # 打印参数量 单位M

    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.3fM" % (total / 1e6))
    print(model(input).shape)

