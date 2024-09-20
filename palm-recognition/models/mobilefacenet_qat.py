"""
用于量化训练的 MobileFaceNet
有几点需要注意的：
  1. 只能用 ReLU 作为激活函数且 inplace 必须为 False
  2. 最好用比较常用的layer顺序，比如 Conv->BN->ReLU, Conv->ReLU, Conv->BN。
     因为量化训练前要进行"model_fuse"将多个 op 的组合 (Conv->BN->ReLU 例如) 合并成一个 operator。
  3. 注意残差连接加操作的写法 skip_add = nn.quantized.FloatFunctional()
"""

import math
import torch
import torch.nn.functional as F
from torch import nn
from vlkit import ops as vlops
from .modules import GDC
from .mobilefacenet import _make_divisible
from torch.quantization import QuantStub, DeQuantStub


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, padding):
        super(DepthwiseSeparableConv, self).__init__()

        self.depthwise = vlops.ConvModule(in_planes, in_planes, kernel_size=kernel_size,
                                          padding=padding, groups=in_planes, bias=False,
                                          act_layer=nn.ReLU, act_args={'inplace': False})

        self.pointwise = vlops.ConvModule(in_planes, out_planes, kernel_size=1, bias=False,
                                          act_layer=nn.ReLU, act_args={'inplace': False})

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, qat):
        super(InvertedResidual, self).__init__()
        self.qat = qat
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.has_residual = (inp == oup and stride == 1)

        # point-wise convolution
        self.conv_pw = vlops.ConvModule(inp, hidden_dim, kernel_size=1, bias=False, act_args={'inplace': False})

        # depth-wise convolution
        self.conv_dw = vlops.ConvModule(hidden_dim, hidden_dim, kernel_size=3, stride=stride,
                                        groups=hidden_dim, bias=False, act_args={'inplace': False})

        # point-wise linear convolution
        self.conv_pwl = vlops.ConvModule(hidden_dim, oup, kernel_size=1, bias=False, act_layer=None)

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        residual = x
        x = self.conv_pw(x)
        x = self.conv_dw(x)
        x = self.conv_pwl(x)

        if self.has_residual:
            if self.qat:
                x = self.skip_add.add(x, residual)
            else:
                x = x + residual
        return x


class MobileFaceNet(nn.Module):
    def __init__(self, width_mult=1.0, inverted_residual_setting=None, round_nearest=8,
                 input_size=224, input_channel=3, last_channel=512, output_name="GDC",
                 qat=False):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileFaceNet, self).__init__()
        block = InvertedResidual
        # channel of input images, (3 for rgb images and 1 for IR images)
        self.input_channel = input_channel
        # input channels for each stage
        self.last_channel = last_channel
        self.qat = qat

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [2, 64, 5, 2],
                [4, 128, 1, 2],
                [2, 128, 6, 1],
                [4, 128, 1, 2],
                [2, 128, 2, 1],
            ]
        input_channel = inverted_residual_setting[0][1]  # 64 by default

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        self.conv1 = vlops.ConvModule(self.input_channel, input_channel, kernel_size=3, stride=2,
                                      bias=False, act_args={'inplace': False})
        self.dw_conv = DepthwiseSeparableConv(input_channel, input_channel, kernel_size=3, padding=1)

        features = []
        layers = []
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride,
                                      expand_ratio=t, qat=qat))
                input_channel = output_channel

            layers.append(features)
            features = []

        self.layer0 = nn.Sequential(*layers[0])
        self.layer1 = nn.Sequential(*layers[1])
        self.layer2 = nn.Sequential(*layers[2])
        self.layer3 = nn.Sequential(*layers[3])
        self.layer4 = nn.Sequential(*layers[4])

        self.last_channel = _make_divisible(self.last_channel * max(1.0, width_mult), round_nearest)
        self.conv2 = vlops.ConvModule(input_channel, self.last_channel, kernel_size=1,
                                      bias=False, act_args={'inplace': False})

        x_ = torch.zeros(2, 3, input_size, input_size)
        input_size = self.layer4(self.layer3(self.layer2(self.layer1(self.layer0(self.conv1(x_)))))).shape[-1]

        self.output = GDC(self.last_channel, self.last_channel, input_size=[input_size, ] * 2)

        # only used for quantization-aware training
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        self.init_weights()

    def fuse_model(self):
        for name, m in self.named_modules():
            if isinstance(m, vlops.ConvModule):
                if m.act is not None:
                    torch.quantization.fuse_modules(m, ["conv", "norm", "act"], inplace=True)
                else:
                    torch.quantization.fuse_modules(m, ["conv", "norm"], inplace=True)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.qat:
            x = self.quant(x)

        x = self.conv1(x)
        x = self.dw_conv(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv2(x)
        x = self.output(x)

        if self.qat:
            return self.dequant(x)
        else:
            return x
