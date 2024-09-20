# from https://github.com/foamliu/MobileFaceNet
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
from .modules import GNAP, GDC, GWP, SqueezeExcite

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=False)
        )


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, padding, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, padding=padding, groups=in_planes,
                                   bias=bias)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=False)

        self.pointwise = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class GDConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, padding, bias=False):
        super(GDConv, self).__init__()
        self.depthwise = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, groups=in_planes,
                                   bias=bias)
        self.bn = nn.BatchNorm2d(in_planes)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.has_residual = (inp == oup and stride == 1)

        # point-wise convolution
        self.conv_pw = ConvBNReLU(inp, hidden_dim, kernel_size=1)

        # depth-wise convolution
        self.conv_dw = ConvBNReLU(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim)

        # point-wise linear convolution
        self.conv_pwl = nn.Sequential(nn.Conv2d(hidden_dim, oup, kernel_size=1, bias=False), nn.BatchNorm2d(oup))

    def forward(self, x):
        residual = x
        x = self.conv_pw(x)
        x = self.conv_dw(x)
        x = self.conv_pwl(x)

        if self.has_residual:
            x += residual

        return x


class MobileFaceNet(nn.Module):
    def __init__(self, width_mult=1.0, inverted_residual_setting=None, round_nearest=8,
                 input_size=224, input_channel=3, last_channel=512, output_name="GDC"):
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

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [2, 64, 5, 2],
                [4, 128, 1, 2],
                [2, 128, 6, 1],
                [4, 128, 1, 2],
                [2, 128, 2, 1],
            ]
        input_channel = inverted_residual_setting[0][1] # 64 by default

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        # input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(self.last_channel * max(1.0, width_mult), round_nearest)
        self.conv1 = ConvBNReLU(self.input_channel, input_channel, stride=2)
        self.dw_conv = DepthwiseSeparableConv(input_channel, input_channel, kernel_size=3, padding=1)
        features = []
        layers = []
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

            layers.append(features)
            features = []

        self.layer0 = nn.Sequential(*layers[0])
        self.layer1 = nn.Sequential(*layers[1])
        self.layer2 = nn.Sequential(*layers[2])
        self.layer3 = nn.Sequential(*layers[3])
        self.layer4 = nn.Sequential(*layers[4])

        self.conv2 = ConvBNReLU(input_channel, self.last_channel, kernel_size=1)

        x_ = torch.zeros(2,3,input_size, input_size)
        input_size = self.layer4(self.layer3(self.layer2(self.layer1(self.layer0(self.conv1(x_)))))).shape[-1]

        if output_name == "GDC":
            self.output = GDC(self.last_channel, self.last_channel, input_size=[input_size]*2)
        elif output_name == "GWP":
            self.output = GWP(self.last_channel, self.last_channel)
        elif output_name == "GNAP":
            self.output = GNAP(self.last_channel, self.last_channel)
        else:
            raise ValueError("Unsupported output_name: %s" % output_name)

        self.init_weights()

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
        x = self.conv1(x)
        x = self.dw_conv(x)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv2(x)
        x = self.output(x)

        return x

