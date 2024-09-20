# code writen by Kai Zhao
import torch
import torch.nn as nn
import torch.nn.functional as F
from vlkit import ops as vlops
import math, sys


def ConvBNReLU(in_chs, out_chs, kernel_size=3, stride=1, groups=1):
        return vlops.ConvModule(in_chs, out_chs, kernel_size=kernel_size,
                stride=stride, groups=groups, act_layer=nn.ReLU)

def ConvBNPReLU(in_chs, out_chs, kernel_size=3, stride=1, groups=1):
    return vlops.ConvModule(in_chs, out_chs, kernel_size=kernel_size,
            stride=stride, groups=groups,
            act_layer=nn.PReLU, act_args=dict(num_parameters=out_chs))

class NormFace(nn.Module):
    def __init__(self, in_features, out_features, s=32):
        super(NormFace, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.s = s
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight.data, 0, 0.1)

    def forward(self, input):
        cosine = torch.mm(F.normalize(input), F.normalize(self.weight).t())
        cosine = cosine.clamp(-1, 1)
        return cosine * self.s

class ArcFace(nn.Module):
    """
    ArcFace https://arxiv.org/pdf/1801.07698
    """
    def __init__(self, in_features, out_features, w_transpose=False,
            s=32, m=0.5, warmup_iters=-1, return_m=False):

        super(ArcFace, self).__init__()
        self.w_transpose = w_transpose

        if w_transpose:
            self.weight = nn.Parameter(torch.zeros(in_features, out_features))
        else:
            self.weight = nn.Parameter(torch.zeros(out_features, in_features))

        self.s = s
        self.m = m
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        self.warmup_iters = warmup_iters
        self.return_m = return_m
        self.iter = 0

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight.data, mean=0.0, std=0.01)

    def forward(self, input, label=None):
        if self.w_transpose:
            cosine = torch.mm(F.normalize(input), F.normalize(self.weight, dim=0))
        else:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight, dim=1))

        cosine = cosine.clamp(-1, 1)

        if label is None or self.m == 0:
            return cosine * self.s

        if self.warmup_iters > 0:
            self.iter = self.iter + 1
            if self.iter < self.warmup_iters:
                m = (1 - math.cos((math.pi / self.warmup_iters) * self.iter)) / 2 * self.m
            else:
                m = self.m
            self.cos_m = math.cos(m)
            self.sin_m = math.sin(m)

            if self.iter % (self.warmup_iters // 10) == 0:
                print("ArcFace: iter %d, m=%.3e" % (self.iter, m))
        else:
            m = self.m

        # sin(theta)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        # psi = cos(theta + m)
        psi_theta = cosine * self.cos_m - sine * self.sin_m
        # see http://data.kaizhao.net/notebooks/arcface-psi.html
        psi_theta = torch.where(cosine > -self.cos_m, psi_theta, -psi_theta - 2)

        onehot = torch.zeros_like(cosine).byte()
        onehot = onehot.scatter(dim=1, index=label.view(-1, 1).long(), value=1)

        output = torch.where(onehot, psi_theta, cosine) * self.s

        if self.return_m:
            return output, cosine * self.s, m
        else:
            return output, cosine * self.s

    def __str__(self):
        return "ArcFace(in_features=%d out_features=%d s=%.3f m=%.3f warmup_iters=%d, return_m=%r)" % \
               (self.weight.shape[1], self.weight.shape[0],
                       self.s, self.m, self.warmup_iters, self.return_m)
    def __repr__(self):
        return self.__str__()
    def extra_repr(self):
        return self.__str__()

class GNAP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNAP, self).__init__()

        if in_channels != out_channels:
            self.conv1 = nn.Conv2d(in_channels, out_channels, bias=False)
        else:
            self.conv1 = None

        self.bn1 = nn.BatchNorm2d(out_channels, affine=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.conv1 is not None:
            x = self.conv1(x)

        x = self.bn1(x)

        x_norm_mean = torch.norm(x, p=2, dim=1).mean()
        x = F.normalize(x, p=2, dim=1)
        x *= x_norm_mean

        x = self.pool(x)
        x = self.bn2(x)
        x = x.view(x.shape[0], -1)

        return x

class GWP(nn.Module):
    def __init__(self, in_channels, out_channels, input_size=[14, 14], reduction=16, return_att=False):
        super(GWP, self).__init__()
        self.return_att = False

        self.se = nn.Sequential(
                nn.Conv2d(in_channels, in_channels//reduction, kernel_size=input_size, bias=False),
                nn.PReLU(),
                nn.Conv2d(in_channels//reduction, input_size[0]*input_size[1], kernel_size=1, bias=False),
                nn.Sigmoid()
        )
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 =nn. BatchNorm2d(out_channels)

    def forward(self, x):
        n, c, h, w = x.shape

        att = self.se(x).view(n, 1, h, w)
        x = (x * att.expand_as(x))
        x = F.avg_pool2d(x, kernel_size=(h, w))
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = x.view(n, -1)

        if self.return_att:
            return x, att
        else:
            return x


class GDC(nn.Module):
    def __init__(self, in_channels, out_channels, input_size=[14,14]):
        super(GDC, self).__init__()
        if in_channels != out_channels:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.conv1 = None
        self.conv2 = vlops.ConvModule(out_channels, out_channels, groups=out_channels, padding=0,
                kernel_size=input_size[0], bias=False, act_layer=None)
        self.conv3 = vlops.ConvModule(out_channels, out_channels, kernel_size=1, act_layer=None, bias=None)

    def forward(self, x):
        if self.conv1 is not None:
            x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)

        return x


class Attention(nn.Module):
    def __init__(self, in_channels, out_channels, return_att=False):
        super(Attention, self).__init__()
        self.return_att = return_att

        if in_channels != out_channels:
            self.conv1 = nn.Conv2d(in_channels, out_channels, bias=False)
        else:
            self.conv1 = None
        self.bn1 = nn.BatchNorm2d(out_channels, affine=False)
        self.att = nn.Sequential(
                nn.Conv2d(out_channels, 1, kernel_size=1, bias=False),
                nn.Sigmoid())
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        if self.conv1 is not None:
            x = self.conv1(x)
        x = self.bn1(x)
        att = self.att(x)
        x = x * att
        x = self.pool(x)
        x = x.view(x.shape[0], -1)

        if self.return_att:
            return x, att
        else:
            return x

class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, act_layer=nn.ReLU, act_args={"inplace": True}):
        super(SqueezeExcite, self).__init__()
        reduced_chs = int(se_ratio * in_chs)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, kernel_size=1, bias=True)
        self.act1 = act_layer(**act_args)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_se = self.pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se).expand_as(x)

        return x * self.sigmoid(x_se)


class RBN(nn.Module):
    def __init__(self, in_chs):
        super(RBN, self).__init__()
        self.w = nn.Parameter(torch.zeros(1, in_chs, 1, 1))
        self.bn = nn.BatchNorm2d(in_chs)

    def forward(self, x):
        m = x.mean(dim=(2,3), keepdim=True)
        return self.bn(x + m*self.w)


class SeLU(nn.Module):
    def __init__(self, slope=1e-1):
        super().__init__()
        self.slope = slope
        self.register_buffer('mag', torch.ones(1))

    def forward(self, x):
        y = x.clone()
        y = torch.where(x >=  self.mag, self.slope * (x - self.mag) + self.mag, y)
        y = torch.where(x <= -self.mag, self.slope * (x + self.mag) - self.mag, y)

        num_within = torch.logical_and(x < self.mag, x > -self.mag).sum()
        r = num_within / x.numel()
        if r <= 2 / 3:
            self.mag += 1e-2
        return y

def initialize_weights(modules):
    """ Weight initilize, conv2d and linear is initialized with kaiming_normal
    """
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight,
                                    mode='fan_out',
                                    nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight,
                                    mode='fan_out',
                                    nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.zero_()
