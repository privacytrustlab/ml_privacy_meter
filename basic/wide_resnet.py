import functools
from typing import Callable, List

import torch
import torch.nn as nn

BN_MOM = 0.9
BN_EPS = 1e-5
"""
Pytorch implentation of Wide Residual Networks based on Jax implementation of widresnet: https://github.com/google/objax/blob/d0aefeeb573fb366f2ee547f6869f2ca1b7ef284/objax/zoo/wide_resnet.py

Reference:
http://arxiv.org/abs/1605.07146
https://github.com/szagoruyko/wide-residual-networks
"""


class WRNBlock(nn.Module):
    def __init__(
        self,
        nin: int,
        nout: int,
        stride: int = 1,
        bn: Callable = functools.partial(nn.BatchNorm2d, momentum=BN_MOM, eps=BN_EPS),
    ):
        super(WRNBlock, self).__init__()
        self.proj_conv = None

        if nin != nout or stride > 1:
            self.proj_conv = nn.Conv2d(
                nin, nout, 1, stride=stride, padding=0, bias=False
            )

        self.norm_1 = bn(nin)
        self.conv_1 = nn.Conv2d(nin, nout, 3, stride=stride, padding=1, bias=False)
        self.norm_2 = bn(nout)
        self.conv_2 = nn.Conv2d(nout, nout, 3, stride=1, padding=1, bias=False)

    def forward(self, x):
        o1 = torch.relu(self.norm_1(x))
        y = self.conv_1(o1)
        o2 = torch.relu(self.norm_2(y))
        z = self.conv_2(o2)
        return z + self.proj_conv(o1) if self.proj_conv else z + x


class WideResNetGeneral(nn.Module):
    def __init__(
        self,
        nin: int,
        nclass: int,
        blocks_per_group: List[int],
        width: int,
        bn: Callable = functools.partial(nn.BatchNorm2d, momentum=BN_MOM, eps=BN_EPS),
    ):
        super(WideResNetGeneral, self).__init__()
        widths = [
            int(v * width)
            for v in [16 * (2**i) for i in range(len(blocks_per_group))]
        ]
        n = 16
        ops = [nn.Conv2d(nin, n, 3, padding=1, bias=False)]
        for i, (block, width) in enumerate(zip(blocks_per_group, widths)):
            stride = 2 if i > 0 else 1
            ops.append(WRNBlock(n, width, stride, bn))
            for b in range(1, block):
                ops.append(WRNBlock(width, width, 1, bn))
            n = width
        ops += [
            bn(n),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(n, nclass),
        ]
        self.layers = nn.Sequential(*ops)

    def forward(self, x):
        return self.layers(x)


class WideResNet(WideResNetGeneral):
    def __init__(
        self,
        nin: int,
        nclass: int,
        depth: int = 28,
        width: int = 2,
        bn: Callable = functools.partial(nn.BatchNorm2d, momentum=BN_MOM, eps=BN_EPS),
    ):
        assert (depth - 4) % 6 == 0, "depth should be 6n+4"
        n = (depth - 4) // 6
        blocks_per_group = [n] * 3
        print("load the correct model")
        super().__init__(nin, nclass, blocks_per_group, width, bn)
