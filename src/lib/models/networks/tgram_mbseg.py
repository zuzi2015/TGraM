from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging
import numpy as np
from os.path import join

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from dcn_v2 import DCN

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV3_Small(nn.Module):
    def __init__(self):
        super(MobileNetV3_Small, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.bneck1 = Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 2)
        self.bneck2 = nn.Sequential(
            Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1),
        )
        self.bneck3 = nn.Sequential(
            Block(5, 24, 96, 40, hswish(), SeModule(40), 2),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 120, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 144, 48, hswish(), SeModule(48), 1),
        )
        self.bneck4 = nn.Sequential(
            Block(5, 48, 288, 96, hswish(), SeModule(96), 2),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
        )
        self.channels = [3, 16, 16, 24, 48, 96]

    def forward(self, x):
        y = [x]
        x = self.hs1(self.bn1(self.conv1(x)))

        y.append(x)
        for i in range(4):
            x = getattr(self, 'bneck{}'.format(i + 1))(x)
            y.append(x)

        return y


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.conv = DCN(chi, cho, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x


class IDAUp(nn.Module):

    def __init__(self, o, channels, up_f):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            proj = DeformConv(c, o)
            node = DeformConv(o, o)

            up = nn.ConvTranspose2d(o, o, f * 2, stride=f,
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])


class DLAUp(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(channels[j], in_channels[j:],
                          scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]]  # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) - i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


class Interpolate(nn.Module):
    def __init__(self, scale, mode):
        super(Interpolate, self).__init__()
        self.scale = scale
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=False)
        return x


class MBV3Seg(nn.Module):
    def __init__(self, heads, down_ratio, final_kernel,
                 last_level, head_conv, out_channel=0):
        super(MBV3Seg, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        self.base = MobileNetV3_Small()
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)

        if out_channel == 0:
            out_channel = channels[self.first_level]

        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level],
                            [2 ** i for i in range(self.last_level - self.first_level)])

        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(channels[self.first_level], head_conv,
                              kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes,
                              kernel_size=final_kernel, stride=1,
                              padding=final_kernel // 2, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(channels[self.first_level], classes,
                               kernel_size=final_kernel, stride=1,
                               padding=final_kernel // 2, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

        self.out_channel = 96

        self.mid_channel = int(self.out_channel / 4)
        self.feat_cur = nn.Conv2d(self.out_channel, self.mid_channel, kernel_size=1)
        self.feat_prev = nn.Conv2d(self.out_channel, self.mid_channel, kernel_size=1)
        self.linear_e = nn.Linear(self.mid_channel, self.mid_channel, bias=False)

        self.linear_graph = nn.Linear(self.mid_channel, self.mid_channel, bias=False)
        self.graph_act = nn.Sigmoid()
        self.gate = nn.Conv2d(self.mid_channel, 1, kernel_size=1, bias=False)
        self.gate_s = nn.Sigmoid()
        self.feat_back = nn.Conv2d(self.mid_channel, self.out_channel, kernel_size=1)

        self.conv_fusion = nn.Conv2d(self.out_channel * 2, self.out_channel, kernel_size=3,
                                     stride=1, dilation=1, padding=1, bias=True)
        self.conv_u = nn.Conv2d(self.out_channel * 2, self.out_channel, kernel_size=1)
        self.p_fcn = nn.Sequential(
            nn.Conv2d(self.out_channel, self.out_channel, kernel_size=(3, 3),
                      stride=1, dilation=(1, 1), padding=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channel, self.out_channel, kernel_size=1))

    def forward(self, x):
        # print('\ninput', np.shape(x))

        feats = self.base(x[:, 0, :, :, :])
        feat_cur = feats[-1]
        # feat_cur = self.dla_up(feat_cur)
        # y = []
        # for i in range(self.last_level - self.first_level):
        #     y.append(feat_cur[i].clone())
        # self.ida_up(y, 0, len(y))
        # feat_cur = y[-1]
        # feat_inner = y[-1]

        # feat_cur = F.interpolate(feat_cur, scale_factor=0.25, mode="bilinear", align_corners=True)

        # print('\nfeat_cur', np.shape(feat_cur))

        edges = []
        _, num_frames, _, _, _ = x.shape
        # print('\nnum_frames', num_frames)
        for i in range(num_frames - 1):
            feat_prev = x[:, i + 1, :, :, :]
            # print('\nfeat_prev_ori', np.shape(feat_prev))
            feat_prev = self.base(feat_prev)[-1]

            edges.append(self.generate_attention(feat_cur, feat_prev))

        # print('\n message 1')
        # print('\nedge', np.shape(edges))
        message = self.conv_fusion(torch.cat(edges, 1))
        # message = F.interpolate(message, size=feat_inner.shape[2:], mode="bilinear", align_corners=True)

        # print('\n message 2')
        state = torch.tanh(self.conv_u(torch.cat([feat_cur, message], 1)))

        feat_new = self.p_fcn(state)

        # print('\nfeat_new', np.shape(feat_new))
        # import sys
        # sys.exit(0)

        # print('\n message 3')

        feats[-1] = feat_new

        feats = self.dla_up(feats)
        y = []
        for i in range(self.last_level - self.first_level):
            y.append(feats[i].clone())
        self.ida_up(y, 0, len(y))

        # print('\nlast feat', np.shape(y[-1]))

        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(y[-1])
        return [z]#, y[-1], feat_new

    def generate_attention(self, exemplar, query):
        exemplar = self.feat_cur(exemplar)
        query = self.feat_prev(query)

        fea_size = query.size()[2:]
        exemplar_flat = exemplar.view(-1, self.mid_channel, fea_size[0] * fea_size[1])  # N,C,H*W
        query_flat = query.view(-1, self.mid_channel, fea_size[0] * fea_size[1])  # N,C,H*W
        query_t = torch.transpose(query_flat, 1, 2).contiguous()  # N,H*W,C
        exemplar_t = torch.transpose(exemplar_flat, 1, 2).contiguous()  # batch size x dim x num = N,H*W,C
        exemplar_corr = self.linear_e(exemplar_t)  # N,H*W,C
        A = torch.bmm(exemplar_corr, query_flat)  # N,H*W,H*W

        exemplar_att = torch.bmm(A, query_t).contiguous()  # N,H*W,C
        # graph convolution
        exemplar_att = self.graph_act(self.linear_graph(exemplar_att))

        input1_att = exemplar_att.view(-1, self.mid_channel, fea_size[0], fea_size[1])
        input1_mask = self.gate(input1_att)
        input1_mask = self.gate_s(input1_mask)
        input1_att = input1_att * input1_mask

        input1_att = self.feat_back(input1_att)

        return input1_att


def tgram_mbv3seg(num_layers, heads, head_conv=256, down_ratio=4):
    model = MBV3Seg(heads,
                    down_ratio=down_ratio,
                    final_kernel=1,
                    last_level=5,
                    head_conv=head_conv)
    return model

