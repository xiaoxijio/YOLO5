import argparse
import logging
import math
import sys
from copy import deepcopy
from pathlib import Path

sys.path.append('./')

import torch
import torch.nn as nn

from models.common import Conv, Bottleneck, SPP, DWConv, Focus, BottleneckCSP, Concat, NMS
from models.experimental import MixConv2d, CrossConv, C3
from utils.general import make_divisible, check_anchor_order
from utils.torch_utils import initialize_weights, time_synchronized, scale_img, fuse_conv_and_bn


class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # 类别数量
        self.no = nc + 5  # 每个锚的输出数量 nc + xywhc
        self.nl = len(anchors)  # 检测层数
        self.na = len(anchors[0]) // 2  # 锚框数量
        self.grid = [torch.zeros(1)] * self.nl  # 初始化网费
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        z = []  # 推理输出
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # 推理
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None):  # model, 通道数, 类别数
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            import yaml
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.FullLoader)

        if nc and nc != self.yaml['nc']:  # 我们的 nc是2, 模型自己的是80, 所以需要重新赋值
            print('重写 model.yaml nc=%g with nc=%g' % (self.yaml['nc'], nc))
            self.yaml['nc'] = nc  # 重写 yaml的 nc
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])

        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 128  # 2x min stride  s / x.shape[-2] 计算出输入和输出在高度（或宽度）维度上的比率，即每层的下采样步幅
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)

    def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si)
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite('img%g.jpg' % s, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        i = 1
        for m in self.model:  # 就是一层一层遍历 把值带进去再输出
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            # print('层数：',i,'特征图大小：',x.shape)
            i += 1
            y.append(x if m.i in self.save else None)  # save output
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            # b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b = mi.bias.view(m.na, -1).clone()  # 创建一个克隆避免 in-place 操作
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward

        return self


def parse_model(d, ch):
    """解析模型结构并动态构建网络层"""
    # 锚框 anchors、类别数 nc、深度和宽度的缩放比例 gd、gw(用于调整模型的规模)
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # 锚框数量 (因为锚框包含wh,所以计算一半)
    no = na * (nc + 5)  # 输出数量 = 锚框 * (类别 + 5)

    layers, save, c2 = [], [], ch[-1]  # 存储模型各层的实例, 记录需要保存输出的层索引, 输出通道数
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        # 每个层的定义包括起始层 f、重复次数 n、模块类型 m 和参数 args
        m = eval(m) if isinstance(m, str) else m
        for j, a in enumerate(args):
            try:  # 确保字符串数值可以被正确转化为 Python 变量或对象
                args[j] = eval(a) if isinstance(a, str) else a
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # 根据深度比例 gd 调整层的重复次数 n
        if m in [Conv, Bottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP, C3]:
            c1, c2 = ch[f], args[0]  # 输入输出通道数 c1 和 c2
            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2  # 根据宽度比例 gw 调整核的数量

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:  # Concat 模块将多个层的输出拼接在一起，因此 c2 为所有拼接层的通道数之和
            c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
        elif m is Detect:
            args.append([ch[x + 1] for x in f])
            if isinstance(args[1], int):  # 若 args[1] 是整数（锚框数量）
                args[1] = [list(range(args[1] * 2))] * len(f)  # 为每个检测层生成对应数量的锚框
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # 生成重复层的实例
        np = sum([x.numel() for x in m_.parameters()])  # 计算该层的参数总数
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # 层编号 i、输入层 f、重复次数 n、参数数量 np、层类型 t 及参数 args
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)
