import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers
from torch.nn.utils import weight_norm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x

class Scale(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

class one_conv(nn.Module):
    def __init__(self,inchanels,growth_rate,kernel_size = 3, relu = True):
        super(one_conv,self).__init__()
        wn = lambda x:torch.nn.utils.weight_norm(x)
        self.conv = nn.Conv2d(inchanels,growth_rate,kernel_size=kernel_size,padding = kernel_size>>1,stride= 1)
        self.flag = relu
        self.conv1 = nn.Conv2d(growth_rate,inchanels,kernel_size=kernel_size,padding = kernel_size>>1,stride= 1)
        if relu:
            self.relu = nn.PReLU(growth_rate)
        self.weight1 = Scale(1)
        self.weight2 = Scale(1)
    def forward(self,x):
        if self.flag == False:
            output = self.weight1(x) + self.weight2(self.conv1(self.conv(x)))
        else:
            output = self.weight1(x) + self.weight2(self.conv1(self.relu(self.conv(x))))
        return output#torch.cat((x,output),1)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

' save版本'
class SEBlock(nn.Module):   # 改进的通道注意力
    def __init__(self, mode, channels, ratio):
        super(SEBlock, self).__init__()
        if mode == "max":
            self.global_pooling = nn.AdaptiveMaxPool2d(1)
        elif mode == "avg":
            self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=channels, out_features=channels // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=channels // ratio, out_features=channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, y):
        b, c, _, _ = x.shape
        v = self.global_pooling(x).view(b, c)
        v = self.fc_layers(v).view(b, c, 1, 1)
        v = self.sigmoid(v)
        return y * v  ## 01/13
class fe_save(nn.Module):
    def __init__(self, inp_channels, out_channels, exp_ratio=4, act_type='relu'):
        super(fe_save, self).__init__()
        self.act_type = act_type
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.conv0 = nn.Conv2d(inp_channels, out_channels * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels*2)
        self.conv1 = nn.Conv2d(int(out_channels * 2 / 4 * 1), int(out_channels * 2 / 4 * 1), kernel_size=3, stride=1, padding=1, groups=int(out_channels * 2 / 4 * 1), bias=False)
        self.bn2 = nn.BatchNorm2d(int(out_channels * 2 / 4 * 1))
        self.conv2 = nn.Conv2d(int(out_channels * 2 / 4 * 2), int(out_channels * 2 / 4 * 1), kernel_size=3, stride=1, padding=1, groups=int(out_channels * 2 / 4 * 1), bias=False)
        self.bn3 = nn.BatchNorm2d(int(out_channels * 2 / 4 * 1))
        self.conv3 = nn.Conv2d(int(out_channels * 2 / 4 * 2), int(out_channels * 2 / 4 * 1), kernel_size=3, stride=1, padding=1, groups=int(out_channels * 2 / 4 * 1), bias=False)
        self.bn4 = nn.BatchNorm2d(int(out_channels * 2 / 4 * 1))
        self.conv4 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False)
        self.bn5 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock('max', out_channels,9)
        # self.norm = nn.BatchNorm2d(int(out_channels*2))
        if self.act_type == 'linear':
            self.act = None
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'gelu':
            self.act = nn.GELU()
        else:
            raise ValueError('unsupport type of activation')
    def forward(self,x):
        x1 = self.conv0(x)
        x1 = self.act(self.bn1(x1))
        y1, y2, y3, y4 = torch.split(x1, [self.out_channels * 2 // 4, self.out_channels * 2 // 4, self.out_channels * 2 // 4,
                                          self.out_channels * 2 // 4], dim=1)
        y2 = self.conv1(y2)
        y2 = self.act(self.bn2(y2))
        y3 = self.conv2(torch.concat([y2, y3], dim=1))
        y3 = self.act(self.bn3(y3))
        y4 = self.conv3(torch.concat([y3, y4], dim=1))
        y4 = self.act(self.bn4(y4))
        y = self.conv4(torch.concat([y1, y2, y3, y4], dim=1))
        y = self.act(self.bn5(y))
        y = self.se(x, y)
        y = channel_shuffle(y, 2)   ## 在消融实验里，channel-shuffle的操作在CA之前，后面看需不需要重新跑一下  save  版本的代码
        return y
class LFE_3_save(nn.Module):
    def __init__(self, inp_channels, out_channels, exp_ratio=4, act_type='relu'):
        super(LFE_3_save, self).__init__()
        self.exp_ratio = exp_ratio
        self.act_type  = act_type

        self.conv0 = fe_save(inp_channels, out_channels)
        self.conv1 = fe_save(inp_channels, out_channels)
        self.conv2 = fe_save(inp_channels, out_channels)


        if self.act_type == 'linear':
            self.act = None
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'gelu':
            self.act = nn.GELU()
        else:
            raise ValueError('unsupport type of activation')

    def forward(self, x):
        y = self.conv0(x)
        y = self.act(y)
        y = self.conv1(y)
        y = self.act(y)
        y = self.conv2(y)
        return y


'save 版本'
class GMSA_1_save(nn.Module):
    def __init__(self, channels, shifts=4, window_sizes=[4, 8, 12], calc_attn=True):
        super(GMSA_1_save, self).__init__()
        self.channels = channels
        self.shifts   = shifts
        self.window_sizes = window_sizes
        self.calc_attn = calc_attn

        if self.calc_attn:
            self.split_chns  = [channels*2//3, channels*2//3, channels*2//3]
            self.project_inp = nn.Sequential(
                nn.Conv2d(self.channels, self.channels*2, kernel_size=1),
                nn.BatchNorm2d(self.channels*2)
            )
            self.t1 = nn.Conv2d(channels, channels*2//3, kernel_size=1)
            self.t2 = nn.Conv2d(channels, channels*2//3, kernel_size=1)
            self.project_out = nn.Conv2d(channels, channels, kernel_size=1)
        else:
            self.split_chns  = [channels//3, channels//3,channels//3]
            self.project_inp = nn.Sequential(
                nn.Conv2d(self.channels, self.channels, kernel_size=1),
                nn.BatchNorm2d(self.channels)
            )
            self.project_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x, prev_atns = None):
        b,c,h,w = x.shape
        x = self.project_inp(x)
        xs = torch.split(x, self.split_chns, dim=1)
        ys = []
        atns = []
        if prev_atns is None:
            for idx, x_ in enumerate(xs):
                wsize = self.window_sizes[idx]

                # print(x_.shape)
                if idx==1:
                    x_ = torch.cat([x_,y_],dim=1) #01/27
                    x_ = self.t1(x_)
                if idx==2:
                    x_ = torch.cat([x_,y_],dim=1) #01/27
                    x_ = self.t2(x_)

                if self.shifts > 0:
                    x_ = torch.roll(x_, shifts=(-wsize//2, -wsize//2), dims=(2,3))
                q, v = rearrange(
                    x_, 'b (qv c) (h dh) (w dw) -> qv (b h w) (dh dw) c',
                    qv=2, dh=wsize, dw=wsize
                )
                atn = (q @ q.transpose(-2, -1))
                atn = atn.softmax(dim=-1)
                y_ = (atn @ v)
                y_ = rearrange(
                    y_, '(b h w) (dh dw) c-> b (c) (h dh) (w dw)',
                    h=h//wsize, w=w//wsize, dh=wsize, dw=wsize
                )
                # print(y_.shape)
                if self.shifts > 0:
                    y_ = torch.roll(y_, shifts=(wsize//2, wsize//2), dims=(2, 3))
                ys.append(y_)
                atns.append(atn)
            y = torch.cat(ys, dim=1)
            y = self.project_out(y)
            return y, atns
        else:
            for idx, x_ in enumerate(xs):
                wsize = self.window_sizes[idx]
                if self.shifts > 0:
                    x_ = torch.roll(x_, shifts=(-wsize//2, -wsize//2), dims=(2,3))
                atn = prev_atns[idx]
                v = rearrange(
                    x_, 'b (c) (h dh) (w dw) -> (b h w) (dh dw) c',
                    dh=wsize, dw=wsize
                )
                y_ = (atn @ v)
                y_ = rearrange(
                    y_, '(b h w) (dh dw) c-> b (c) (h dh) (w dw)',
                    h=h//wsize, w=w//wsize, dh=wsize, dw=wsize
                )
                if self.shifts > 0:
                    y_ = torch.roll(y_, shifts=(wsize//2, wsize//2), dims=(2, 3))
                ys.append(y_)
            y = torch.cat(ys, dim=1)
            y = self.project_out(y)
            return y, prev_atns

class MBAB(nn.Module):
    def __init__(self, inp_channels, out_channels, exp_ratio=2, shifts=4, window_sizes=[4,8,16], shared_depth=0):
        super(MBAB, self).__init__()
        self.exp_ratio = exp_ratio
        self.shifts = shifts
        self.window_sizes = window_sizes
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.shared_depth = shared_depth

        modules_lfe = {}
        modules_gmsa = {}
        modules_lfe['lfe_0'] = LFE_3_save(inp_channels=inp_channels, out_channels=out_channels, exp_ratio=exp_ratio)
        modules_gmsa['gmsa_0'] = GMSA_1_save(channels=inp_channels, shifts=shifts, window_sizes=window_sizes, calc_attn=True)
        # for i in range(shared_depth):
        #     modules_lfe['lfe_{}'.format(i + 1)] = LFE_3_save(inp_channels=inp_channels, out_channels=out_channels,
        #                                               exp_ratio=exp_ratio)
        #     modules_gmsa['gmsa_{}'.format(i + 1)] = GMSA_1_save(channels=inp_channels, shifts=shifts,
        #                                                  window_sizes=window_sizes, calc_attn=False)
        self.modules_lfe = nn.ModuleDict(modules_lfe)
        self.modules_gmsa = nn.ModuleDict(modules_gmsa)

    def forward(self, x):
        atn = None

        for i in range(1 + self.shared_depth):
            if i == 0:  ## only calculate attention for the 1-st module
                x = self.modules_lfe['lfe_{}'.format(i)](x) + x
                y, atn = self.modules_gmsa['gmsa_{}'.format(i)](x, None)
                x = y + x
            else:
                x = self.modules_lfe['lfe_{}'.format(i)](x) + x
                y, atn = self.modules_gmsa['gmsa_{}'.format(i)](x, atn)
                x = y + x
        return x