from turtle import forward
import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops

class NAFNet(nn.Module):
    '''
    NAFNet for Super-Resolution
    '''
    def __init__(self, width=48, num_blks=16, img_channel=3, drop_path_rate=0., drop_out_rate=0., fusion_from=-1, fusion_to=10000, dual=True):
        super().__init__()
        self.dual = dual    # dual input for stereo SR (left view, right view)
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1)
        self.body = MySequential(
            *[DropPath(
                drop_path_rate, 
                BasicBlock(
                    width, 
                    fusion=(fusion_from <= i and (i+1) % 2 == 0), 
                    drop_out_rate=drop_out_rate
                )) for i in range(num_blks)]
        )

        self.conv = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1)

    def forward(self, inp):
        if self.dual:
            inp = inp.chunk(2, dim=1)
        else:
            inp = (inp, )
        feats = [self.intro(x) for x in inp]
        feats = self.body(*feats)
        [left, right] = [self.conv(x) for x in feats]
        return left, right

class BasicBlock(nn.Module):
    '''
    NAFBlock for Super-Resolution
    '''
    def __init__(self, c, fusion=False, drop_out_rate=0.):
        super().__init__()
        self.blk = NAFBlock(c, drop_out_rate=drop_out_rate)
        self.fusion = MF_Fusion(c) if fusion else None

    def forward(self, *feats):
        feats = tuple([self.blk(x) for x in feats])
        if self.fusion:
            feats = self.fusion(*feats)
        return feats


class MF_Fusion(nn.Module):
    def __init__(self, c, drop_out_rate=0.3):
        super().__init__()
        self.norm1 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.conv1 = nn.Conv2d(c, c//2, 1, 1, 0)
        #self.conv1_halfc = nn.Conv2d(c//2, c//2, 1, 1, 0)
        self.conv3 = nn.Conv2d(c//2, c//2, 3, 1, 1)
        self.conv5 = nn.Conv2d(c//2, c//2, 5, 1, 2)
        self.maxpool = nn.MaxPool2d(3, 1, 1)
        self.fin_conv = nn.Conv2d(4*c, c, 1, 1, 0)

    def forward(self, x_l, x_r):
        x_l, x_r = self.norm1(x_l), self.norm1(x_r)

        x_1_l, x_1_r = self.conv1(x_l), self.conv1(x_r)
        x_3_l, x_3_r = self.conv3(self.conv1(x_l)), self.conv3(self.conv1(x_r))
        x_5_l, x_5_r = self.conv5(self.conv1(x_l)), self.conv5(self.conv1(x_r))
        x_maxpool_l, x_maxpool_r = self.conv1(self.maxpool(x_l)), self.conv1(self.maxpool(x_r))

        x_fin = self.fin_conv(
            torch.cat([x_1_l, x_1_r, x_3_l, x_3_r, \
                x_5_l, x_5_r, x_maxpool_l, x_maxpool_r], 1)
        )

        x_fin = self.dropout1(x_fin)

        return x_fin + x_l, x_fin + x_r


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1_1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1)
        self.conv1_2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel)
        self.conv1_3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1)
        
        # Simplified Channel Attention
        self.sca = nn.SequentialCell(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv2_1 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1)
        self.conv2_2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv2_3 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()


        self.conv3_1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1)
        self.conv3_2 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1)

        self.beta = mindspore.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = mindspore.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.alpha = mindspore.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv1_3(x)
        x = self.dropout1(x) #这里的问题在于dropout放置的位置
        y = inp + x * self.beta

        x = self.conv2_1(self.norm2(x))
        x = self.conv2_2(x)
        x = self.sg(x)
        x = self.conv2_3(x)
        x = self.dropout2(x)
        y = y + x * self.gamma

        z = self.conv3_1(inp)
        z = self.sg(z)
        z = self.conv3_2(z)

        return y + z * self.alpha

class MySequential(nn.SequentialCell):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', mindspore.Parameter(torch.ones(channels)))
        self.register_parameter('bias', mindspore.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class DropPath(nn.Module):
    def __init__(self, drop_rate, module):
        super().__init__()
        self.drop_rate = drop_rate
        self.module = module

    def forward(self, *feats):
        if self.training and np.random.rand() < self.drop_rate:
            return feats

        new_feats = self.module(*feats)
        factor = 1. / (1 - self.drop_rate) if self.training else 1.

        if self.training and factor != 1.:
            new_feats = tuple([x+factor*(new_x-x) for x, new_x in zip(feats, new_feats)])
        return new_feats






class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', mindspore.Parameter(torch.ones(channels)))
        self.register_parameter('bias', mindspore.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)