
import numpy as np
""""
backbone is SwinTransformer
"""
import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt
import torch.nn.functional as F
import os

# import onnx
from model.MobileViT import mobile_vit_small

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )

def conv1x1(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=has_bias)


def conv1x1_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv1x1(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


class MALNet(nn.Module):
    def __init__(self):
        super(MALNet, self).__init__()

        self.rgb = mobile_vit_small()
        self.fm1 = MFFM(64)
        self.fm2 = MFFM(96)
        self.fm3 = MFFM(128)
        self.fm4 = MFFM(160)



        self.fms = [self.fm1,self.fm2,self.fm3,self.fm4]
        self.decoder = Decode(64,64,64,64)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        fuses = []

        x = self.rgb.conv_1(x)
        x = self.rgb.layer_1(x)

        x = self.rgb.layer_2(x)
        x,fuse = self.fms[0](x)
        fuses.append(fuse)

        x = self.rgb.layer_3(x)
        x, fuse = self.fms[1](x)
        fuses.append(fuse)

        x = self.rgb.layer_4(x)
        x, fuse = self.fms[2](x)
        fuses.append(fuse)

        x = self.rgb.layer_5(x)
        x, fuse = self.fms[3](x)
        fuses.append(fuse)

        pred1,pred2,pred3,pred4 = self.decoder(fuses[0],fuses[1],fuses[2],fuses[3],352)

        return pred1,pred2,pred3,pred4,self.sig(pred1),self.sig(pred2),self.sig(pred3),self.sig(pred4)


    def load_pre(self, pre_model_r):
        self.rgb.load_state_dict(torch.load(pre_model_r))
        print(f"RGB SwinTransformer loading pre_model ${pre_model_r}")

        # self.depth.load_state_dict(torch.load(pre_model_d))
        # print(f"Depth PyramidVisionTransformerImpr loading pre_model ${pre_model_d}")

class MFFM(nn.Module):
    def __init__(self, dim_r):
        super(MFFM, self).__init__()
        self.r_att = Attention(dim_r)
        self.max_sa = SpatialAttention()
        self.mean_sa = SpatialAttention1()
        self.conv = nn.Conv2d(dim_r, 64, kernel_size=1)
        self.br1 = conv1x1(32,32)
        self.br2 = conv1x1(32, 32)
        self.conv_end = nn.Conv2d(64 * 2, 64, kernel_size=1, stride=1)
        self.weight = nn.Parameter(torch.ones(6, dtype=torch.float32), requires_grad=True)
    def forward(self, r):
        out_r = self.r_att(r)
        max_sa = self.max_sa(r)
        mean_sa = self.mean_sa(r)
        w = (max_sa-mean_sa) * (max_sa-mean_sa)
        out_r = (out_r.mul(w) + r) * (max_sa * self.weight[0] + mean_sa * self.weight[1])
        out = self.conv(out_r)
        x1,x2 = torch.chunk(out,2,dim=1)
        br1 = self.br1(x1) * (max_sa * self.weight[2] + mean_sa * self.weight[3])
        br2 = self.br2(x2) * (max_sa * self.weight[4] + mean_sa * self.weight[5])
        out = self.conv_end(torch.cat((out, br1, br2), 1))
        return out_r,out

class PM(nn.Module):
    def __init__(self, dim):
        super(PM, self).__init__()
        self.max_sa = SpatialAttention()
        self.mean_sa = SpatialAttention1()
        self.br1 = conv1x1(dim//2, dim//2)
        self.br2 = conv1x1(dim//2, dim//2)
        self.conv_end = nn.Conv2d(dim * 2, dim, kernel_size=1, stride=1)
        self.weight = nn.Parameter(torch.ones(4, dtype=torch.float32), requires_grad=True)
    def forward(self, x):
        max_sa = self.max_sa(x)
        mean_sa = self.mean_sa(x)
        x1,x2 = torch.chunk(x,2,dim=1)
        br1 = self.br1(x1) * (max_sa * self.weight[0] + mean_sa * self.weight[1])
        br2 = self.br2(x2) * (max_sa * self.weight[2] + mean_sa * self.weight[3])
        out = self.conv_end(torch.cat((x, br1, br2), 1))
        return out

class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()

        self.atten_H = H(dim)
        self.atten_W = W(dim)

        self.weight = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
    def forward(self, x):

        x1 = self.atten_W(x) * self.weight[0]
        x2 = self.atten_H(x) * self.weight[1]
        x = x * x1 * x2

        return x


class H(nn.Module):
    def __init__(self, inp, reduction=32):
        super(H, self).__init__()
        self.pool_h = nn.AdaptiveMaxPool2d((None, 1))
        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)


    def forward(self, rgb):
        x = rgb
        n, c, h, w = x.size()
        y= self.pool_h(x)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        a_h = self.conv_h(y).sigmoid()
        return a_h

class W(nn.Module):
    def __init__(self, inp, reduction=32):
        super(W, self).__init__()
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, rgb):
        x = rgb
        n, c, h, w = x.size()
        y = self.pool_w(x)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        a_w = self.conv_w(y).sigmoid()

        return a_w


class Decode(nn.Module):
    def __init__(self, in1,in2,in3,in4):
        super(Decode, self).__init__()
        self.pool = nn.AvgPool2d(2)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_dw1 = nn.Sequential(
            nn.Conv2d(in_channels=in1//2, out_channels=in2//2, kernel_size=1, bias=False),
            PM(in2//2),
            nn.GELU(),
            self.pool
        )
        self.conv_dw2 = nn.Sequential(
            nn.Conv2d(in_channels=in2, out_channels=in3//2, kernel_size=1, bias=False),
            PM(in3//2),
            nn.GELU(),
            self.pool
        )
        self.conv_dw3 = nn.Sequential(
            nn.Conv2d(in_channels=in3, out_channels=in4//2, kernel_size=1, bias=False),
            PM(in4//2),
            nn.GELU(),
            self.pool
        )
        self.conv_dw4 = nn.Sequential(
            nn.Conv2d(in_channels=in4, out_channels=in4//2, kernel_size=1, bias=False),
            PM(in4//2),
            nn.GELU(),
        )

        self.conv_dw11 = nn.Sequential(
            nn.Conv2d(in_channels=in1//2, out_channels=in2//2, kernel_size=1, bias=False),
            nn.BatchNorm2d(in2//2),
            nn.GELU(),
            self.pool
        )
        self.conv_dw21 = nn.Sequential(
            nn.Conv2d(in_channels=in2, out_channels=in3//2, kernel_size=1, bias=False),
            nn.BatchNorm2d(in3//2),
            nn.GELU(),
            self.pool
        )
        self.conv_dw31 = nn.Sequential(
            nn.Conv2d(in_channels=in3, out_channels=in4//2, kernel_size=1, bias=False),
            nn.BatchNorm2d(in4//2),
            nn.GELU(),
            self.pool
        )

        self.conv_dw41 = nn.Sequential(
            nn.Conv2d(in_channels=in4, out_channels=in4//2, kernel_size=1, bias=False),
            nn.BatchNorm2d(in4//2),
            nn.GELU(),

        )


        self.conv_up4 = nn.Sequential(
            nn.Conv2d(in_channels=in4*2, out_channels=in3, kernel_size=1, bias=False),
            PM(in3),
            nn.GELU(),
            self.upsample2
        )
        self.conv_up3 = nn.Sequential(
            nn.Conv2d(in_channels=in3*2, out_channels=in2, kernel_size=1, bias=False),
            PM(in2),
            nn.GELU(),
            self.upsample2
        )
        self.conv_up2 = nn.Sequential(
            nn.Conv2d(in_channels=in2*2, out_channels=in1, kernel_size=1, bias=False),
            PM(in1),
            nn.GELU(),
            self.upsample2
        )
        self.conv_up1 = nn.Sequential(
            nn.Conv2d(in_channels=in1*2, out_channels=in1, kernel_size=1, bias=False),
            PM(in1),
            nn.GELU(),
            self.upsample2
        )

        self.upb4 = Block(in3)
        self.upb3 = Block(in2)
        self.upb2 = Block(in1)
        self.upb1 = Block(in1)

        self.p_1 = nn.Sequential(
            nn.Conv2d(in_channels=in1, out_channels=in1//2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in1//2),
            nn.GELU(),
            self.upsample2,
            nn.Conv2d(in_channels=in1//2, out_channels=1, kernel_size=3, padding=1, bias=True),
        )

        self.p2 = nn.Conv2d(in1, 1, kernel_size=3, padding=1)
        self.p3 = nn.Conv2d(in2, 1, kernel_size=3, padding=1)
        self.p4 = nn.Conv2d(in3, 1, kernel_size=3, padding=1)
        # self.dw4p = nn.Conv2d(512, 1, kernel_size=3, padding=1)


    def forward(self,x1,x2,x3,x4,s):

        x11, x12 = torch.chunk(x1, 2, dim=1)
        x21, x22 = torch.chunk(x2, 2, dim=1)
        x31, x32 = torch.chunk(x3, 2, dim=1)
        x41, x42 = torch.chunk(x4, 2, dim=1)

        dw11 = self.conv_dw1(x11)
        dw21 = self.conv_dw2(torch.cat((x21, dw11),1))
        dw31 = self.conv_dw3(torch.cat((x31, dw21),1))
        dw41 = self.conv_dw4(torch.cat((x41, dw31),1))

        dw12 = self.conv_dw11(x12)
        dw22 = self.conv_dw21(torch.cat((x22, dw12),1))
        dw32 = self.conv_dw31(torch.cat((x32, dw22),1))
        dw42 = self.conv_dw41(torch.cat((x42, dw31),1))

        up4 = self.upb4(self.conv_up4(torch.cat((dw41,dw42,dw31,dw32),1)))
        up3 = self.upb3(self.conv_up3(torch.cat((up4,dw21,dw22),1)))
        up2 = self.upb2(self.conv_up2(torch.cat((up3,dw11,dw12),1)))
        up1 = self.upb1(self.conv_up1(torch.cat((up2,x11,x12), 1)))

        pred1 = self.p_1(up1)
        pred2 = F.interpolate(self.p2(up2), size=s, mode='bilinear')
        pred3 = F.interpolate(self.p3(up3), size=s, mode='bilinear')
        pred4 = F.interpolate(self.p4(up4), size=s, mode='bilinear')



        return pred1,pred2,pred3,pred4

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        mip = min(8, in_planes // ratio)
        self.max_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, mip, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(mip, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = self.sigmoid(max_out)

        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

class SpatialAttention1(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention1, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out= torch.mean(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x



class Block(nn.Module):
    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
        self.dim = dim
        super().__init__()

        self.branch1 =  nn.Sequential(
            conv1x1_bn_relu(dim//2,dim//4),
            nn.Conv2d(dim//4,dim//4,kernel_size=7,padding=9,groups=dim//4,dilation=3),
            conv1x1(dim//4,dim//2),
        )
        self.pm = PM(dim)
        self.branch2 = PPM(dim//2)
        self.conv = conv1x1_bn_relu(dim * 2, dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pm(x)
        x1, x2 = torch.split(x, [self.dim//2, self.dim//2], dim=1)
        x1 = self.branch2(x1)
        x2 = self.branch1(x2)
        x = self.conv(torch.cat((x,x1,x2),1))
        return x

class PPM(nn.ModuleList):

    def __init__(self,in_dim):
        super(PPM,self).__init__()
        self.ps1c = DWConvFFN(in_dim)
        self.ps2c = nn.Sequential(
            conv1x1_bn_relu(in_dim, in_dim//4),
            nn.Conv2d(in_dim//4, in_dim//4, kernel_size=3, stride=1,padding=1,groups=in_dim//4),
            conv1x1(in_dim//4,in_dim)
        )
        self.ps3c = nn.Sequential(
            conv1x1_bn_relu(in_dim, in_dim//4),
            nn.Conv2d(in_dim//4, in_dim//4, kernel_size=5, stride=1,padding=2,groups=in_dim//4),
            nn.Conv2d(in_dim // 4, in_dim, kernel_size=1, stride=1)
        )

        self.conv_end = conv1x1_bn_relu(in_dim*4,in_dim)
    def forward(self,x):
        x1 = self.ps1c(x)
        x2 = self.ps2c(x)
        x3 = self.ps3c(x)
        out = self.conv_end(torch.cat((x,x3,x2,x1),1))

        return out



class DWConvFFN(nn.ModuleList):
    def __init__(self,dim, drop_rate=0., layer_scale_init_value=1e-6):
        super(DWConvFFN,self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
        x = self.drop_path(x)

        return x

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    unloader = torchvision.transforms.ToPILImage()
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

if __name__ == '__main__':
    import torch
    import torchvision
    from thop import profile

    model = MALNet().cuda()

    a = torch.randn(1, 3, 352, 352).cuda()
    b = torch.randn(1, 3, 384, 384).cuda()
    flops, params = profile(model, (a,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))

