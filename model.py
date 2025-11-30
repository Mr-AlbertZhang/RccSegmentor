import os
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from pvtv2 import pvt_v2_b2

try:
    from timm.models.helpers import named_apply
except Exception:
    def named_apply(fn, module):
        for m in module.modules():
            fn(m, None)
        return module




def _init_weights(module, name=None, scheme=''):
    if isinstance(module, (nn.Conv2d, nn.Conv3d)):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace=inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace=inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace=inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace=inplace)
    else:
        raise NotImplementedError(f'activation layer [{act}] is not found')
    return layer


class BasicConv2d(nn.Module):
    """Conv + BN + ReLU"""
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes,
            kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=False
        )
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def channel_shuffle(x, groups):
    b, c, h, w = x.size()
    channels_per_group = c // groups
    x = x.view(b, groups, channels_per_group, h, w)
    x = x.transpose(1, 2).contiguous()
    x = x.view(b, -1, h, w)
    return x


# --------------------- LK-GCM  --------------------- #

class LGAG(nn.Module):
    """
    Group Gated Attention (GGA) 
    g: High-level feature (f_high)
    x: Low-level feature (f_low)
    """
    def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1, activation='relu'):
        super().__init__()
        if kernel_size == 1:
            groups = 1

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size, stride=1,
                      padding=kernel_size // 2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=kernel_size, stride=1,
                      padding=kernel_size // 2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.activation = act_layer(activation, inplace=True)

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class LargeKernelGatedCascadeModule(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 3 / 5 / 7 
        self.conv_fx = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_fy = BasicConv2d(channel, channel, 5, padding=2)
        self.conv_fz = BasicConv2d(channel, channel, 7, padding=3)

        self.conv_aux1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_aux2 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_concat1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_concat2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_out = BasicConv2d(channel, channel, 3, padding=1)

        self.gga1 = LGAG(channel, channel, channel)
        self.gga2 = LGAG(channel, channel, channel)

    def forward(self, fx, fy, fz):
        # fx: H/32, fy: H/16, fz: H/8
        fx_up_16 = self.upsample(fx)             # -> H/16
        fx_up_8  = self.upsample(fx_up_16)       # -> H/8
        fy_up_8  = self.upsample(fy)             # -> H/8

        fx_branch = self.conv_fx(fx_up_16)
        fy_branch = self.conv_fy(fy_up_8)
        fz_branch = self.conv_fz(fz)

        # GGA(fx_branch, up(fx))
        f_low = self.gga1(
            fx_branch,
            self.conv_aux1(self.upsample(fx))
        )
        f_low = self.conv_concat1(f_low)

        # GGA(fy+fz, up(f_low))
        fused = fy_branch * fz_branch
        f_high = self.gga2(
            fused,
            self.conv_aux2(self.upsample(f_low))
        )
        f_high = self.conv_concat2(f_high)

        out = self.conv_out(f_high)
        return out


# --------------------- MGPU --------------------- #

class MSDC(nn.Module):
    def __init__(self, in_channels, activation='relu6',
                 groups=1, dilation=3):
        super().__init__()

        act = lambda: act_layer(activation, inplace=True)

        # Pf1
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=1, padding=0,
                      groups=groups, bias=False),
            nn.BatchNorm2d(in_channels),
            act()
        )

        def make_branch(k: int):
            pad = k // 2
            return nn.Sequential(
                nn.Conv2d(in_channels, in_channels,
                          kernel_size=(1, k),
                          padding=(0, pad),
                          bias=False),
                nn.BatchNorm2d(in_channels),
                act(),
                nn.Conv2d(in_channels, in_channels,
                          kernel_size=(k, 1),
                          padding=(pad, 0),
                          bias=False),
                nn.BatchNorm2d(in_channels),
                act(),
                nn.Conv2d(in_channels, in_channels,
                          kernel_size=k,
                          padding=dilation * pad,
                          dilation=dilation,
                          groups=groups,
                          bias=False),
                nn.BatchNorm2d(in_channels),
                act()
            )

        self.branch2 = make_branch(3)
        self.branch3 = make_branch(5)
        self.branch4 = make_branch(7)

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        p1 = self.branch1(x)
        p2 = self.branch2(x)
        p3 = self.branch3(x)
        p4 = self.branch4(x)
        return [p1, p2, p3, p4]


class MSCB(nn.Module):

    def __init__(self, in_channels, out_channels, stride,
                 kernel_sizes=[1, 3, 5],   
                 expansion_factor=2,
                 dw_parallel=True,         
                 add=True,                 # add=False -> concat (MGPU)
                 activation='relu6'):
        super().__init__()

        assert stride == 1, 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion_factor = expansion_factor
        self.add = add
        self.activation = activation
        self.n_scales = 4

        self.use_skip_connection = True

        # 1x1 conv 
        self.ex_channels = int(self.in_channels * self.expansion_factor)
        self.pconv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.ex_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_channels),
            act_layer(self.activation, inplace=True)
        )

        # MGPU
        self.msdc = MSDC(
            in_channels=self.ex_channels,
            activation=self.activation,
            groups=1,
            dilation=3
        )

        if self.add:
            self.combined_channels = self.ex_channels
        else:
            self.combined_channels = self.ex_channels * self.n_scales

        self.pconv2 = nn.Sequential(
            nn.Conv2d(self.combined_channels, self.out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.out_channels),
        )

        if self.in_channels != self.out_channels:
            self.conv1x1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False)

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        identity = x

        pout1 = self.pconv1(x)              # [B, ex_channels, H, W]
        msdc_outs = self.msdc(pout1)        # list of 4 tensors

        if self.add:
            dout = 0
            for out in msdc_outs:
                dout = dout + out
        else:
            dout = torch.cat(msdc_outs, dim=1)

        # dout = channel_shuffle(dout, gcd(self.combined_channels, self.out_channels))

        out = self.pconv2(dout)

        if self.in_channels != self.out_channels:
            identity = self.conv1x1(identity)

        out = out + identity
        return out


def MSCBLayer(in_channels, out_channels, n=1, stride=1,
              kernel_sizes=[1, 3, 5],
              expansion_factor=2,
              dw_parallel=True,
              add=True,
              activation='relu6'):
    blocks = []
    blocks.append(
        MSCB(
            in_channels, out_channels, stride,
            kernel_sizes=kernel_sizes,
            expansion_factor=expansion_factor,
            dw_parallel=dw_parallel,
            add=add,
            activation=activation
        )
    )
    for _ in range(1, n):
        blocks.append(
            MSCB(
                out_channels, out_channels, 1,
                kernel_sizes=kernel_sizes,
                expansion_factor=expansion_factor,
                dw_parallel=dw_parallel,
                add=add,
                activation=activation
            )
        )
    return nn.Sequential(*blocks)


# --------------------- L-CAM & L-SAM  --------------------- #

class LightChannelAttentionModule(nn.Module):
    """
    L-CAM: Light Channel Attention Module
    """
    def __init__(self, in_channels, reduction=16, activation='relu'):
        super().__init__()
        if in_channels < reduction:
            reduction = in_channels
        mid = in_channels // reduction

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.act = act_layer(activation, inplace=True)
        self.conv1 = nn.Conv2d(in_channels, mid, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(mid, in_channels, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg = self.avg_pool(x)
        maxv = self.max_pool(x)

        avg = self.conv2(self.act(self.conv1(avg)))
        maxv = self.conv2(self.act(self.conv1(maxv)))
        out = avg + maxv
        return self.sigmoid(out)


class LightSpatialAttentionModule(nn.Module):
    """
    L-SAM: Light Spatial Attention Module
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7, 11)
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        maxv, _ = torch.max(x, dim=1, keepdim=True)
        feat = torch.cat([avg, maxv], dim=1)
        attn = self.conv(feat)
        return self.sigmoid(attn)


# --------------------- Feature Exploration Module  --------------------- #

class FeatureExplorationModule(nn.Module):
    def __init__(self, in_channels=64, mgpu_out_channels=32):
        super().__init__()
        self.l_cam = LightChannelAttentionModule(in_channels)
        self.l_sam = LightSpatialAttentionModule(kernel_size=7)

        
        self.mgpu = MSCBLayer(
            in_channels=in_channels,
            out_channels=mgpu_out_channels,
            n=1,
            stride=1,
            kernel_sizes=[3, 5, 7],
            expansion_factor=6,
            dw_parallel=True,
            add=False,                 # Concat
            activation='relu6'
        )

        
        self.down2 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

    def forward(self, x1):
        # x1: Stage1 feature, [B, 64, H/4, W/4]
        x_cam = self.l_cam(x1) * x1
        x_sam = self.l_sam(x_cam) * x_cam
        x_mgpu = self.mgpu(x_sam)        # [B, C, H/4, W/4]
        x_low = self.down2(x_mgpu)       # [B, C, H/8, W/8]
        return x_low


# --------------------- Decoder --------------------- #

class MultiScaleFeatureAggregationDecoder(nn.Module):

    def __init__(self, encoder_channels=(64, 128, 320, 512), out_channels=32):
        super().__init__()
        _, c2, c3, c4 = encoder_channels
        self.trans2 = BasicConv2d(c2, out_channels, 1)  # x2 -> fz
        self.trans3 = BasicConv2d(c3, out_channels, 1)  # x3 -> fy
        self.trans4 = BasicConv2d(c4, out_channels, 1)  # x4 -> fx

        self.lk_gcm = LargeKernelGatedCascadeModule(out_channels)

    def forward(self, x2, x3, x4):
        x2_t = self.trans2(x2)
        x3_t = self.trans3(x3)
        x4_t = self.trans4(x4)
        x_high = self.lk_gcm(x4_t, x3_t, x2_t)  # [B, C, H/8, W/8]
        return x_high


# --------------------- main --------------------- #

class RccSegmentor(nn.Module):
    """
    RccSegmentor Architecture Overview:
      (a) Encoder: PVTv2 backbone -> x1, x2, x3, x4
      (b) Bottleneck: Feature Exploration Module (FEM) -> x_low
      (c) Decoder: Multi-scale Feature Aggregation Decoder (MFAD) -> x_high
      (d) Prediction: Output = Conv1x1(x_low + x_high)
    """
    def __init__(self, channel=64, backbone_pretrained=None):
        super().__init__()

        # (a) Backbone
        self.backbone = pvt_v2_b2()   # [64, 128, 320, 512]

    
        if backbone_pretrained is not None and os.path.isfile(backbone_pretrained):
            state_dict = torch.load(backbone_pretrained, map_location='cpu')
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            model_dict.update(state_dict)
            self.backbone.load_state_dict(model_dict)

        # (b) Feature Exploration Module
        self.fe_module = FeatureExplorationModule(
            in_channels=64,
            mgpu_out_channels=channel
        )

        # (c) Decoder
        self.decoder = MultiScaleFeatureAggregationDecoder(
            encoder_channels=(64, 128, 320, 512),
            out_channels=channel
        )

        # Final 1x1 convolution for mask prediction
        self.pred_head = nn.Conv2d(channel, 1, kernel_size=1)

    def forward(self, x):
        # PVT backbone
        x1, x2, x3, x4 = self.backbone(x)

        # Feature Exploration (x1 -> x_low)
        x_low = self.fe_module(x1)

        # Multi-scale Aggregation (x2,x3,x4 -> x_high)
        x_high = self.decoder(x2, x3, x4)

        # x_low + x_high
        fuse = x_low + x_high

        logits = self.pred_head(fuse)
        logits = F.interpolate(
            logits,
            size=x.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        return logits


if __name__ == "__main__":
    # Initialize model
    model = RccSegmentor(channel=64, backbone_pretrained=None).cuda()
    
    # Dummy input (Batch=1, Channel=3, Height=512, Width=512)
    x = torch.randn(1, 3, 512, 512).cuda()
    
    # Forward pass
    with torch.no_grad():
        y = model(x)
        
    print("input shape:", x.shape)
    print("output shape:", y.shape)
