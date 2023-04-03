# ConvNeXt

# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pdb
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
#from timm.models.registry import register_model

from mmcv.runner import load_checkpoint
from mmcv.cnn import build_activation_layer, build_conv_layer

from mmseg.utils import get_root_logger
from mmseg.models.builder import BACKBONES

from gaiavision.core.ops import DynamicLinear
from gaiavision.core.bricks import build_norm_layer
from gaiavision.core.ops import DynamicLayerNorm
from gaiavision.core import DynamicMixin

#! to do : Add ElasticLN MODE!

# nn.Conv2d(self,
# in_channels, out_channels, kernel_size,
# stride=1, padding=0, dilation=1, groups=1, bias=True))
# class Block(nn.Module):
class DynamicConvNeXtBlock(nn.Module, DynamicMixin):
    """ ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self,
                 dim,
                 drop_path=0.,
                 layer_scale_init_value=1e-6,
                 conv_cfg=dict(type='DynConv2d'),
                 norm_cfg=dict(type='DynLN',eps=1e-6,data_format="channels_last"),
                 act_cfg=dict(type='GELU')):
        super().__init__()
        # self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.dwconv = build_conv_layer(
            conv_cfg,
            dim,
            dim,
            kernel_size=7,
            padding=3,
            groups=dim)
        #self.norm = DynamicLayerNorm(dim, eps=1e-6)
        self.nrom_name, nrom = build_norm_layer(norm_cfg, dim, postfix=1)
        self.add_module(self.nrom_name, nrom)
        # self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.pwconv1 = DynamicLinear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        # self.act = nn.GELU()
        self.act = build_activation_layer(act_cfg)
        # self.pwconv2 = nn.Linear(4 * dim, dim)
        self.pwconv2 = DynamicLinear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    @property
    def nrom(self):
        return getattr(self, self.nrom_name)

    def manipulate_width(self, width):
        self.width_state = width
        self.dwconv.manipulate_width(width)
        self.pwconv1.manipulate_out_channels(4*width)
        self.pwconv2.manipulate_out_channels(width)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.nrom(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            gamma = self.gamma[:x.size(-1)]
            x = gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x

class DynamicBlock(nn.ModuleList, DynamicMixin):

    def __init__(self,
                 dim,
                 depth,
                 drop_path,
                 layer_scale_init_value=1e-6,
                 conv_cfg=dict(type='DynConv2d'),
                 norm_cfg=dict(type='DynLN',eps=1e-6,data_format="channels_last"),
                 act_cfg=dict(type='GELU')):

        self.depth_state = depth
        blocks = []
        for i in range(depth):
            blocks.append(
                DynamicConvNeXtBlock(
                    dim=dim,
                    drop_path=drop_path[i],
                    layer_scale_init_value=layer_scale_init_value,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        super(DynamicBlock, self).__init__(blocks)

    def manipulate_depth(self, arch_meta):
        assert arch_meta >= 1, 'Depth must be greater than 0, ' \
                           'skipping stage is not supported yet.'
        self.depth_state = arch_meta

    def manipulate_width(self, arch_meta):
        self.arch_mata = arch_meta
        for m in self:
            m.manipulate_width(arch_meta)

    def deploy_forward(self, x):
        del self[self.depth_state:]
        for i in range(self.depth_state):
            x = self[i](x)
        return x

    def forward(self, x):
        if getattr(self, '_deploying', False):
            return self.deploy(x)
        
        for i in range(self.depth_state):
            x = self[i](x)

        return x

@BACKBONES.register_module()
class DynamicConvNeXt(nn.Module, DynamicMixin):
    """ ConvNeXt
        A Dynmaic PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    """
    def __init__(self,
                 depths,
                 dims,
                 in_chans=3,
                 drop_path_rate=0.,
                 out_indices=[0, 1, 2, 3],
                 pretrained=None,
                 layer_scale_init_value=1e-6,
                 conv_cfg=dict(type='DynConv2d'),
                 Channels_first_norm_cfg=dict(type='DynLN',eps=1e-6,data_format="channels_first"),
                 Channels_last_norm_cfg=dict(type='DynLN',eps=1e-6,data_format="channels_last"),
                 act_cfg=dict(type='GELU')):
        super().__init__()
        self.out_indices = out_indices
        norm_layer = partial(DynamicLayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)


        # stem block
        self.stem = build_conv_layer(conv_cfg,in_chans,dims[0],kernel_size=4,stride=4,padding=0)
        #! data_format="channels_first"
        self.stem_ln_name, stem_ln = build_norm_layer(Channels_first_norm_cfg, dims[0], postfix=1)
        self.add_module(self.stem_ln_name, stem_ln)

        # downsample layer1
        #! data_format="channels_first"
        self.ds1_ln_name, ds1_ln = build_norm_layer(Channels_first_norm_cfg, dims[0], postfix=2)
        self.add_module(self.ds1_ln_name, ds1_ln)
        self.ds1_conv = build_conv_layer(conv_cfg,dims[0],dims[1],kernel_size=2,stride=2,padding=0)

        # downsample layer2
        #! data_format="channels_first"
        self.ds2_ln_name, ds2_ln = build_norm_layer(Channels_first_norm_cfg, dims[1], postfix=3)
        self.add_module(self.ds2_ln_name, ds2_ln)
        self.ds2_conv = build_conv_layer(conv_cfg,dims[1],dims[2],kernel_size=2,stride=2,padding=0)

        # downsample layer3
        #! data_format="channels_first"
        self.ds3_ln_name, ds3_ln = build_norm_layer(Channels_first_norm_cfg, dims[2], postfix=4)
        self.add_module(self.ds3_ln_name, ds3_ln)
        self.ds3_conv = build_conv_layer(conv_cfg,dims[2],dims[3],kernel_size=2,stride=2,padding=0)

        # drop path
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        dp_rates_stage = [dp_rates[0:depths[0]],
                            dp_rates[depths[0]:sum(depths[:2])],
                            dp_rates[sum(depths[:2]):sum(depths[:3])],
                            dp_rates[sum(depths[:3]):sum(depths[:4])]]

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.blocks = []
        for i, block_depth in enumerate(depths):
            block = self.make_dynamic_convnext_block(
                dim=dims[i],
                depth=block_depth,
                drop_path=dp_rates_stage[i],
                layer_scale_init_value=layer_scale_init_value,
                conv_cfg=conv_cfg,
                norm_cfg=Channels_last_norm_cfg,
                act_cfg=act_cfg)
            block_name = f'dynamic_convnext_block_{i+1}'
            self.add_module(block_name, block)
            self.blocks.append(block_name)

        # final norm layer
        # self.final_ln_name, final_ln = build_norm_layer(Channels_last_norm_cfg, dims[-1], postfix=5)
        # self.add_module(self.final_ln_name, final_ln)

        self.apply(self._init_weights)
        self.init_weights(pretrained)

    @property
    def stem_ln(self):
        return getattr(self, self.stem_ln_name)
    @property
    def ds1_ln(self):
        return getattr(self, self.ds1_ln_name)
    @property
    def ds2_ln(self):
        return getattr(self, self.ds2_ln_name)
    @property
    def ds3_ln(self):
        return getattr(self, self.ds3_ln_name)
    @property
    def final_ln(self):
        return getattr(self, self.final_ln_name)

    def make_dynamic_convnext_block(self, **kwargs):
        return DynamicBlock(**kwargs)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward_features(self, x):
        # x : B 3 224 224
        x = self.stem(x) # x : B C 56 56
        x = self.stem_ln(x)
        outs = []
        for i, block_name in enumerate(self.blocks):
            block = getattr(self, block_name)
            x = block(x)
            # 0 : B C 56 56
            # 1 : B 2C 28 28
            # 2 : B 4C 14 14
            # 3 : B 8C 7 7
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)
            if i == 0:
                x = self.ds1_ln(x) # x : B C 56 56
                x = self.ds1_conv(x) # x : B 2C 28 28
            elif i == 1:
                x = self.ds2_ln(x) # x : B 2C 28 28
                x = self.ds2_conv(x) # x : B 4C 14 14
            elif i == 2:
                x = self.ds3_ln(x) # x : B 4C 14 14
                x = self.ds3_conv(x) # x : B 8C 7 7
        # global average pooling, (N, C, H, W) -> (N, C)
        # x = self.final_ln(x.mean([-2, -1])) # B 8C

        return tuple(outs)

    # def manipulate_stem(self, arch_meta):
    #     self.stem_state = arch_meta
    #     self.stem.manipulate(aarch_meta)

    def manipulate_body(self, arch_meta):
        self.body_state = arch_meta
        # DL to LD
        sliced_arch_meta = [
            dict(zip(arch_meta, t)) for t in zip(*arch_meta.values())
        ]
        for i, block_name in enumerate(self.blocks):
            block = getattr(self, block_name)
            block.manipulate_arch(sliced_arch_meta[i])
            if i == 0:
                self.stem.manipulate_width(sliced_arch_meta[i]['width'])
            if i == 1:
                self.ds1_conv.manipulate_width(sliced_arch_meta[i]['width'])
            if i == 2:
                self.ds2_conv.manipulate_width(sliced_arch_meta[i]['width'])
            if i == 3:
                self.ds3_conv.manipulate_width(sliced_arch_meta[i]['width'])


    def forward(self, x):
        x = self.forward_features(x)

        return x 

model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}