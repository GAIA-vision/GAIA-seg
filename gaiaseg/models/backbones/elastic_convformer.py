import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import load_checkpoint
from mmcv.cnn import (build_activation_layer, build_conv_layer,
                       kaiming_init)

from mmseg.utils import get_root_logger
from mmseg.models.builder import BACKBONES
from mmcls.models.utils import to_2tuple
from mmcls.models.backbones.base_backbone import BaseBackbone

# local lib
from gaiavision.core import DynamicMixin
from gaiavision.core.ops import ElasticLinear
from gaiavision.core.bricks import build_norm_layer
from ..utils import DropPath

class ElasticFFN(nn.Module, DynamicMixin):
    def __init__(self,
                 embed_dim,
                 feedforward_channels,
                 num_fcs=2,
                 act_cfg=dict(type='GELU'),
                 drop_path=0.1,
                 dropout=0.0,
                 add_residual=True):
        super(ElasticFFN, self).__init__()
        self.embed_dim = embed_dim
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)
        
        self.layer1 = ElasticLinear(embed_dim, feedforward_channels)
        self.layer2 = ElasticLinear(feedforward_channels, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.add_residual = add_residual

    def manipulate_feedforward_channels(self, arch_meta):
        self.feedforward_channels_state = arch_meta
        self.layer1.manipulate_arch(arch_meta)
    
    def forward(self, x, residual=None):
        out = self.layer1(x)
        out = self.activate(out)
        out = self.dropout(out)
        self.layer2.feedforward_channels_state = residual.size(2)
        out = self.layer2(out)
        out = self.dropout(out)
        if not self.add_residual:
            return out
        if residual is None:
            residual = x
        out = self.drop_path(out)
        out += residual
        return out

class ElasticRelativePosition2D(nn.Module, DynamicMixin):
    def __init__(self,
                 max_relative_position,
                 num_units=64):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table_v = nn.Parameter(torch.randn(max_relative_position * 2 + 2, num_units))
        self.embeddings_table_h = nn.Parameter(torch.randn(max_relative_position * 2 + 2, num_units))
        nn.init.normal_(self.embeddings_table_v, std=0.02)
        nn.init.normal_(self.embeddings_table_h, std=0.02)

    def forward(self, length_q, length_k):
        # remove the first cls token distance computation
        length_q = length_q - 1
        length_k = length_k - 1
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        # compute the row and column distance
        distance_mat_v = (range_vec_k[None, :] // int(length_q ** 0.5 )  - range_vec_q[:, None] // int(length_q ** 0.5 ))
        distance_mat_h = (range_vec_k[None, :] % int(length_q ** 0.5 ) - range_vec_q[:, None] % int(length_q ** 0.5 ))
        # clip the distance to the range of [-max_relative_position, max_relative_position]
        distance_mat_clipped_v = torch.clamp(distance_mat_v, -self.max_relative_position, self.max_relative_position)
        distance_mat_clipped_h = torch.clamp(distance_mat_h, -self.max_relative_position, self.max_relative_position)

        # translate the distance from [1, 2 * max_relative_position + 1], 0 is for the cls token
        final_mat_v = distance_mat_clipped_v + self.max_relative_position + 1
        final_mat_h = distance_mat_clipped_h + self.max_relative_position + 1
        # pad the 0 which represent the cls token
        final_mat_v = torch.nn.functional.pad(final_mat_v, (1,0,1,0), "constant", 0)
        final_mat_h = torch.nn.functional.pad(final_mat_h, (1,0,1,0), "constant", 0)

        final_mat_v = torch.LongTensor(final_mat_v).cuda()
        final_mat_h = torch.LongTensor(final_mat_h).cuda()
        # get the embeddings with the corresponding distance
        embeddings = self.embeddings_table_v[final_mat_v] + self.embeddings_table_h[final_mat_h]

        return embeddings

class ElasticMHA(nn.Module, DynamicMixin):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 proj_drop=0.0,
                 attn_drop=0.0,
                 drop_path=0.1,
                 relative_position = False,
                 max_relative_position=14):
        super(ElasticMHA, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.w_ks = ElasticLinear(embed_dim, num_heads * 64)
        self.w_qs = ElasticLinear(embed_dim, num_heads * 64)
        self.w_vs = ElasticLinear(embed_dim, num_heads * 64)
        self.scale = 64 ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = ElasticLinear(num_heads * 64, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.relative_position = relative_position
        self.max_relative_position = max_relative_position
        self.rel_pos_embed_k = ElasticRelativePosition2D(max_relative_position, 64)
        self.rel_pos_embed_v = ElasticRelativePosition2D(max_relative_position, 64)

    def manipulate_num_heads(self, arch_meta):
        self.num_heads_state = arch_meta
        self.num_heads = arch_meta['num_heads']
        arch_meta_ktmp1 = {'num_heads': 576}
        arch_meta_qtmp1 = {'num_heads': 576}
        arch_meta_vtmp1 = {'num_heads': 576}
        arch_meta_ktmp1['num_heads'] = arch_meta['num_heads'] * 64
        arch_meta_qtmp1['num_heads'] = arch_meta['num_heads'] * 64
        arch_meta_vtmp1['num_heads'] = arch_meta['num_heads'] * 64
        self.w_ks.manipulate_arch(arch_meta_ktmp1)
        self.w_qs.manipulate_arch(arch_meta_qtmp1)
        self.w_vs.manipulate_arch(arch_meta_vtmp1)

    def forward(self, q, k, v, mask=None, residual=None):
        # q,k,v,residual : (B,N,D)
        qkv_dim, num_heads = 64, self.num_heads
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        if residual is None:
            residual = q

        # w_qs(q):(B,N,num_heads*q_dim)
        # q : (B,N,D) >> (B,N,num_heads*q_dim) >> (B,N,num_heads,q_dim)
        q = self.w_qs(q).view(sz_b, len_q, num_heads, qkv_dim).contiguous()
        k = self.w_ks(k).view(sz_b, len_k, num_heads, qkv_dim).contiguous()
        v = self.w_vs(v).view(sz_b, len_v, num_heads, qkv_dim).contiguous()
        # q : (B,N,num_heads,q_dim) >> (B,num_heads,N,q_dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)
        # attn : (B,num_heads,N,N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.relative_position:
            # r_p_k : (N,N,64)
            r_p_k = self.rel_pos_embed_k(len_k, len_k)
            attn = attn + (q.permute(2, 0, 1, 3).reshape(len_k, num_heads * sz_b, -1) @ r_p_k.transpose(2, 1)).transpose(1, 0).reshape(sz_b, num_heads, len_k, len_k)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # output : (B,N,num_heads*q_dim)
        output = (attn @ v).transpose(1,2).reshape(sz_b, len_v, -1)
        if self.relative_position:
            # r_p_v : (N,N,64)
            r_p_v = self.rel_pos_embed_v(len_v, len_v)
            # attn_1 : (N,B*nh,N)
            attn_1 = attn.permute(2, 0, 1, 3).reshape(len_v, sz_b * num_heads, -1)
            # The size of attention is (B, num_heads, N, N), reshape it to (N, B*num_heads, N) and do batch matmul with
            # the relative position embedding of V (N, N, head_dim) get shape like (N, B*num_heads, head_dim). We reshape it to the
            # same size as output (B, num_heads, N, hidden_dim)
            output = output + (attn_1 @ r_p_v).transpose(1, 0).reshape(sz_b, num_heads, len_v, -1).transpose(2,1).reshape(sz_b, len_v, -1)

        self.proj.feedforward_channels_state = residual.size(2)
        # todo : decide whether add droppath
        # output : (B,N,num_heads*q_dim) >> (B,N,D)
        output = self.proj_drop(self.proj(output))
        output = self.drop_path(output)
        output += residual

        return output

class Elastic_trans_Block(nn.Module, DynamicMixin):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 proj_drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_cfg=dict(type='ElaLN'),
                 act_cfg=dict(type='GELU'),
                 num_fcs=2,
                 relative_position=False,):
                #  norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(Elastic_trans_Block, self).__init__()
        self.embed_dim_state = dim
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, dim, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.attn = ElasticMHA(embed_dim=dim,
                               num_heads=num_heads,
                               proj_drop=proj_drop,
                               attn_drop=attn_drop,
                               drop_path=drop_path,
                               relative_position=relative_position)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, dim, postfix=2)
        self.add_module(self.norm2_name, norm2)
        feedforward_channels = int(dim * mlp_ratio)
        self.mlp = ElasticFFN(embed_dim=dim,
                              feedforward_channels=feedforward_channels,
                              num_fcs=num_fcs,
                              act_cfg=act_cfg,
                              drop_path=drop_path,
                              dropout=proj_drop)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def manipulate_width(self, arch_meta):
        self.embed_dim_state = arch_meta

    def manipulate_MHA(self, arch_meta):
        self.attn.manipulate_arch(arch_meta)

    def manipulate_FFN(self, arch_meta):
        arch_meta_tmp = {'feedforward_channels': {'feedforward_channels': 40}}
        arch_meta_tmp['feedforward_channels']['feedforward_channels'] = int(arch_meta['feedforward_channels']['feedforward_channels'] / 10 * self.embed_dim_state)
        self.mlp.manipulate_arch(arch_meta_tmp)

    def forward(self, x):
        norm_x = self.norm1(x)
        x = self.attn(norm_x, norm_x, norm_x, residual=x)
        x = self.mlp(self.norm2(x), residual=x)
        return x

class Elastic_conv_Block(nn.Module, DynamicMixin):

    search_space = {'width'}

    def __init__(self,
                 inplanes,
                 outplanes,
                 stride=1,
                 res_conv=False,
                 groups=1,
                 drop_block=None,
                 drop_path=None,
                 conv_cfg=dict(type='DynConv2d'),
                 norm_cfg=dict(type='DynBN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super(Elastic_conv_Block, self).__init__()
        self.expansion = 4
        med_planes = outplanes // self.expansion

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            med_planes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, med_planes, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.act1 = build_activation_layer(act_cfg)

        self.conv2 = build_conv_layer(
            conv_cfg,
            med_planes,
            med_planes,
            kernel_size=3,
            stride=stride,
            groups=groups,
            padding=1,
            bias=False)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, med_planes, postfix=2)
        self.add_module(self.norm2_name, norm2)
        self.act2 = build_activation_layer(act_cfg)

        self.conv3 = build_conv_layer(
            conv_cfg,
            med_planes,
            outplanes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.norm3_name, norm3 = build_norm_layer(norm_cfg, outplanes, postfix=3)
        self.add_module(self.norm3_name, norm3)
        self.act3 = build_activation_layer(act_cfg)

        if res_conv:
            self.residual_conv = build_conv_layer(
                conv_cfg,
                inplanes,
                outplanes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False)
            self.norm4_name, norm4 = build_norm_layer(norm_cfg, outplanes, postfix=4)
            self.add_module(self.norm4_name, norm4)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.norm3.weight)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    @property
    def norm4(self):
        return getattr(self, self.norm4_name)

    def manipulate_width(self, arch_meta):
        self.width_state = arch_meta
        # manipulate each layer
        self.conv1.manipulate_width(arch_meta // self.expansion)
        self.conv2.manipulate_width(arch_meta // self.expansion)
        self.conv3.manipulate_width(arch_meta)
        if self.res_conv:
            self.residual_conv.manipulate_width(arch_meta)

    def forward(self, x, x_t=None, return_x_2=True):
        residual = x

        x = self.conv1(x)
        x = self.norm1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)
        #print("conv_block，x.shape", x.shape)
        #if x_t is not None:
        #    print("conv_block，x_t.shape", x_t.shape)
        x = self.conv2(x) if x_t is None else self.conv2(x + x_t)
        x = self.norm2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.norm3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.norm4(residual)

        x += residual
        x = self.act3(x)

        if return_x_2:
            return x, x2
        else:
            return x

class Elastic_conv2trans(nn.Module, DynamicMixin):
    """ CNN feature maps -> Transformer patch embeddings
    """
    def __init__(self,
                 inplanes,
                 outplanes,
                 dw_stride,
                 conv_cfg=dict(type='DynConv2d'),
                 norm_cfg=dict(type='ElaLN'),
                 act_cfg=dict(type='GELU')):
        super(Elastic_conv2trans, self).__init__()
        self.dw_stride = dw_stride
        self.conv_project = build_conv_layer(
            conv_cfg,
            inplanes,
            outplanes,
            kernel_size=1,
            stride=1,
            padding=0)
        self.sample_pooling = nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride)
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, outplanes, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.act = build_activation_layer(act_cfg)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def forward(self, x, x_t):
        x = self.conv_project(x)  # [N, C, H, W]
        x = self.sample_pooling(x).flatten(2).transpose(1, 2)
        x = self.norm1(x)
        x = self.act(x)
        x = torch.cat([x_t[:, 0][:, None, :], x], dim=1)

        return x

class Elastic_trans2conv(nn.Module, DynamicMixin):
    """ Transformer patch embeddings -> CNN feature maps
    """

    def __init__(self,
                 inplanes,
                 outplanes,
                 up_stride,
                 conv_cfg=dict(type='DynConv2d'),
                 norm_cfg=dict(type='DynBN', requires_grad=True),
                 act_cfg=dict(type='ReLU')):
        super(Elastic_trans2conv, self).__init__()
        self.up_stride = up_stride
        self.conv_project = build_conv_layer(
            conv_cfg,
            inplanes,
            outplanes,
            kernel_size=1,
            stride=1,
            padding=0)
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, outplanes, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.act = build_activation_layer(act_cfg)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def forward(self, x, H, W):
        B, _, C = x.shape
        # [N, 197, 384] -> [N, 196, 384] -> [N, 384, 196] -> [N, 384, 14, 14]
        x_r = x[:, 1:].transpose(1, 2).reshape(B, C, H, W)
        x_r = self.act(self.norm1(self.conv_project(x_r)))

        return F.interpolate(x_r, size=(H * self.up_stride, W * self.up_stride))

class Elastic_ConvTrans_Block(nn.Module, DynamicMixin):
    """
    Basic module for ConvTransformer, keep feature maps for CNN block and patch embeddings for transformer encoder block
    """

    search_space = {'convblock', 'embed_dim', 'transblock'}

    def __init__(self,
                 inplanes,
                 outplanes,
                 stage,
                 res_conv,
                 stride,
                 dw_stride,
                 embed_dim,
                 num_heads,
                 mlp_ratio,
                 proj_drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 last_fusion=False,
                 groups=1,
                 conv_cfg=dict(type='DynConv2d'),
                 relative_position=False):
        super(Elastic_ConvTrans_Block, self).__init__()
        self.stage = stage
        if stage:
            self.conv_1 = Elastic_conv_Block(inplanes=inplanes,
                                         outplanes=outplanes,
                                         res_conv=True,
                                         stride=1)
            self.trans_patch_conv = build_conv_layer(
                conv_cfg,
                inplanes,
                embed_dim,
                kernel_size=dw_stride,
                stride=dw_stride,
                padding=0)
            self.trans_1 = Elastic_trans_Block(dim=embed_dim,
                                            num_heads=num_heads,
                                            mlp_ratio=mlp_ratio,
                                            proj_drop=proj_drop,
                                            attn_drop=attn_drop,
                                            drop_path=drop_path,
                                            relative_position=relative_position)
        else:
            expansion = 4
            self.cnn_block = Elastic_conv_Block(
                inplanes=inplanes,
                outplanes=outplanes,
                res_conv=res_conv,
                stride=stride,
                groups=groups)
            if last_fusion:
                self.fusion_block = Elastic_conv_Block(
                    inplanes=outplanes,
                    outplanes=outplanes,
                    stride=2,
                    res_conv=True,
                    groups=groups)
            else:
                self.fusion_block = Elastic_conv_Block(
                    inplanes=outplanes,
                    outplanes=outplanes,
                    groups=groups)
            self.squeeze_block = Elastic_conv2trans(
                inplanes=outplanes // expansion,
                outplanes=embed_dim,
                dw_stride=dw_stride)
            self.expand_block = Elastic_trans2conv(
                inplanes=embed_dim,
                outplanes=outplanes // expansion,
                up_stride=dw_stride)
            self.trans_block = Elastic_trans_Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
                drop_path=drop_path,
                relative_position=relative_position)
            self.dw_stride = dw_stride
            self.embed_dim = embed_dim
            self.last_fusion = last_fusion
            self.expansion = expansion

    def manipulate_convblock(self, arch_meta):
        if self.stage:
            self.conv_1.manipulate_arch(arch_meta)
        else:
            self.cnn_block.manipulate_arch(arch_meta)
            self.fusion_block.manipulate_arch(arch_meta)
            arch_meta_tmp = arch_meta.copy()
            arch_meta_tmp['width'] //= self.expansion
            self.expand_block.conv_project.manipulate_arch(arch_meta_tmp)

    def manipulate_embed_dim(self, arch_meta):
        self.embed_dim = arch_meta
        if self.stage:
            self.trans_patch_conv.manipulate_arch(arch_meta)
            # 
            self.trans_1.manipulate_arch(arch_meta)
        else:
            self.squeeze_block.conv_project.manipulate_arch(arch_meta)
            #
            self.trans_block.manipulate_arch(arch_meta)

    def manipulate_transblock(self, arch_meta):
        if self.stage:
            self.trans_1.manipulate_arch(arch_meta)
        else:
            self.trans_block.manipulate_arch(arch_meta)

    def forward(self, x, x_t=None, cls_tokens=None):
        # only 1st layer in stage1
        if self.stage:
            x_base = x
            x = self.conv_1(x_base, return_x_2=False) # bs 64 56 56
            x_t = self.trans_patch_conv(x_base).flatten(2).transpose(1, 2) # bs 384 14 14 --> bs 196 384
            x_t = torch.cat([cls_tokens[:,:,:x_t.size(2)], x_t], dim=1) # bs 197 384
            x_t = self.trans_1(x_t) # bs 197 384

            return x, x_t
        else:
            x, x2 = self.cnn_block(x)
            _, _, H, W = x2.shape
            x_st = self.squeeze_block(x2, x_t)
            x_t = self.trans_block(x_st + x_t)
            x_t_r = self.expand_block(x_t, H // self.dw_stride, W // self.dw_stride)
            #print("x.shape",x.shape)
            #print("x_t_r.shape", x_t_r.shape)
            x = self.fusion_block(x, x_t_r, return_x_2=False)
            #print("fusion")
            return x, x_t

class Elastic_Block(nn.ModuleList, DynamicMixin):

    search_space = {'depth','block'}
    
    def init_state(self, depth=None, block=None, **kwargs):
        if depth is not None:
            self.depth_state = depth
        if block is not None:
            self.block_state = block
        for k,v in kwargs.items():
            setattr(self, f'{k}_state', v)
            
    def __init__(self,
                 inplanes,
                 outplanes,
                 depth,
                 stage,
                 dw_stride,
                 embed_dim,
                 num_heads,
                 mlp_ratio,
                 proj_drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 last_fusion=False,
                 groups=1,
                 relative_position=False):

        self.depth_state = depth
        blocks = []
        for i in range(depth):
            if stage != 1 and stage !=4 and i == 0:
                stride = 2
            else:
                stride = 1

            if i > 0 or stage == 4:
                res_conv = False
            else:
                res_conv = True

            if stage == 1 and i == 0:
                if_first = True
            else:
                if_first = False

            # if stage == 1 or stage == 4:
            if stage == 4:
                inplanes_tmp = inplanes
            else:
                if i == 0:
                    inplanes_tmp = inplanes[0]
                else:
                    inplanes_tmp = inplanes[1]

            if stage != 4:
                drop_path_tmp = drop_path[i]
            else:
                drop_path_tmp = drop_path

            blocks.append(
                Elastic_ConvTrans_Block(
                    inplanes=inplanes_tmp,
                    outplanes=outplanes,
                    stage=if_first,
                    res_conv=res_conv,
                    stride=stride,
                    dw_stride=dw_stride,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    proj_drop=proj_drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path_tmp,#drop_path[i],
                    last_fusion=last_fusion,
                    groups=groups,
                    relative_position=relative_position))
        super(Elastic_Block, self).__init__(blocks)

    def manipulate_depth(self, arch_meta):
        assert arch_meta >= 1, 'Depth must be greater than 0, ' \
                           'skipping stage is not supported yet.'
        self.depth_state = arch_meta

    def manipulate_block(self, arch_meta):
        self.arch_meta = arch_meta
        for m in self:
            m.manipulate_arch(arch_meta)

    def deploy_forward(self, x, x_t=None, cls_tokens=None):
        # remove unused layers based on depth_state
        del self[self.depth_state:]
        for i in range(self.depth_state):
            # if x_t==None:
                # print(i,"   before:     ",x.size(),"     ",x_t,"     ",cls_tokens)
            # else:
                # print(i,"   before:     ",x.size(),"     ",x_t.size(),"     ",cls_tokens)
            x, x_t = self[i](x=x, x_t=x_t, cls_tokens=cls_tokens)
            # print(i,"   after:     ",x.size(),"     ",x_t.size(),"     ",cls_tokens)
        return x, x_t

    def forward(self, x, x_t=None, cls_tokens=None):
        if getattr(self, '_deploying', False):
            return self.deploy_forward(x=x, x_t=x_t, cls_tokens=cls_tokens)

        for i in range(self.depth_state):
            # if x_t==None:
            #     print(i,"   before:     ",x.size(),"     ",x_t,"     ",cls_tokens)
            # else:
            #     print(i,"   before:     ",x.size(),"     ",x_t.size(),"     ",cls_tokens)
            x, x_t = self[i](x=x, x_t=x_t, cls_tokens=cls_tokens)
            # print(i,"   after:     ",x.size(),"     ",x_t.size(),"     ",cls_tokens)
        return x, x_t

@BACKBONES.register_module()
class ElasticConvformer(BaseBackbone, DynamicMixin):

    search_space = {'stem', 'body'}
    
    def init_state(self, stem=None, body=None, **kwargs):
        if stem is not None:
            self.stem_state = stem
        if body is not None:
            self.body_state = body
        for k,v in kwargs.items():
            setattr(self, f'{k}_state', v)

    def __init__(self,
                 embed_dim,
                 num_heads,
                 mlp_ratio,
                 depth,
                 stem_width,
                 body_width,#=[64,128,256],
                 body_depth,#=[3,4,4],
                 patch_size=16,
                 in_chans=3,
                 proj_drop=0.,
                 attn_drop=0.,
                 drop_path=0.1,
                 conv_cfg=dict(type='DynConv2d'),
                 norm_cfg=dict(type='DynBN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 pretrained=None,
                 init_cfg=None,
                 relative_position=False):

        # Transformer
        super(ElasticConvformer,self).__init__()

        self.pretrained = pretrained
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
        else:
            raise TypeError('pretrained must be a str or None')

        self.embed_dim = embed_dim  # num_features for consistency with other models
        self.body_depth = body_depth
        assert depth % 3 == 0
        self.norm_eval =True
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.trans_dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.trans_dpr_stage = [self.trans_dpr[0:body_depth[0]],
                                self.trans_dpr[body_depth[0]:body_depth[0]+body_depth[1]],
                                self.trans_dpr[body_depth[0]+body_depth[1]:body_depth[0]+body_depth[1]+body_depth[2]],
                                self.trans_dpr[body_depth[0]+body_depth[1]+body_depth[2]]
                                ]

        self.init_state(stem={'width':stem_width},
                        body={'depth': body_depth,
                              'block': {
                                  'convblock': {
                                      'width': body_width},
                                  'embed_dim': {
                                      'width': embed_dim},
                                  'transblock': {
                                      'MHA': {
                                          'num_heads': {
                                              'num_heads': (num_heads,num_heads,num_heads)}},
                                      'FFN': {
                                          'feedforward_channels': {
                                              'feedforward_channels': (mlp_ratio,mlp_ratio,mlp_ratio)}}}}})

        # Stem stage
        self.conv1 = build_conv_layer(
            conv_cfg,
            in_chans,
            stem_width,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False) # 1 / 2 [112, 112]
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, stem_width, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.act1 = build_activation_layer(act_cfg)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # 1 / 4 [56, 56]

        trans_dw_stride = patch_size // 4
        dw_stride = [trans_dw_stride, trans_dw_stride // 2, trans_dw_stride // 4]

        self.layers = []
        self.inplanes = [stem_width,body_width[0]]
        for i, num_blocks in enumerate(self.body_depth):
            layer = self.make_layer(
                inplanes=self.inplanes,
                outplanes=body_width[i], 
                depth=num_blocks,
                stage=i+1,
                dw_stride=dw_stride[i],
                embed_dim=embed_dim,
                num_heads=num_heads[i],
                mlp_ratio=mlp_ratio[i],
                proj_drop=0.,
                attn_drop=0.,
                drop_path=self.trans_dpr_stage[i],
                last_fusion=False,
                groups=1)
            if i < 2:
                self.inplanes = [body_width[i],body_width[i+1]]
            layer_name = f'conv_trans_{i+1}'
            self.add_module(layer_name, layer)
            self.layers.append(layer_name)

        last_layer = self.make_layer(
                inplanes=body_width[2],
                outplanes=body_width[2], 
                depth=1,
                stage=4,
                dw_stride=dw_stride[2],
                embed_dim=embed_dim,
                num_heads=num_heads[2],
                mlp_ratio=mlp_ratio[2],
                proj_drop=0.,
                attn_drop=0.,
                drop_path=self.trans_dpr_stage[-1],
                last_fusion=True,
                groups=1,
                relative_position=relative_position)
        last_layer_name = f'conv_trans_{4}'
        self.add_module(last_layer_name, last_layer)
        self.layers.append(last_layer_name)

        nn.init.normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        self.init_weights()

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    def init_weights(self):
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, self.pretrained, strict=False, logger=logger, map_location='cpu')
        elif self.pretrained is None:
            self.apply(self._init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    # not use currently
    def make_stem_layer(self, in_channels, stem_width):
        # Stem stage: get the feature maps by conv block (copied form resnet.py)
        # stem_width : 64
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            stem_width,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False) # 1 / 2 [112, 112]
        self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, stem_width, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # 1 / 4 [56, 56]

    def make_layer(self, **kwargs):
        return Elastic_Block(**kwargs)

    def manipulate_stem(self, arch_meta):
        self.stem_state = arch_meta
        self.conv1.manipulate_arch(arch_meta)

    def manipulate_body(self, arch_meta):
        self.body_state = arch_meta
        for i, layer_name in enumerate(self.layers[:-1]):
            arch_meta_stage_tmp = {'depth':1,
                    'block':{
                        'convblock':{
                            'width':80},
                        'embed_dim':{
                            'width':576},
                        'transblock':{
                            'MHA':{
                                'num_heads':{'num_heads':9}},
                            'FFN':{
                                'feedforward_channels':{'feedforward_channels':40}}}}}
            arch_meta_stage_tmp['depth'] = arch_meta['depth'][i]
            arch_meta_stage_tmp['block']['convblock']['width'] = arch_meta['block']['convblock']['width'][i]
            arch_meta_stage_tmp['block']['embed_dim']['width'] = arch_meta['block']['embed_dim']['width']
            arch_meta_stage_tmp['block']['transblock']['MHA']['num_heads']['num_heads'] = arch_meta['block']['transblock']['MHA']['num_heads']['num_heads'][i]
            arch_meta_stage_tmp['block']['transblock']['FFN']['feedforward_channels']['feedforward_channels'] = arch_meta['block']['transblock']['FFN']['feedforward_channels']['feedforward_channels'][i]

            layer = getattr(self, layer_name)
            layer.manipulate_arch(arch_meta_stage_tmp)
            if i == 2:
                arch_meta_last_layer_tmp = {'depth':1,
                    'block':{
                        'convblock':{
                            'width':80},
                        'embed_dim':{
                            'width':576},
                        'transblock':{
                            'MHA':{
                                'num_heads':{'num_heads':9}},
                            'FFN':{
                                'feedforward_channels':{'feedforward_channels':40}}}}}
                arch_meta_last_layer_tmp['depth'] = 1
                arch_meta_last_layer_tmp['block']['convblock']['width'] = arch_meta['block']['convblock']['width'][i]
                arch_meta_last_layer_tmp['block']['embed_dim']['width'] = arch_meta['block']['embed_dim']['width']
                arch_meta_last_layer_tmp['block']['transblock']['MHA']['num_heads']['num_heads'] = arch_meta['block']['transblock']['MHA']['num_heads']['num_heads'][i]
                arch_meta_last_layer_tmp['block']['transblock']['FFN']['feedforward_channels']['feedforward_channels'] = arch_meta['block']['transblock']['FFN']['feedforward_channels']['feedforward_channels'][i]

                last_layer_name = f'conv_trans_{4}'
                last_layer = getattr(self, last_layer_name)
                last_layer.manipulate_arch(arch_meta_last_layer_tmp)

    def forward(self, x):
        output = []
        # x : bs 3 224 224
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1) # bs 1 embed_dim
        x_base = self.maxpool(self.act1(self.bn1(self.conv1(x)))) # bs 64 56 56

        # 2~4 stage
        #  x : 128 64 56 56, x_t : 128 197 384
        # 5~8 stage
        #  x : 128 128 28 28, x_t : 128 197 384
        # 9~12 stage
        #  x : 128 256 14 14, x_t : 128 197 384
        #import pdb; pdb.set_trace()
        x, x_t = eval('self.conv_trans_' + str(1))(x=x_base, cls_tokens=cls_tokens)
        #import pdb; pdb.set_trace()
        output.append(x)
        # print("conv_trans_1:  ",x_base.size(),"     ",x.size(),"     ",x_t.size())
        for i in range(1, len(self.body_depth)):
            #import pdb; pdb.set_trace()
            x, x_t = eval('self.conv_trans_' + str(i+1))(x=x, x_t=x_t)
            output.append(x)
            # if i == 1: print("conv_trans_2:  ",x_base.size(),"     ",x.size(),"     ",x_t.size())
            # if i == 2: print("conv_trans_3:  ",x_base.size(),"     ",x.size(),"     ",x_t.size())
        #import pdb; pdb.set_trace()
        x, x_t = eval('self.conv_trans_' + str(4))(x=x, x_t=x_t)
        output.append(x)
        # print("conv_trans_4:  ",x_base.size(),"     ",x.size(),"     ",x_t.size())
        # return x, x_t
        # print(output[0].size(),output[1].size(),output[2].size(),output[3].size())
        return tuple(output)
    
    def _freeze_stages(self):
        self.bn1.eval()
        for m in [self.conv1, self.bn1]:
            for param in m.parameters():
                param.requires_grad = False

    def freeze_bn(self, m):
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(ElasticConvformer, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            self.apply(self.freeze_bn)