import pdb
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import load_checkpoint
from mmcv.runner import BaseModule, ModuleList, _load_checkpoint
from mmcv.cnn import (build_activation_layer, build_conv_layer,
                       kaiming_init)
from mmcv.cnn.utils.weight_init import (constant_init, trunc_normal_,
                                        trunc_normal_init)

from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger
from mmseg.ops import resize
from mmcls.models.utils import to_2tuple
from mmcls.models.backbones.base_backbone import BaseBackbone

# local lib
from gaiavision.core import DynamicMixin
from gaiavision.core.ops import ElasticLinear
from gaiavision.core.bricks import build_norm_layer

from gaiaseg.models.utils import DropPath

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
        # x, residual : (B,N,D)
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
                 relative_position = True,
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
        arch_meta_ktmp1 = arch_meta.copy()
        arch_meta_qtmp1 = arch_meta.copy()
        arch_meta_vtmp1 = arch_meta.copy()
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
            #assert 1==2
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

class ElasticTransformerEncoderLayer(nn.Module, DynamicMixin):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 feedforward_channels,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 drop_path=0.1,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='DynLN',eps=1e-6,data_format="channels_last"),
                 num_fcs=2,
                 relative_position = False):
        super(ElasticTransformerEncoderLayer, self).__init__()
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, embed_dim, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.attn = ElasticMHA(embed_dim=embed_dim,
                               num_heads=num_heads,
                               proj_drop=0.0,
                               attn_drop=0.0,
                               drop_path=drop_path,
                               relative_position = relative_position)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, embed_dim, postfix=2)
        self.add_module(self.norm2_name, norm2)
        self.mlp = ElasticFFN(embed_dim=embed_dim, feedforward_channels=feedforward_channels, num_fcs=num_fcs, act_cfg=act_cfg, drop_path=drop_path, dropout=proj_drop)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        norm_x = self.norm1(x)
        x = self.attn(norm_x, norm_x, norm_x, residual=x)
        #x(B,N,D)
        x = self.mlp(self.norm2(x), residual=x)
        return x

class ElasticEncoder(nn.ModuleList, DynamicMixin):

    def init_state(self, num_layers=None, num_heads=None, feedforward_channels=None, **kwargs):
        if num_layers is not None:
            self.num_layers_state = num_layers
        if num_heads is not None:
            self.num_heads_state = num_heads
        if feedforward_channels is not None:
            self.feedforward_channels_state = feedforward_channels
        for k,v in kwargs.items():
            setattr(self, f'{k}_state', v)

    def __init__(self,
                 embed_dim,
                 num_heads,
                 feedforward_channels,
                 num_layers,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 drop_path=0.0,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='DynLN',eps=1e-6,data_format="channels_last"),
                 num_fcs=2,
                 relative_position=False):
        self.embed_dim_state = embed_dim
        self.init_state(num_layers=num_layers,
                        num_heads={
                            'num_heads': {
                                'num_heads': num_heads}},
                        feedforward_channels={
                            'feedforward_channels': {
                                'feedforward_channels': feedforward_channels}})
        layers = []
        # dpr = [x.item() for x in torch.linspace(0, drop_path, num_layers)]  # stochastic depth decay rule
        for i in range(num_layers):
            layers.append(
                ElasticTransformerEncoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    feedforward_channels=feedforward_channels,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    drop_path=drop_path[i],
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    num_fcs=num_fcs,
                    relative_position=relative_position))
        super(ElasticEncoder, self).__init__(layers)

    def manipulate_num_layers(self, arch_meta):
        # print("manipulate_num_layers: ",arch_meta,"\n")
        assert arch_meta >= 1, 'Depth must be greater than 0, ' \
                           'skipping stage is not supported yet.'
        self.num_layers_state = arch_meta

    def manipulate_embed_dim(self, arch_meta):
        # print("manipulate_embed_dim: ",arch_meta,"\n")
        self.embed_dim_state = arch_meta

    def manipulate_num_heads(self, arch_meta):
        # print("manipulate_num_heads: ",arch_meta,"\n")
        self.num_heads_state = arch_meta
        for i, m in enumerate(self):
            arch_meta_tmp = {'num_heads': {'num_heads': 7}}
            arch_meta_tmp['num_heads']['num_heads'] = \
                arch_meta['num_heads']['num_heads'][i]
            # print("manipulate_num_heads: ",i,arch_meta_tmp,"\n")
            m.attn.manipulate_arch(arch_meta_tmp)

    def manipulate_feedforward_channels(self, arch_meta):
        # print("manipulate_feedforward_channels: ",arch_meta,"\n")
        self.feedforward_channels_state = arch_meta
        for i, n in enumerate(self):
            arch_meta_tmp = {'feedforward_channels': {'feedforward_channels': 40}}
            arch_meta_tmp['feedforward_channels']['feedforward_channels'] = \
                int(arch_meta['feedforward_channels']['feedforward_channels'][i] / 10 * self.embed_dim_state)
            # print("manipulate_feedforward_channels: ",i,arch_meta_tmp,"\n")
            n.mlp.manipulate_arch(arch_meta_tmp)

    def deploy_forward(self, x):
        # remove unused layers based on num_layers_state
        del self[self.num_layers_state:]
        for i in range(self.num_layers_state):
            x = self[i](x)
        return x

    def forward(self, x, H, W, patch_size, out_indices, has_cls_token=True):
        if getattr(self, '_deploying', False):
            return self.deploy_forward(x)
        #pdb.set_trace()
        #print("out_indices: ", out_indices)
        for idx,each in enumerate(out_indices):
            out_indices[idx] = each if each>=0 else self.num_layers_state+each
        outs = []
        for i in range(self.num_layers_state):
            x = self[i](x)

            if i in out_indices:
                if has_cls_token: # TODO: optional for cls_token
                    out = x[:, 1:]
                else:
                    out = x
                B, _, C = out.shape
                out = out.reshape(B,H//patch_size,W//patch_size,-1)
                out = out.permute(0,3,1,2).contiguous() # [N,C,H,W]
                outs.append(out)        
        #print("len: ", len(outs))
        return outs

class ElasticPatchEmbed(nn.Module, DynamicMixin):
    def __init__(self,
                 embed_dim,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 conv_cfg=dict(type='ElasticConv2d')):
        super(ElasticPatchEmbed, self).__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        num_patches = (self.img_size[1] // self.patch_size[1]) * (self.img_size[0] // self.patch_size[0])
        self.num_patches = num_patches

        # Use conv layer to embed
        self.projection = build_conv_layer(conv_cfg,in_channels,embed_dim,kernel_size=patch_size,stride=patch_size)

        self.init_weights()

    def init_weights(self):
        kaiming_init(self.projection, mode='fan_in', nonlinearity='linear')

    def forward(self, x):
        # embedding: x : B,C,H,W >> B,N,D, N=H*W/P/P, D is embid_dim
        B, C, H, W = x.shape
        # FIXME: look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't " \
            f'match model ({self.img_size[0]}*{self.img_size[1]}).'
        x = self.projection(x).flatten(2).transpose(1, 2)
        return x

@BACKBONES.register_module()
class ElasticTransformer1(BaseBackbone, DynamicMixin):
    search_space = {'embedding','encoder'}
    
    def init_state(self,embedding=None, encoder=None, **kwargs):
        if embedding is not None:
            self.embedding_state = embedding
        if encoder is not None:
            self.encoder_state = encoder
        for k,v in kwargs.items():
            setattr(self, f'{k}_state', v)
            
    def __init__(self,
                 embed_dim,
                 num_heads,
                 feedforward_channels,
                 num_layers,
                 patch_size=16,
                 img_size=224,
                 in_channels=3,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 drop_rate=0.0,
                 drop_path = 0.1,
                 norm_cfg=dict(type='DynLN',eps=1e-6,data_format="channels_last"),
                 act_cfg=dict(type='GELU'),
                 num_fcs=2,
                 relative_position=False,
                 out_indices=-1,
                 final_norm=False,
                 init_cfg=None,
                 pretrained=None,
                 interpolate_mode='bicubic'):
        super(ElasticTransformer1, self).__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        if isinstance(out_indices, int):
            if out_indices == -1:
                out_indices = num_layers - 1
            self.out_indices = [out_indices]
        elif isinstance(out_indices, list) or isinstance(out_indices, tuple):
            self.out_indices = out_indices
        else:
            raise TypeError('out_indices must be type of int, list or tuple')
        self.interpolate_mode = interpolate_mode
        self.final_norm = final_norm
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.feedforward_channels = feedforward_channels
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.img_size = img_size
        self.elastic_patch_embed = ElasticPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim)
        num_patches = self.elastic_patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        self.init_state(embedding={'embed_dim': embed_dim},
                        encoder={'num_layers': num_layers,
                                 'num_heads': {'num_heads': {'num_heads': [num_heads,num_heads,num_heads]}}, 
                                 'feedforward_channels': {'feedforward_channels': {'feedforward_channels': [feedforward_channels,feedforward_channels,feedforward_channels]}}})

        # dpr_tmp = [x.item() for x in torch.linspace(0, drop_path, sum(num_layers))]  # stochastic depth decay rule
        # dpr = [dpr_tmp[0:num_layers[0]],
        #        dpr_tmp[num_layers[0]:num_layers[0]+num_layers[1]],
        #        dpr_tmp[num_layers[0]+num_layers[1]:num_layers[0]+num_layers[1]+num_layers[2]]]
        dpr = [x.item() for x in torch.linspace(0, drop_path, num_layers)]  # stochastic depth decay rule

        self.layers = []
        # for i, num_layer in enumerate(self.num_layers):
        #     elastic_encoder = self.make_encoder(
        #                 embed_dim=embed_dim,
        #                 num_heads=num_heads,
        #                 feedforward_channels=feedforward_channels,
        #                 num_layers=num_layer,
        #                 attn_drop=attn_drop,
        #                 proj_drop=proj_drop,
        #                 drop_path=dpr[i],
        #                 act_cfg=act_cfg,
        #                 norm_cfg=norm_cfg,
        #                 num_fcs=num_fcs)
        #     elastic_encoder_name = f'elastic_encoder{i + 1}'
        #     self.add_module(elastic_encoder_name, elastic_encoder)
        #     self.layers.append(elastic_encoder_name)
        elastic_encoder = self.make_encoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            feedforward_channels=feedforward_channels,
            num_layers=num_layers,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=dpr,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            num_fcs=num_fcs,
            relative_position=relative_position)
        elastic_encoder_name = f'elastic_encoder'
        self.add_module(elastic_encoder_name,elastic_encoder)
        self.layers.append(elastic_encoder_name)

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, embed_dim, postfix=1)
        self.add_module(self.norm1_name, norm1)

        self.apply(self._init_weights)
        self.init_weights()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.normal_(m.bias, std=1e-6)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def resize_pos_embed(pos_embed, input_shpae, pos_shape, mode):
        """Resize pos_embed weights.
        Resize pos_embed using bicubic interpolate method.
        Args:
            pos_embed (torch.Tensor): Position embedding weights.
            input_shpae (tuple): Tuple for (downsampled input image height,
                downsampled input image width).
            pos_shape (tuple): The resolution of downsampled origin training
                image.
            mode (str): Algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'``. Default: ``'nearest'``
        Return:
            torch.Tensor: The resized pos_embed of shape [B, L_new, C]
        """
        assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
        pos_h, pos_w = pos_shape
        cls_token_weight = pos_embed[:, 0]
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w):]
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, pos_embed.shape[2]).permute(0, 3, 1, 2)
        #pdb.set_trace()
        pos_embed_weight = resize(
            pos_embed_weight, size=input_shpae, align_corners=False, mode=mode)
        cls_token_weight = cls_token_weight.unsqueeze(1)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
        return pos_embed

    def init_weights(self):
        if (isinstance(self.init_cfg, dict)
                and self.init_cfg.get('type') == 'Pretrained'):
            logger = get_root_logger()
            #pdb.set_trace()
            print(self.init_cfg['checkpoint'])
            checkpoint = _load_checkpoint(
                self.init_cfg['checkpoint'], logger=logger, map_location='cpu')

            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            img_size = to_2tuple(self.img_size)
            #pdb.set_trace()
            if 'pos_embed' in state_dict.keys():
                if self.pos_embed.shape != state_dict['pos_embed'].shape:
                    logger.info(msg=f'Resize the pos_embed shape from '
                                f'{state_dict["pos_embed"].shape} to '
                                f'{self.pos_embed.shape}')
                    h, w = img_size
                    #pdb.set_trace()
                    pos_size = int(
                        math.sqrt(state_dict['pos_embed'].shape[1] - 1))
                    state_dict['pos_embed'] = self.resize_pos_embed(
                        state_dict['pos_embed'],
                        (h // self.patch_size, w // self.patch_size),
                        (pos_size, pos_size), self.interpolate_mode)

            #self.load_state_dict(state_dict, True)
            #pdb.set_trace()
            if "resize" not in self.init_cfg['checkpoint']:
                checkpoint['state_dict'] = state_dict
                pretrained = self.init_cfg['checkpoint'][:-4]+"_pos_resize.pth"
                torch.save(checkpoint, pretrained)
            else:
                if 'state_dict' not in checkpoint:
                    pass
                pretrained = self.init_cfg['checkpoint']
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
            #pdb.set_trace()
        elif self.init_cfg is not None:
            super(ElasticTransformer1, self).init_weights()
        else:
            # We only implement the 'jax_impl' initialization implemented at
            # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L353  # noqa: E501
            trunc_normal_(self.pos_embed, std=.02)
            trunc_normal_(self.cls_token, std=.02)
            for n, m in self.named_modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if m.bias is not None:
                        if 'ffn' in n:
                            nn.init.normal_(m.bias, mean=0., std=1e-6)
                        else:
                            nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    kaiming_init(m, mode='fan_in', bias=0.)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                    constant_init(m, val=1.0, bias=0.) 

    def make_encoder(self, **kwargs):
        return ElasticEncoder(**kwargs)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def manipulate_embedding(self, arch_meta):
        # print("manipulate_embedding: ",arch_meta,"\n")
        self.embedding_state = arch_meta
        self.elastic_patch_embed.projection.manipulate_arch(arch_meta)
        #
        for layer_name in self.layers:
            layer = getattr(self, layer_name)
            layer.manipulate_arch(arch_meta)

    def manipulate_encoder(self, arch_meta):
        # print("manipulate_encoder: ",arch_meta,"\n")
        self.encoder_state = arch_meta
        for layer_name in self.layers:
            layer = getattr(self, layer_name)
            layer.manipulate_arch(arch_meta)
        # for i, layer_name in enumerate(self.layers): 
        #     arch_meta_tmp = {
        #         'num_layers':5,
        #         'num_heads':{'num_heads':{'num_heads':7}},
        #         'feedforward_channels':{'feedforward_channels':{'feedforward_channels':40}}}
        #     arch_meta_tmp['num_layers'] = arch_meta['num_layers'][i]
        #     arch_meta_tmp['num_heads']['num_heads']['num_heads'] = arch_meta['num_heads']['num_heads']['num_heads'][i]
        #     arch_meta_tmp['feedforward_channels']['feedforward_channels']['feedforward_channels'] = arch_meta['feedforward_channels']['feedforward_channels']['feedforward_channels'][i]
        #     layer = getattr(self, layer_name)
        #     layer.manipulate_arch(arch_meta_tmp)

    def forward(self, x):
        #pdb.set_trace()
        # embedding: x : B,C,H,W >> B,N,D
        #pdb.set_trace()
        B = x.shape[0]
        H,W = x.shape[2],x.shape[3]
        x = self.elastic_patch_embed(x)
        # add tokens & position embedding
        cls_tokens = self.cls_token[:,:,:x.size(2)].expand(B, -1, -1)
        pos_embed = self.pos_embed[:,:x.size(1) + 1,:x.size(2)]
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + pos_embed
        x = self.drop_after_pos(x) 

        # transformer layer
        for i,layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            outs = layer(x, H, W, self.patch_size, self.out_indices,has_cls_token=True) # 

        # x[:,0] : cls_token output, x[:,1] : dist_token output
        return outs
        #return x[:,0], x[:,1]















