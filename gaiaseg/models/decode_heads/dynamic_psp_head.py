import pdb

# 3rd party lib
import torch
import torch.nn as nn
import torch.nn.functional as F


# mm lib
from mmcv.cnn import ConvModule
from mmcv.runner import auto_fp16, force_fp32
from mmseg.ops import resize
from mmseg.models.builder import HEADS
from mmseg.models.builder import build_loss
from mmseg.models.losses import accuracy

# gaia lib

from gaiavision.core import DynamicMixin
from gaiavision.core.bricks import build_norm_layer, DynamicBottleneck, DynamicConvModule
from gaiavision.core.ops import DynamicConv2d

from .psp_head import PSPHead

class DynamicPPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.
    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales, in_channels, channels, conv_cfg, norm_cfg,
                 act_cfg, align_corners):
        super(DynamicPPM, self).__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    # ConvModule -> DynamicynConvModule
                    DynamicConvModule(
                        self.in_channels,
                        self.channels,
                        1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg)))
            

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = resize(
                ppm_out,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs

@HEADS.register_module()
class DynamicPSPHead(PSPHead,DynamicMixin):
    """(Dynamic versrion)Pyramid Scene Parsing Network.
    This head is the implementation of
    `PSPNet <https://arxiv.org/abs/1612.01105>`_.
    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module. Default: (1, 2, 3, 6).
    """

    def __init__(self,
                 in_channels,
                 channels,
                 num_classes,
                 pool_scales=(1, 2, 3, 6),
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 input_transform=None,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 ignore_index=255,
                 sampler=None,
                 align_corners=False):
        
        nn.Module.__init__(self)
        
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index
        self.loss_decode = build_loss(loss_decode)
        self.ignore_index = ignore_index
        self.align_corners = align_corners

        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None
        # From Conv2d -> DynamicConv2d
        self.conv_seg = DynamicConv2d(channels, num_classes, kernel_size=1, padding=0)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.fp16_enabled = False

        self.pool_scales = pool_scales
        self.psp_modules = DynamicPPM(
            self.pool_scales,
            self.in_channels,
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        # From ConvModule -> DynamicConModule
        self.bottleneck = DynamicConvModule(
            self.in_channels + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        
        # for compute distillation loss
        loss['resize_logit'] = seg_logit
        
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)
        loss['loss_seg'] = self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        return loss


    @force_fp32(apply_to=('seg_logit','teacher_logits'))
    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg, **kwargs):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs)
        #############################NEW ADDED##########################
        # according to the Universally Slimmable Networks's experiments result.
        # Only compute the inplace distillation, don't contain the loss of gt and subnet's output
        interpolation = kwargs.get('interpolation',False)
        teacher_logits = kwargs.get('teacher_logits',None)
        distillation_weight = kwargs.get('distillation_weight',0.5)
        #pdb.set_trace()
        T = kwargs.get('T',2)
        if teacher_logits is not None:
            
            #pdb.set_trace()
            if(interpolation):
                seg_logits = resize(input=seg_logits,
                                    size=gt_semantic_seg.shape[2:],
                                    mode='bilinear',
                                    align_corners=self.align_corners)

                teacher_logits = resize(input=teacher_logits,
                                        size=gt_semantic_seg.shape[2:],
                                        mode='bilinear',
                                        align_corners=self.align_corners)
            #print("distillation_weight: ",distillation_weight)
            #print("T:, ",T)
            teacher_score = F.softmax(teacher_logits/T, dim=1)
            student_score = F.softmax(seg_logits/T, dim=1)
            batch_size = teacher_score.shape[0]
            #import pdb
            #pdb.set_trace()
            teacher_score = teacher_score.reshape((batch_size, -1)).unsqueeze(2)
            student_score = student_score.reshape((batch_size, -1)).unsqueeze(1)

            cross_entropy_loss = (-torch.bmm(student_score.log(), teacher_score)).mean()/1000
            #print("cross_entropy_loss: ",cross_entropy_loss)
            #student_score = torch.clamp(student_score,1e-7,1)
            #temp_loss = torch.nn.KLDivLoss()(student_score.log(), teacher_score)
            #assert temp_loss >= 0

            losses = dict()
            if not interpolation:
                teacher_logits = resize(input=teacher_logits,
                                        size=gt_semantic_seg.shape[2:],
                                        mode='bilinear',
                                        align_corners=self.align_corners)
            gt_semantic_seg = gt_semantic_seg.squeeze(1)
            losses['acc_seg'] = accuracy(teacher_logits, gt_semantic_seg)
            losses['loss_seg'] = (cross_entropy_loss*distillation_weight)
            #print('distillation losses[loss_seg]: ', losses['loss_seg'])   
            return losses     
        #############################NEW ADDED##########################
        #seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg)
        
        return losses
