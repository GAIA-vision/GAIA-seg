import pdb

# 3rd party lib
import torch
import torch.nn as nn
import torch.nn.functional as F

# mm lib
from mmcv.cnn import ConvModule
from mmseg.models.builder import HEADS
from mmseg.core import build_pixel_sampler
from mmseg.ops import resize
from mmseg.models.builder import build_loss
from mmseg.models.losses import accuracy
from .fcn_head import FCNHead
from mmcv.runner import auto_fp16, force_fp32

# gaia lib
from gaiavision.core import DynamicMixin
from gaiavision.core import DynamicConv2d
from gaiavision.core.bricks import build_norm_layer, DynamicBottleneck, DynamicConvModule

@HEADS.register_module()
class DynamicFCNHead(FCNHead,DynamicMixin):
    """(Dynamic version)Fully Convolution Networks for Semantic Segmentation.

    This head is dynamic version implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 num_classes,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
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

        # nn.Conv2d -> DynamicConv2d
        self.conv_seg = DynamicConv2d(channels, num_classes, kernel_size=1, padding=0)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.fp16_enabled = False

        assert num_convs >= 0
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size

        if num_convs == 0:
            assert self.in_channels == self.channels

        convs = []
        convs.append(
            # ConvModule -> DynamicConvmodule
            DynamicConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        for i in range(num_convs - 1):
            convs.append(
                # ConvModule -> DynamicConvmodule
                DynamicConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            # ConvModule -> DynamicConvModule
            self.conv_cat = DynamicConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        output = self.convs(x)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        output = self.cls_seg(output)
        return output

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        
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
        #print('aux loss: ',loss['loss_seg'])
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
        teacher_logits = kwargs.get('aux_teacher_logits',None)
        #pdb.set_trace()
        distillation_weight = kwargs.get('distillation_weight',0.5)
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

            #pdb.set_trace()
            teacher_score = teacher_score.reshape((batch_size, -1)).unsqueeze(2)
            student_score = student_score.reshape((batch_size, -1)).unsqueeze(1)

            # 这个地方loss大是合理的才对，为啥正常train是auxillary 地方loss小？ 
            cross_entropy_loss = (-torch.bmm(student_score.log(), teacher_score)).mean()/2000
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