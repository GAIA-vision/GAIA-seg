import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

# mm lib
from mmseg.core import add_prefix
from mmseg.models import SEGMENTORS, EncoderDecoder
from mmseg.ops import resize

# gaia lib
from gaiavision.core import DynamicMixin


@SEGMENTORS.register_module()
class DynamicEncoderDecoder(EncoderDecoder, DynamicMixin):
    # search_space = {'backbone', 'neck', 'roi_head', 'rpn_head'}
    search_space = {'backbone','decode_head','neck','auxiliary_head'}

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(DynamicEncoderDecoder, self).__init__(
                 backbone=backbone,
                 decode_head=decode_head,
                 neck=neck,
                 auxiliary_head=auxiliary_head,
                 train_cfg=train_cfg,
                 test_cfg=test_cfg,
                 pretrained=pretrained)


    def manipulate_backbone(self, arch_meta):
        self.backbone.manipulate_arch(arch_meta)


    def manipulate_decode_head(self, arch_meta):
        pass
    
    def manipulate_neck(self, arch_meta):
        pass

    def manipulate_auxiliary_head(self, arch_meta):
        pass


    def train_step(self, data_batch, optimizer, **kwargs):
        teacher_logits = kwargs.pop('teacher_logits',None)
        losses = self(**data_batch, teacher_logits=teacher_logits, **kwargs)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data_batch['img'].data),
            aux_logits=losses.get('aux_logits',None),
            logits=losses.get('logits',None))

        return outputs

    def simple_test(self, img, img_meta, rescale=True, **kwargs):
        """Simple test with single image."""
                
        if kwargs.get('return_logits',False):
            #pdb.set_trace()
            seg_logit = self.inference(img, img_meta, rescale=False,**kwargs)
            return seg_logit
        
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def forward_train(self, img, img_metas, gt_semantic_seg, **kwargs):
        """Forward function for training.
        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        #pdb.set_trace()
        x = self.extract_feat(img)

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg, **kwargs)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg, **kwargs)
            losses.update(loss_aux)

        return losses

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg, **kwargs):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg,**kwargs)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg,**kwargs)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg, **kwargs):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg,
                                                     **kwargs)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """


        #pdb.set_trace()
        if kwargs.get('return_logits',False):
            return self.simple_test(imgs, img_metas, **kwargs)

        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got '
                                f'{type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) != '
                             f'num of image meta ({len(img_metas)})')
        # all images in the same aug batch all of the same ori_shape and pad
        # shape
        for img_meta in img_metas:
            ori_shapes = [_['ori_shape'] for _ in img_meta]
            assert all(shape == ori_shapes[0] for shape in ori_shapes)
            img_shapes = [_['img_shape'] for _ in img_meta]
            assert all(shape == img_shapes[0] for shape in img_shapes)
            pad_shapes = [_['pad_shape'] for _ in img_meta]
            assert all(shape == pad_shapes[0] for shape in pad_shapes)

        if num_augs == 1:
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            return self.aug_test(imgs, img_metas, **kwargs)

    def inference(self, img, img_meta, rescale, **kwargs):
        """Inference with slide/whole style.
        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.
        Returns:
            Tensor: The output segmentation map.
        """
        if kwargs.get('return_logits',False):
            rescale=False
            return self.whole_inference(img, img_meta, rescale, **kwargs)

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def whole_inference(self, img, img_meta, rescale, **kwargs):
        """Inference with full image."""
        
        seg_logit = self.encode_decode(img, img_meta, not_resize=True, **kwargs)
        if kwargs.get('return_aux_logits',False):
            aux_seg_logit = self.encode_decode_aux(img, img_meta, not_resize=True, **kwargs)
            return (seg_logit,aux_seg_logit)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit
        
    def encode_decode(self, img, img_metas, **kwargs):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        out = self._decode_head_forward_test(x, img_metas)
        if kwargs.get('not_resize', False):
            return out
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def encode_decode_aux(self, img, img_metas, **kwargs):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        out = self._auxiliary_head_forward_test(x, img_metas)
        if kwargs.get('not_resize', False):
            return out
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _auxiliary_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.auxiliary_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits