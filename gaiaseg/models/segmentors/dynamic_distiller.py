''' 
参考OpenSelf的实现方式，里面有两个网络，一个是fix住的teacher，一个是
动态可变的student网络. 有一个细节需要注意，按照pairwise loss[1]进行蒸馏的时候
feature map必须是同样的H,W。

Ref:
    [1]: Structured Knowledge Distillation for Semantic Segmentation
    [2]: Distilling the knowledge in a neural network
'''
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import interpolate

from mmcv.runner import load_checkpoint, get_dist_info
from mmseg.core import add_prefix
from mmseg.ops import resize
from mmseg.models import builder, build_segmentor
from mmseg.models.builder import SEGMENTORS
from mmseg.models.segmentors.base import BaseSegmentor

from gaiavision.core import DynamicMixin

def deal_with_position_embedding(ckpt, model):
    mark = 0 # 如果本来就匹配，没必要进行处理。
    rank, world_size = get_dist_info()
    pth_file = torch.load(ckpt, map_location='cpu')
    state_dict = pth_file['state_dict']

    all_keys = list(state_dict.keys())
    for key in all_keys:
        if "relative_position_index" in key: # 这个好像是自动加载的，并不是梯度更新的参数，所以直接去掉即可。
            state_dict.pop(key)

        if "relative_position_bias_table" in key:
            rel_pos_bias = state_dict[key]
            src_num_pos, num_attn_heads = rel_pos_bias.size()
            dst_num_pos, _ = model.state_dict()[key].size()
            dst_patch_shape = model.backbone.patch_embed.patch_shape
            if dst_patch_shape[0] != dst_patch_shape[1]:
                raise NotImplementedError()
            num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)
            src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
            dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
            if src_size != dst_size:
                if rank == 0:
                    print("Position interpolate for %s from %dx%d to %dx%d" % (
                    key, src_size, src_size, dst_size, dst_size))
                extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                def geometric_progression(a, r, n):
                    return a * (1.0 - r ** n) / (1.0 - r)

                left, right = 1.01, 1.5
                while right - left > 1e-6:
                    q = (left + right) / 2.0
                    gp = geometric_progression(1, q, src_size // 2)
                    if gp > dst_size // 2:
                        right = q
                    else:
                        left = q

                # if q > 1.13492:
                #     q = 1.13492

                dis = []
                cur = 1
                for i in range(src_size // 2):
                    dis.append(cur)
                    cur += q ** (i + 1)

                r_ids = [-_ for _ in reversed(dis)]

                x = r_ids + [0] + dis
                y = r_ids + [0] + dis

                t = dst_size // 2.0
                dx = np.arange(-t, t + 0.1, 1.0)
                dy = np.arange(-t, t + 0.1, 1.0)
                if rank == 0:
                    print("x = {}".format(x))
                    print("dx = {}".format(dx))

                all_rel_pos_bias = []

                for i in range(num_attn_heads):
                    z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
                    f = interpolate.interp2d(x, y, z, kind='cubic')
                    all_rel_pos_bias.append(
                        torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(rel_pos_bias.device))

                rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)
                new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
                state_dict[key] = new_rel_pos_bias
                mark = 1

    if 'pos_embed' in state_dict:
        pos_embed_checkpoint = state_dict['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.backbone.patch_embed.num_patches
        num_extra_tokens = model.backbone.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            if rank == 0:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            state_dict['pos_embed'] = new_pos_embed
            mark = 1

    # interpolate position bias table if needed. 没get到这个地方是干嘛，上面不是已经对
    # relative_position_bias_table 进行插值了吗？
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for table_key in relative_position_bias_table_keys:
        table_pretrained = state_dict[table_key]
        table_current = model.state_dict()[table_key]
        L1, nH1 = table_pretrained.size()
        L2, nH2 = table_current.size()
        if nH1 != nH2:
            logger.warning(f"Error in loading {table_key}, pass")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                table_pretrained_resized = F.interpolate(
                     table_pretrained.permute(1, 0).view(1, nH1, S1, S1),
                     size=(S2, S2), mode='bicubic')
                state_dict[table_key] = table_pretrained_resized.view(nH2, L2).permute(1, 0)
                mark = 1
    if mark == 0:
        return ckpt
    new_ckpt_file = ckpt[:-4] + "_new_" + str(model.backbone.patch_embed.img_size[0]) + "_" + str(model.backbone.patch_embed.patch_size[0]) + ".pth"
    torch.save(pth_file, new_ckpt_file)
    return new_ckpt_file


@SEGMENTORS.register_module()
class DynamicDistiller(BaseSegmentor, DynamicMixin):
    """Distiller, A fix teacher segmentor with dynamic student segmentor.
       Note that: teacher segmentor can be any type, the student segmentor must 
       be the type of encoder_decoder. cascate_encoder_decoder for student subnet
       is not supported yet.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 teacher_segmentor=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 teacher_ckpt=None,
                 has_distill_loss=True,
                 distill_loss_temperature=1,
                 has_pairwise_loss=True,
                 pairwise_loss_temperature=1,
                 distill_loss_weight=1,
                 pairwise_loss_weight=1,
                 ):
        super(DynamicDistiller, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

        assert self.with_decode_head

        if has_distill_loss==False and has_pairwise_loss==False:
            self.has_distill_loss = has_distill_loss
            self.has_pairwise_loss = has_pairwise_loss
            print("No distill, debug mode")
        else:
            self.teacher_segmentor = build_segmentor(teacher_segmentor, test_cfg=test_cfg)
            assert teacher_ckpt is not None, "Teacher ckpt is missed !"
            teacher_ckpt = deal_with_position_embedding(teacher_ckpt, self.teacher_segmentor) # 目前仅支持teacher_ckpt是本地pth文件，不支持url格式。
            ckpt = load_checkpoint(self.teacher_segmentor, teacher_ckpt, map_location='cpu')
            self.teacher_segmentor = self.teacher_segmentor.cuda()
            self.teacher_segmentor.eval() # 这种方式有个问题就是耗显存，每个卡都耗，不过一卡耗和多卡耗好像区别不大。。。
            self.has_distill_loss = has_distill_loss
            self.has_pairwise_loss = has_pairwise_loss
            self.pairwise_loss_weight = pairwise_loss_weight
            self.distill_loss_temperature = distill_loss_temperature
            self.distill_loss_weight = distill_loss_weight
            self.pairwise_loss_temperature = pairwise_loss_temperature

    def manipulate_backbone(self, arch_meta):
        self.backbone.manipulate_arch(arch_meta)

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        #super(DynamicDistiller, self).init_weights(pretrained)
        #self.backbone.init_weights()
        self.decode_head.init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        out = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out
    
    def prepare_distill_feature(self, img, img_metas):
        # 有torch.no_grad(), 不用再detach了吧？
        with torch.no_grad():
            x = self.teacher_segmentor.extract_feat(img)
            out = self.teacher_segmentor._decode_head_forward_test(x, img_metas)
            out = resize(
                input=out,
                size=img.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            return (x, out)

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)
        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses
    
    def pairwise_loss(self, student_x, teacher_y, weight=1, temperature=1):
        """Pairwise loss for distillation.

        Args:
            student_x (Tensor): source(student) tensor. [N,C,H,W]
            y (Tensor): target(teacher) tensor  [N,C,H,W]

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # GAIA-openself 里面直接有这个，搬运过来即可。
        batch_size = student_x.size(0)
        H,W = student_x.size(2),student_x.size(3)

        step_h = int(0.5*H)
        step_w = int(0.5*W)
        choice_h = np.random.uniform(0,0.5)
        choice_w = np.random.uniform(0,0.5)
        start_h = int(choice_h*H)
        start_w = int(choice_w*W)
        student_x = student_x[:,:,start_h:start_h+step_h,start_w+step_w]
        teacher_y = teacher_y[:,:,start_h:start_h+step_h,start_w+step_w]
        
        student_x = torch.nn.functional.normalize(student_x, dim=1)
        teacher_y = torch.nn.functional.normalize(teacher_y, dim=1)
        student_x = student_x.view(student_x.size(0), student_x.size(1), -1) #[N,C,HW]
        teacher_y = teacher_y.view(teacher_y.size(0), teacher_y.size(1), -1) #[N,C,HW]
        student_x = torch.bmm(student_x.transpose(1,2), student_x) #[N,HW,HW]  [1] 里面并不是全局，只是局部，这个代码好写吗？卷积实现？
        teacher_y = torch.bmm(teacher_y.transpose(1,2), teacher_y) #[N,HW,HW]  
        
        return weight*-torch.sum(F.softmax(teacher_y / temperature, dim=1) * F.log_softmax(student_x / temperature, dim=2))/(batch_size*step_h*step_w)

    def distill_loss(self, student_x, teacher_y, weight=1, temperature=1):
        """normal distillation loss, see [2].

        Args:
            student_x (Tensor): source(student) tensor. [N,Class_num,H,W]
            teacher_y (Tensor): target(teacher) tensor  [N,Class_num,H,W]

        Returns:
            loss (Tensor): a loss component
        """
        # GAIA-openself 里面直接有这个，搬运过来即可。就是正常的cross entropy loss
        H,W = student_x.size(2),student_x.size(3)
        batch_size = student_x.size(0)

        # cross entropy loss -qlogp.  按照逐pixel进行计算
        return weight*-torch.sum(F.softmax(teacher_y / temperature, dim=1) * F.log_softmax(student_x / temperature, dim=1))/(batch_size*H*W)
        # 不用softmax, 用 torch.nn.functional.normalize() 然后用 对应数值的平方当做概率呢？
        # x = torch.nn.functional.normalize(x,dim=1)
        # x = x*x
        # y 同上, 这样x,y 在对应维度上也变成了概率分布。
        # -torch.sum(y/temperature * F.log(x/temperature))


    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def forward_train(self, img, img_metas, gt_semantic_seg):
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
            distill_loss (bool): Whther with distill loss.
            pairewise_loss (bool): Whther with pairewise loss, see [1].

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        #pdb.set_trace()
        if self.has_distill_loss or self.has_pairwise_loss:
            teacher_x, teacher_seglogits = self.prepare_distill_feature(img, img_metas)

        x = self.extract_feat(img) # [N,C,H,W]
        #pdb.set_trace()
        seg_logits = self.decode_head.forward(x) # [N,C,H',W']
        losses = self.decode_head.losses(seg_logits, gt_semantic_seg)

        if self.has_distill_loss:
            seg_logits = resize(
                input=seg_logits,
                size=img.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            losses['distill_loss_seg'] = self.distill_loss(seg_logits, teacher_seglogits, self.distill_loss_weight, self.distill_loss_temperature)
        if self.has_pairwise_loss:
            losses['pairwise_loss_seg'] = self.pairwise_loss(x[-1], teacher_x[-1], self.pairwise_loss_weight, self.pairwise_loss_temperature)


        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        return losses

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            seg_logit = resize(
                seg_logit,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale):
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

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
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

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred