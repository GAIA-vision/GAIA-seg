import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss
import pdb


def _expand_onehot_labels(labels, label_weights, target_shape, ignore_index):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_zeros(target_shape)
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(valid_mask, as_tuple=True)

    if inds[0].numel() > 0:
        if labels.dim() == 3:
            bin_labels[inds[0], labels[valid_mask], inds[1], inds[2]] = 1
        else:
            bin_labels[inds[0], labels[valid_mask]] = 1

    valid_mask = valid_mask.unsqueeze(1).expand(target_shape).float()
    if label_weights is None:
        bin_label_weights = valid_mask
    else:
        bin_label_weights = label_weights.unsqueeze(1).expand(target_shape)
        bin_label_weights *= valid_mask

    return bin_labels, bin_label_weights

def origin_focal(pred,
                     label,
                     weight=None,
                     class_weight=None,
                     reduction='mean',
                     avg_factor=None,
                     ignore_index=255):
    '''
    基于focal loss 初始论文公式(5),α= 0.25,γ= 2实现，这两个超参数
    在论文中的目标检测里面是结果最好的，但用到这边分割之后，应该还是要调一下参

    original focal不用修改直接二分类还是多分类都一样用，一开始写这个函数的时候受自己在实现
    BCE的时候的影响，以为也要像cross_entropy_loss.py里面的实现一样， 实现
    一个multiclass focal 和一个 binary class focal。
    '''

    #pdb.set_trace() 

    pred_logit = F.sigmoid(pred) # 这个地方可以sigmoid，或者softmax，哪一个选择更好，可能得ablation study验证下
    
    alpha = 0.25
    gamma = 2

    # 先把label扩充成one-hot encoding的形式
    if pred.dim() != label.dim():
        assert (pred.dim() == 2 and label.dim() == 1) or (
                pred.dim() == 4 and label.dim() == 3), \
            'Only pred shape [N, C], label shape [N] or pred shape [N, C, ' \
            'H, W], label shape [N, H, W] are supported'
        label, weight = _expand_onehot_labels(label, weight, pred.shape,
                                              ignore_index)
    
    # 初始的focal loss其实本质上就是对于BCE的权重是根据预测的score自动设置的
    # 所以实现上，其实和EQL一样，就是改weight，然后再用BCE就好了。

    class_weight = torch.ones(pred.shape)
    temp_pred = pred.cpu()
    temp_label = label.cpu()
    class_weight[temp_label==1] = alpha*((1-temp_pred[temp_label==1]).pow(gamma))
    class_weight[temp_label==0] = alpha*((temp_pred[temp_label==0]).pow(gamma))

    class_weight = class_weight.cuda().detach() 

    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), weight=class_weight, reduction='none')

    
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss


class cross_entropy_focal_loss(nn.Module):
    '''
    按照softmax 然后cross entropy的形式实现focal loss没办法直接用
    F.CrossEntropyLoss这个func，通过改weight的形式来实现。
    所以这里自己根据对应形式实现下
    '''
    def __init__(self):
        super(cross_entropy_focal_loss, self).__init__()

    def	forward(self, input, target):
        N = target.size(0) # N 是batch size
        input = F.softmax(input,1)
        # pdb.set_trace()
        # focal loss的两个hyperparameter
        alpha = 0.25
        gamma = 2
        W,H = input.shape[2],input.shape[3]
        
        temp_input = input.cpu().detach()
        temp_target = target.cpu().detach()
        weight = torch.zeros(input.shape)
        weight[temp_target == 1] = alpha*((1-temp_input[temp_target == 1]).pow(gamma)) 
        weight = weight.cuda()
        loss = -torch.log(input)*weight/(W*H)
        loss = loss.sum()/N # 

        return loss

def cross_entropy_focal(pred,
                     label,
                     weight=None,
                     class_weight=None,
                     reduction='mean',
                     avg_factor=None,
                     ignore_index=255):
    if pred.dim() != label.dim():
        label, weight = _expand_onehot_labels(label, weight, pred.shape,ignore_index)
    # pdb.set_trace()
    loss = cross_entropy_focal_loss()(pred,label)

    return loss


@LOSSES.register_module()
class FocalLoss(nn.Module):
    """Focal Loss.

    Args:
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float], optional): Weight of each class.
            Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 reduction='mean',
                 class_weight=None, # class_weight is for specify category. loss_weight is for different loss head
                 loss_weight=1.0,
                 use_origin_focal=True):
        super(FocalLoss, self).__init__()

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.use_origin_focal = use_origin_focal

        if self.use_origin_focal:
            self.cls_criterion = origin_focal
        else:
            self.cls_criterion = cross_entropy_focal

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls
