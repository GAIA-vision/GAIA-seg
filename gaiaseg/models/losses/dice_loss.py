import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss
import pdb

class dice_loss(nn.Module):
    # Origin dice loss for binary-pixel-classification
    # For one category and background.
    def __init__(self):
        super(dice_loss, self).__init__()

    def	forward(self, input, target):
        N = target.size(0) # N 是batch size
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)
        intersection = input_flat * target_flat
        # pdb.set_trace()
        # print("Debug this Code")
        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss

class MulticlassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
    batch size and C is number of classes
    """
    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()
    
    def forward(self, input, target, weights=None):
        assert len(input.shape) == 4 # make sure it is (N,C,H,W) for input's shape 
        C = target.shape[1] # Get the category number

        # if weights is None:
        #     weights = torch.ones(C)

        Dice = dice_loss()
        totalLoss = 0

        # 这里可能需要一些修改，weight可能不是在这一步就考虑。
        for i in range(C):
            diceLoss = Dice(input[:,i], target[:,i])
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss/C

        return totalLoss

def dice(pred,
        label,
        weight=None,
        class_weight=None,
        reduction='mean',
        avg_factor=None,
        ignore_index=-100):
    """The wrapper function for dice loss`"""
    # class_weight is a manual rescaling weight given to each class.
    # If given, has to be a Tensor of size C element-wise losses
    pred_logit = F.sigmoid(pred)
    loss = dice_loss()(pred, label) 
    return loss


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

def multi_class_dice(pred,
                     label,
                     weight=None,
                     class_weight=None,
                     reduction='mean',
                     avg_factor=None,
                     ignore_index=255):
    pred_logit = F.softmax(pred,1)

    # support weighted multi-category dice loss
    if pred.dim() != label.dim():
        label, weight = _expand_onehot_labels(label, weight, pred.shape,ignore_index)

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()
    loss = MulticlassDiceLoss()(pred_logit,label,class_weight)

    return loss

@LOSSES.register_module()
class DiceLoss(nn.Module):
    """Dice Loss.

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
                 binary_class=False):
        super(DiceLoss, self).__init__()

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.binary_class = binary_class

        if self.binary_class:
            self.cls_criterion = dice
        else:
            self.cls_criterion = multi_class_dice

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
