import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb
import numpy as np

from ..builder import LOSSES
from .accuracy import Accuracy, accuracy

from .cross_entropy_loss import CrossEntropyLoss
from .dice_loss import DiceLoss


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


@LOSSES.register_module()
class MixedLoss(nn.Module):

    """
    MixedLoss.
    """
    # 仔细想了下，是根据配置要混合的损失函数名称，然后用loss的registry去提取，这倒是一个方案，但是
    # 这样的话，这些loss的参数配置会比较麻烦。

    # 现在的实现方案是，直接按照config去配置，不过这个时候对没类loss的weight提取要写的优雅就可能
    # 要想一下，怎么比较好拓展，仔细想想让别人用的时候把weight按照字典的方式传入，这样自己代码是不麻烦，
    # 比如：dict(type='MixedLoss', each_loss_weight=dict(CrossEntropy=0.4,Dice=1),cross_entropy_config=xxx, dice_config=xxx))
    
    # 或者按照一个weight是一个列表形式，与各个loss的对应关系是，在字典中输入的loss config的配置顺序对应
    # 比如：dict(type='MixedLoss', each_loss_weight=[0.4,1.0],cross_entropy_config=xxx, dice_config=xxx)) # xxx都是字典

    def __init__(self,
                 each_loss_weight = None, # loss总共有三类weight，一个是针对auxiliary 和 main loss 
                                     # 第二个是对于每个loss会针对每一类有个权重
                                     # 第三个则是这里的对于mixed loss,每一类loss有个权重
                 cross_entropy_config=None,
                 dice_config=None,
                 loss_weight=1.0):
        super(MixedLoss, self).__init__()
        
        self.each_loss_weight = each_loss_weight
        
        self.cross_entropy_config = cross_entropy_config
        self.dice_config = dice_config
        
 
    def forward(self,
                cls_score,
                label,
                weight=None,
                **kwargs):
        """Forward function."""
        A_mark = 0
        loss_cls = None
        # pdb.set_trace()
        if self.cross_entropy_config!=None:
            loss_cls = self.each_loss_weight['CrossEntropy'] * CrossEntropyLoss(**self.cross_entropy_config)(cls_score,label)
            A_mark = 1

        if self.dice_config!=None:
            if loss_cls!=None:
                loss_cls += self.each_loss_weight['Dice']*DiceLoss(**self.dice_config)(cls_score, label)
            else:
                loss_cls = self.each_loss_weight['Dice']*DiceLoss(**self.dice_config)(cls_score, label)
            A_mark = 1

        try:
            assert A_mark != 0
        except:
            print("Mixed Loss contains no any valid loss type, please check")

        return loss_cls
