# mm lib
from mmseg.models import SEGMENTORS, EncoderDecoder

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

