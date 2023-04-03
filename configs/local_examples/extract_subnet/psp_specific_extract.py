model = dict(
    type='DynamicEncoderDecoder',
    backbone=dict(
        type='DynamicResNet',
        in_channels=3,
        stem_width=64,
        body_depth=[4,6,29,4],
        body_width=[80,160,320,640],
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        conv_cfg=dict(type='DynConv2d'),
        norm_cfg=dict(type='DynSyncBN', requires_grad=True),
        style='pytorch'),
    decode_head=dict(
        type='DynamicPSPHead',
        in_channels=640*4,
        in_index=3,
        conv_cfg=dict(type='DynConv2d'),
        channels=512,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        num_classes=4,
        norm_cfg=dict(type='DynSyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='DynamicFCNHead',
        conv_cfg=dict(type='DynConv2d'),
        in_channels=320*4,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=4,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)))
train_cfg = dict()
test_cfg = dict(mode='whole')

input_shape_cands = dict(
    key='data.input_shape',
    candidates=(480, 560, 640, 720, 800, 880, 960)
)
stem_width_range = dict(
    key='arch.backbone.stem.width',
    start=32,
    end=64,
    step=16,
)
body_width_range = dict(
    key='arch.backbone.body.width',
    start=[48, 96, 192, 384],
    end=[80, 160, 320, 640],
    step=[16, 32, 64, 128],
    ascending=True,
)
body_depth_range = dict(
    key='arch.backbone.body.depth',
    start=[2, 2, 5, 2],
    end=[4, 6, 29, 4],
    step=[1, 2, 2, 1],
)

# predefined model anchors
MAX = {
    'name': 'MAX',
    'arch.backbone.stem.width': stem_width_range['end'],
    'arch.backbone.body.width': body_width_range['end'],
    'arch.backbone.body.depth': body_depth_range['end'],
    'data.input_shape': 800,
}
MIN = {
    'name': 'MIN',
    'arch.backbone.stem.width': stem_width_range['start'],
    'arch.backbone.body.width': body_width_range['start'],
    'arch.backbone.body.depth': body_depth_range['start'],
    'data.input_shape': 800,
}
RSPECEFIC = {
    'name': 'RSPECEFIC',
    'arch.backbone.stem.width': 32,
    'arch.backbone.body.width': [48, 96, 192, 384],
    'arch.backbone.body.depth': [2, 2, 4, 2],
    'data.input_shape': 800,
}
R77 = {
    'name': 'R77',
    'arch.backbone.stem.width': 64,
    'arch.backbone.body.width': [64, 128, 256, 512],
    'arch.backbone.body.depth': [3, 4, 15, 3],
    'data.input_shape': 800,
}
R101 = {
    'name': 'R101',
    'arch.backbone.stem.width': 64,
    'arch.backbone.body.width': [64, 128, 256, 512],
    'arch.backbone.body.depth': [3, 4, 23, 3],
    'data.input_shape': 800,
}



train_sampler = dict(
    type='anchor',
    anchors=[
        dict(
            **RSPECEFIC,
        )
    ]
)


dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
checkpoint_config = dict(by_epoch=False, interval=8000)
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
