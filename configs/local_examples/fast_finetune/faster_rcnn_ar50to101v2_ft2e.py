_base_ = [
    '../../_dynamic_/models/faster_rcnn_fpn_ar50to101v2_gsync_ft.py',
    '../../_dynamic_/datasets/uni_trial.py',
    '../../_dynamic_/schedules/schedule_ft2e.py',
    '../../_dynamic_/rules/r50_s560_rules.py',
    # '../../_dynamic_/rules/r101_s640_rules.py',
    '../../_base_/default_runtime.py',
]
fp16=dict(loss_scale=512.)
train_cfg = dict(
    rpn=dict(sampler=dict(neg_pos_ub=5), allowed_border=-1),
    rcnn=dict(
        sampler=dict(
            _delete_=True,
            type='CombinedSampler',
            num=512,
            pos_fraction=0.25,
            add_gt_as_proposals=True,
            pos_sampler=dict(type='InstanceBalancedPosSampler'),
            neg_sampler=dict(
                type='IoUBalancedNegSampler',
                floor_thr=-1,
                floor_fraction=0,
                num_bins=3))))


# model
model = dict(
    type='DynamicFasterRCNN',
    # train from scratch
    backbone=dict(
        type='DynamicResNet',
        in_channels=3,
        stem_width=64,
        body_depth=[4, 6, 29, 4],
        body_width=[80, 160, 320, 640],
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        conv_cfg=dict(type='DynConv2d'),
        norm_cfg=dict(
            type='DynSyncBN',
            requires_grad=True,
            group_size=1),
        style='pytorch'),
    neck=dict(
        type='DynamicFPN',
        # in_channels should be 4 times the stage_channels
        in_channels=[320, 640, 1440, 2560],
        out_channels=256,
        conv_cfg=dict(type='DynConv2d'),
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        # NOTE: special setting
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),

        # NOTE: special setting
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),

    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=696, # unified label space
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=4.0), # 1.0 -> 4.0
            loss_bbox=dict(type='L1Loss', loss_weight=4.0)))) # 1.0 -> 4.0
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            match_low_quality=True,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        # NOTE: special setting
        # nms_post=1000,
        # max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            match_low_quality=False,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100)
    # soft-nms is also supported for rcnn testing
    # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
)

# dataset types
dataset_types = {
    'coco': 'CocoDataset',
    'object365': 'NamedCustomDataset',
    'openimages': 'NamedCustomDataset',
}
# annotation root
dataset_root = '/home/ma-user/work/pengjunran/data/det/'

# data root
data_roots = {
    'cocotrain': '/home/ma-user/work/pengjunran/data/det/coco/images/train2017',
    'cocoval': '/home/ma-user/work/pengjunran/data/det/coco/images/val2017',
    'openimages': '/home/ma-user/work/pengjunran/data/det/oidv6/images',
    'object365': '/home/ma-user/work/pengjunran/data/det/obj365v1/train',
    'obj_debug': '/home/ma-user/work/pengjunran/data/det/obj_debug/train',
}

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize',
         img_scale=[(480, 1333), (512, 1333), (544, 1333),
                    (576, 1333), (608, 1333), (640, 1333),
                    (672, 1333), (704, 1333), (736, 1333),
                    (768, 1333)],
         multiscale_mode='value',
         override=True,
         keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# ------------- dataset infomation -----------------
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=0,
    train=dict(
        type='UniConcatDataset',
        label_pool_file='hubs/labels/uni.0.0.3.csv',
        datasets=[
            # coco
            # dict(
            #     type=dataset_types['coco'],
            #     ann_file=dataset_root + 'coco/annotations/instances_train2017.json',
            #     img_prefix=data_roots['cocotrain'],
            #     pipeline=train_pipeline,
            # ),
            # object365v1
            # dict(
            #     type=dataset_types['object365'],
            #     name='object365',
            #     ann_file=dataset_root + 'obj365v1/annotations/objects365_generic_train.json',
            #     img_prefix=data_roots['object365'],
            #     pipeline=train_pipeline,
            # ),
            # for debug
            dict(
                type=dataset_types['object365'],
                name='object365',
                ann_file=dataset_root + 'obj_debug/annotations/objects365_debug.json',
                img_prefix=data_roots['obj_debug'],
                pipeline=train_pipeline,
            ),
        ],
    ),
    val=dict(
        type='UniConcatDataset',
        label_pool_file='hubs/labels/uni.0.0.3.csv',
        dataset_names=['coco'],
        test_mode=True,
        datasets=[
            dict(
                type=dataset_types['coco'],
                ann_file=dataset_root + 'coco/annotations/instances_val2017.json',
                img_prefix=data_roots['cocoval'],
                pipeline=test_pipeline
            ),
        ],
    ),
    test=dict(
        samples_per_gpu=8,
        type='UniConcatDataset',
        label_pool_file='hubs/labels/uni.0.0.3.csv',
        dataset_names=['coco'],
        test_mode=True,
        datasets=[
            dict(
                type=dataset_types['coco'],
                ann_file=dataset_root + 'coco/annotations/instances_val2017.json',
                img_prefix=data_roots['cocoval'],
                pipeline=test_pipeline
            ),
        ],
    ),
)
evaluation = dict(
    interval=1,
    metric='bbox',
    save_best='arch_avg',
    dataset_name='coco',
)

# schedules
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.000001) # default 1e-4
optimizer_config = dict(grad_clip=None)
lr_scaler = dict(
    policy='linear',
    base_lr=0.0001875, # 0.00125 -> 0.0001875
)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1,
    warmup_ratio=0.001,
    warmup_by_epoch=True,
    step=[1])
total_epochs = 2


# rules
model_space_path = 'hubs/flops/ar50to101v2_flops.json'
model_sampling_rules = dict(
    type='sequential',
    rules=[
        # 1. scale constraints
        dict(
            type='parallel',
            rules=[
                dict(func_str='lambda x: x[\'data.input_shape\'][-1] == 800'),
            ]
        ),
        # 2. depth constraints
        dict(func_str='lambda x: x[\'arch.backbone.body.depth\'] == (3, 4, 23, 3)'),
        dict(func_str='lambda x: x[\'arch.backbone.body.width\'] == (64, 128, 256, 512)'),
        # dict(func_str='lambda x: x[\'arch.backbone.stem.width\'] == 32'),
        # dict(func_str='lambda x: x[\'arch.backbone.stem.width\'] == 48'),
        dict(func_str='lambda x: x[\'arch.backbone.stem.width\'] == 64'),
        # 3. sample
        dict(
            type='sample',
            operation='random',
            value=1,
            # value=3,
            mode='number',
        ),
        # 4. merge all groups if more than one
        dict(type='merge'),
    ]
)

# default_runtime
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
