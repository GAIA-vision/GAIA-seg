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
    'obj_debug': '/cache/data/det/obj_debug/train',
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
            dict(
                type=dataset_types['coco'],
                ann_file=dataset_root + 'coco/annotations/instances_train2017.json',
                img_prefix=data_roots['cocotrain'],
                pipeline=train_pipeline,
            ),
            # object365 v1
            # dict(
            #     type=dataset_types['object365'],
            #     name='object365',
            #     ann_file=dataset_root + 'obj365v1/annotations/objects365_generic_train.json',
            #     img_prefix=data_roots['object365'],
            #     pipeline=train_pipeline,
            # ),
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
        samples_per_gpu=8, # match syncbn group size during training
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
