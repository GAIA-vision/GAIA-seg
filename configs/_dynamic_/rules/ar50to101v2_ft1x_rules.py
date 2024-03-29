model_space_filename = '/home/ma-user/work/pengjunran/workspace/experiments/ar50to101_nsync_all/profiling/fastft/metrics.json'

model_sampling_rules = dict(
    # NOTE: there are two strategies, pre-metric or post-metric, check which is better
    type='sequential',
    rules=[
        # 1. first 50% mAP models
        dict(
            type='sample',
            operation='top',
            key='fastft_metric.coco_bbox_mAP',
            value=1,
            mode='number',
        ),
    ]
)
