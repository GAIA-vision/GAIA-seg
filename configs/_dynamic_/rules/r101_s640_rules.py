model_space_path = 'hubs/flops/ar50to101v2_flops.json'
model_sampling_rules = dict(
    type='sequential',
    rules=[
        # 1. scale constraints
        dict(
            type='parallel',
            rules=[
                dict(func_str='lambda x: x[\'data.input_shape\'][-1] == 640'),
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
