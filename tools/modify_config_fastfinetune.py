# 配合auto_run，作为随时变化的config  用于fastfinetune
_base_ = '/data1/haoran_yin/cq_temp/GAIA/seperate_GAIA-seg/workdirs/resnet/street/gaiaseg_train_supernet_idd.py'

optimizer = dict(lr=6e-05 * 0.2)
runner = dict(type='IterBasedRunner', max_iters=8000)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=4000, metric='mIoU')

# fastfinetune
model_sampling_rules = dict(rules=[dict(func_str='lambda x: x[\"overhead.flops\"] <=485*1e9 and x[\"overhead.flops\"] >=385*1e9'),dict(type='sample',operation='random',value=2,mode='number'),dict(type='merge')])