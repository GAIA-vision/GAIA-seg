# 配合auto_run，作为随时变化的config 用于超参搜索
_base_ = '/data1/haoran_yin/cq_temp/GAIA/seperate_GAIA-seg/workdirs/resnet/street/gaiaseg_train_supernet_idd.py'

optimizer = dict(lr=6e-05 * 0.2)
runner = dict(type='IterBasedRunner', max_iters=8000)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=4000, metric='mIoU')
data = dict(samples_per_gpu=4)