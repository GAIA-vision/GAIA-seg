# standard lib
import random
import math
import warnings
import pdb

# 3rd-party lib
import numpy as np
import torch
from torch.nn.modules.batchnorm import _BatchNorm

# mm lib
from mmcv.runner import get_dist_info, init_dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import build_optimizer, build_runner


from mmseg.datasets import build_dataloader, build_dataset
from mmseg.utils import get_root_logger
from mmseg.core import DistEvalHook, EvalHook

# gaia lib
from gaiavision.core import ManipulateArchHook

# local lib
from ..core.evaluation import CrossArchEvalHook, DistCrossArchEvalHook

# from ..datasets import build_dataset, UniConcatDataset

def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_segmentor(model,
                   train_sampler, # model sampler
                   val_sampler, # model sampler
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMSeg V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    # data_loader for training and validation
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            drop_last=True) for ds in dataset
    ]

    # put model on gpus
    if distributed:
        # Set `find_unused_parameters` True to enable sub-graph sampling
        find_unused_parameters = cfg.get('find_unused_parameters', True)
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        _, world_size = get_dist_info()
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
        world_size = 1

    # decide whether to scale up lr
    lr_scaler_config = cfg.get('lr_scaler', None)
    if lr_scaler_config is not None:
        total_batch_size = world_size * cfg.data.samples_per_gpu
        base_lr = lr_scaler_config['base_lr']
        scale_type = lr_scaler_config.get('policy', 'linear')
        if scale_type == 'linear':
            scaled_lr = base_lr * total_batch_size
        elif scale_type == 'power':
            temp = lr_scaler_config.get('temperature', 0.7)
            scaled_lr = base_lr * math.pow(total_batch_size, temp)
        cfg.optimizer.lr = scaled_lr

    optimizer = build_optimizer(model, cfg.optimizer)


    if cfg.get('runner') is None:
        cfg.runner = {'type': 'IterBasedRunner', 'max_iters': cfg.total_iters}
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
        
    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # register hooks
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    if cfg.get('manipulate_arch',True):
        #assert 1==2,"not manipulate arch"
        # add hook for architecture manipulation
        manipulate_arch_hook = ManipulateArchHook(train_sampler)
        runner.register_hook(manipulate_arch_hook)


    # register eval hooks
    if validate:
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = ('IterBasedRunner' not in cfg.runner['type'])
        if not isinstance(cfg.data.val,list):
            cfg.data.val = [cfg.data.val]
        for each in cfg.data.val:
            val_dataset = build_dataset(each, dict(test_mode=True))
            val_dataloader = build_dataloader(
                val_dataset,
                samples_per_gpu=1,
                workers_per_gpu=cfg.data.workers_per_gpu,
                dist=distributed,
                shuffle=False)

            if cfg.get('manipulate_arch',True):
                eval_hook = DistCrossArchEvalHook if distributed else CrossArchEvalHook
                runner.register_hook(
                    eval_hook(val_dataloader, val_sampler, **eval_cfg))
            else:
                eval_hook = DistEvalHook if distributed else EvalHook
                runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)

    calib_bn = cfg.get('caliberate_bn', None)
    if calib_bn:
        if calib_bn.get('reset_stats', False):
            def clean_bn_stats(m):
                if isinstance(m, _BatchNorm):
                    m.running_mean.zero_()
                    m.running_var.fill_(1)
            model.apply(clean_bn_stats)

    runner.run(data_loaders, cfg.workflow)
