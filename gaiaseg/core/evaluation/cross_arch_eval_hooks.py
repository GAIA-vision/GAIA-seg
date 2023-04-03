import os.path as osp
import warnings
from math import inf
from collections.abc import Sequence


# 3rd parth lib
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data import DataLoader

# mm lib
from mmcv.runner import Hook

# gaia lib
from gaiavision import broadcast_object
from gaiavision import DynamicMixin, fold_dict

# local lib
from .test_parallel import TestDistributedDataParallel


class CrossArchEvalHook(Hook):
    """Evaluation hook.
    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
    """

    def __init__(self, dataloader, interval=1, by_epoch=False, model_sampler=None,**eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError('dataloader must be a pytorch DataLoader, but got '
                            f'{type(dataloader)}')
        self.dataloader = dataloader
        self.interval = interval
        self.by_epoch = by_epoch
        self.eval_kwargs = eval_kwargs
        assert model_sampler is not None,"In cross arch mode, the val sampler should be specified in cfg"
        self.model_sampler = model_sampler


    def after_train_iter(self, runner):
        """After train epoch hook."""
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        from mmseg.apis import single_gpu_test
        runner.log_buffer.clear()

        if not hasattr(self.model_sampler, 'traverse'):
            raise AttributeError(f'{type(self.model_sampler)} has no attribute `traverse`')
        all_res = {}
        for i, meta in enumerate(self.model_sampler.traverse()):
            if hasattr(self.model_sampler, 'anchor_name'):
                anchor_id = self.model_sampler.anchor_name(i)
            else:
                anchor_id = i

            meta = broadcast_object(fold_dict(meta))
            self.manipulate_arch(runner, meta['arch'])

            results = single_gpu_test(runner.model, self.dataloader, show=False)
            self.evaluate(runner, results)


    def after_train_epoch(self, runner):
        """After train epoch hook."""
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        from mmseg.apis import single_gpu_test
        runner.log_buffer.clear()
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.evaluate(runner, results)


    def evaluate(self, runner, results):
        """Call evaluate function of dataset."""
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True


    def manipulate_arch(self, runner, arch_meta):
        if isinstance(runner.model, DynamicMixin):
            runner.model.manipulate_arch(arch_meta)
        elif isinstance(runner.model.module, DynamicMixin):
            runner.model.module.manipulate_arch(arch_meta)
        else:
            raise Exception(
                'Current model does not support arch manipulation.')


class DistCrossArchEvalHook(CrossArchEvalHook):
    """Distributed evaluation hook.
    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
        tmpdir (str | None): Temporary directory to save the results of all
            processes. Default: None.
        gpu_collect (bool): Whether to use gpu or cpu to collect results.
            Default: False.
    """

    def __init__(self,
                 dataloader,
                 model_sampler,
                 interval=1,
                 gpu_collect=False,
                 by_epoch=False,
                 **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError(
                'dataloader must be a pytorch DataLoader, but got {}'.format(
                    type(dataloader)))
        self.dataloader = dataloader
        self.interval = interval
        self.gpu_collect = gpu_collect
        self.by_epoch = by_epoch
        self.eval_kwargs = eval_kwargs
        self.model_sampler = model_sampler

    def after_train_iter(self, runner):
        """After train epoch hook."""
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        from mmseg.apis import multi_gpu_test
        runner.log_buffer.clear()
        
        if not hasattr(self.model_sampler, 'traverse'):
            raise AttributeError(f'{type(self.model_sampler)} has no attribute `traverse`')
        all_res = {}
        for i, meta in enumerate(self.model_sampler.traverse()):
            print()
            if hasattr(self.model_sampler, 'anchor_name'):
                anchor_id = self.model_sampler.anchor_name(i)
            else:
                anchor_id = i
            meta = broadcast_object(fold_dict(meta))
            self.manipulate_arch(runner, meta['arch'])
            print('Architecture: ',meta['arch'])
            results = multi_gpu_test(
                runner.model,
                self.dataloader,
                tmpdir=osp.join(runner.work_dir, '.eval_hook'),
                gpu_collect=self.gpu_collect)
            if runner.rank == 0:
                print("start evaluate results")
                print('\n')
                self.evaluate(runner, results)
                print("one Architecture has beed evaluated sucessfully")

    def after_train_epoch(self, runner):
        """After train epoch hook."""
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        from mmseg.apis import multi_gpu_test
        runner.log_buffer.clear()
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results)
