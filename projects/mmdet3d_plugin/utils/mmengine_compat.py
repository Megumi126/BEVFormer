"""Compatibility helpers for migrating from MMCV Runner to MMEngine."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any, List

import numpy as np
import torch
import torch.distributed as dist
from mmengine.config import Config, ConfigDict, DictAction
from mmengine.dist import get_dist_info, init_dist
from mmengine.hooks import DistEvalHook, DistSamplerSeedHook, EvalHook, Hook
from mmengine.model import (MMDataParallel, MMDistributedDataParallel,
                            BaseModule, ModuleList, Sequential, auto_fp16,
                            force_fp32, wrap_fp16_model)
from mmengine.registry import (HOOKS, OPTIMIZERS, Registry, RUNNERS,
                               build_from_cfg)
from mmengine.runner import BaseRunner, EpochBasedRunner
from mmengine.runner.checkpoint import load_checkpoint, save_checkpoint
from mmengine.runner.utils import get_host_info
from mmengine.utils import digit_version, to_2tuple
from mmengine.utils.dl_utils import TORCH_VERSION
from torch.utils.data.dataloader import default_collate


class DataContainer:
    """A container for any type of data.

    This is a minimal replacement for mmcv.parallel.DataContainer used in
    MMCV v1.x. It supports CPU-only data and stacked tensors for batching.
    """

    def __init__(self,
                 data: Any,
                 stack: bool = False,
                 padding_value: float = 0,
                 cpu_only: bool = False,
                 pad_dims: int | None = None) -> None:
        self.data = data
        self.stack = stack
        self.padding_value = padding_value
        self.cpu_only = cpu_only
        self.pad_dims = pad_dims

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (f'{self.__class__.__name__}(stack={self.stack}, '
                f'cpu_only={self.cpu_only}, pad_dims={self.pad_dims})')


def _pad_tensor(tensor: torch.Tensor, pad_dims: int, pad_value: float) -> torch.Tensor:
    if pad_dims <= 0:
        return tensor
    max_shape = list(tensor.shape)
    for dim in range(-pad_dims, 0):
        max_shape[dim] = max(max_shape[dim], tensor.shape[dim])
    padded = tensor.new_full(max_shape, pad_value)
    slices = tuple(slice(0, dim) for dim in tensor.shape)
    padded[slices] = tensor
    return padded


def collate(batch: List[Any], samples_per_gpu: int = 1) -> Any:
    """Puts each data field into a tensor/DataContainer with outer dimension batch size."""
    if not isinstance(batch, Sequence):
        return batch
    if len(batch) == 0:
        return batch

    elem = batch[0]
    if isinstance(elem, DataContainer):
        if elem.cpu_only:
            return DataContainer([item.data for item in batch], cpu_only=True)
        if elem.stack:
            data = [item.data for item in batch]
            if elem.pad_dims is None:
                return DataContainer(default_collate(data), stack=True)
            pad_dims = elem.pad_dims
            padded = [
                _pad_tensor(item, pad_dims, elem.padding_value) for item in data
            ]
            return DataContainer(default_collate(padded), stack=True)
        return DataContainer([item.data for item in batch], stack=False)
    if isinstance(elem, torch.Tensor):
        return default_collate(batch)
    if isinstance(elem, np.ndarray):
        return default_collate([torch.from_numpy(item) for item in batch])
    if isinstance(elem, (float, int, str)):
        return default_collate(batch)
    if isinstance(elem, Mapping):
        return {key: collate([d[key] for d in batch], samples_per_gpu)
                for key in elem}
    if isinstance(elem, Sequence):
        transposed = list(zip(*batch))
        return [collate(samples, samples_per_gpu) for samples in transposed]
    return default_collate(batch)


def deprecated_api_warning(*args: Any, **kwargs: Any):  # type: ignore[override]
    """Compatibility wrapper for deprecated_api_warning in older MMCV."""

    def decorator(func):
        return func

    return decorator


def build_optimizer(model: torch.nn.Module, cfg: dict):
    optimizer_cfg = cfg.copy()
    optimizer_cfg.setdefault('params', model.parameters())
    return OPTIMIZERS.build(optimizer_cfg)


def build_runner(cfg: dict, default_args: dict | None = None):
    runner_cfg = cfg.copy()
    runner_type = runner_cfg.pop('type', 'EpochBasedRunner')
    runner_cfg['type'] = runner_type
    return RUNNERS.build(runner_cfg, default_args=default_args)


class OptimizerHook(Hook):
    """Simple optimizer hook for compatibility with MMCV Runner."""

    def __init__(self, grad_clip: dict | None = None, **kwargs: Any) -> None:
        self.grad_clip = grad_clip
        self.kwargs = kwargs

    def after_train_iter(self, runner, batch_idx: int, data_batch=None, outputs=None):
        if outputs is None:
            return
        loss = outputs['loss'] if isinstance(outputs, dict) and 'loss' in outputs else outputs
        runner.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(runner.model.parameters(), **self.grad_clip)
        runner.optimizer.step()


class Fp16OptimizerHook(OptimizerHook):
    """FP16 optimizer hook using torch.cuda.amp for compatibility."""

    def __init__(self, loss_scale: str | float = 'dynamic', **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.loss_scale = loss_scale
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(loss_scale == 'dynamic'),
            init_scale=(1.0 if loss_scale == 'dynamic' else float(loss_scale)),
        )

    def after_train_iter(self, runner, batch_idx: int, data_batch=None, outputs=None):
        if outputs is None:
            return
        loss = outputs['loss'] if isinstance(outputs, dict) and 'loss' in outputs else outputs
        runner.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        if self.grad_clip is not None:
            self.scaler.unscale_(runner.optimizer)
            torch.nn.utils.clip_grad_norm_(runner.model.parameters(), **self.grad_clip)
        self.scaler.step(runner.optimizer)
        self.scaler.update()


def reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Reduce tensor mean across distributed ranks."""
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments."""
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))
