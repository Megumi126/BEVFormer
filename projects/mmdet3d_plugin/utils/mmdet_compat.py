"""Compatibility helpers for MMDetection v3 registries and utils."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from mmdet.models.task_modules.assigners import AssignResult, BaseAssigner
from mmdet.models.task_modules.coders import BaseBBoxCoder
from mmdet.registry import MODELS, TASK_UTILS

from projects.mmdet3d_plugin.utils.mmengine_compat import multi_apply, reduce_mean

BACKBONES = MODELS
DETECTORS = MODELS
HEADS = MODELS
TRANSFORMER = MODELS

BBOX_CODERS = TASK_UTILS
BBOX_ASSIGNERS = TASK_UTILS
MATCH_COST = TASK_UTILS

__all__ = [
    'AssignResult',
    'BaseAssigner',
    'BaseBBoxCoder',
    'BACKBONES',
    'BBOX_ASSIGNERS',
    'BBOX_CODERS',
    'DETECTORS',
    'HEADS',
    'MATCH_COST',
    'MODELS',
    'TASK_UTILS',
    'TRANSFORMER',
    'build_bbox_coder',
    'build_match_cost',
    'build_transformer',
    'inverse_sigmoid',
    'multi_apply',
    'reduce_mean',
]


def build_bbox_coder(cfg: dict, *args: Any, **kwargs: Any):
    return TASK_UTILS.build(cfg, *args, **kwargs)


def build_match_cost(cfg: dict, *args: Any, **kwargs: Any):
    return TASK_UTILS.build(cfg, *args, **kwargs)


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def build_transformer(cfg: dict, *args: Any, **kwargs: Any):
    return MODELS.build(cfg, *args, **kwargs)


def sync_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor
