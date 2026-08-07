from __future__ import annotations

import random

from torch import nn

from ..model.model import ProgressiveVideoRAE


STAGES = ("stage1a", "stage1b", "stage2a")


def _set_trainable(module: nn.Module, trainable: bool) -> None:
    module.requires_grad_(trainable)
    module.train(trainable)


def configure_stage(model: ProgressiveVideoRAE, stage: str) -> None:
    if stage not in STAGES:
        raise ValueError(f"Unsupported stage {stage}; expected one of {STAGES}")
    _set_trainable(model.encoder, False)
    if stage in ("stage1a", "stage1b"):
        _set_trainable(model.projector, True)
        _set_trainable(model.decoder, True)
        _set_trainable(model.repa_projection, True)
    else:
        _set_trainable(model.projector, False)
        _set_trainable(model.decoder, True)
        _set_trainable(model.repa_projection, False)


def sample_prefix(stage: str, optimizer_step: int, minimum: int = 1, maximum: int = 63) -> int:
    if stage in ("stage1a", "stage2a"):
        return 64
    if optimizer_step % 2 == 0:
        return 64
    return random.randint(minimum, maximum)

