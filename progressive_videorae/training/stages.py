from __future__ import annotations

import math
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


def sample_prefix(
    stage: str,
    optimizer_step: int,
    minimum: int = 1,
    maximum: int = 63,
    *,
    schedule: str | None = None,
    full_probability: float = 0.5,
) -> int:
    schedule = schedule or ("full" if stage in ("stage1a", "stage2a") else "alternating")
    if schedule == "full":
        return 64
    if schedule == "alternating" and optimizer_step % 2 == 0:
        return 64
    if schedule == "random":
        if not 0.0 <= full_probability <= 1.0:
            raise ValueError("full_probability must be in [0, 1]")
        if random.random() < full_probability:
            return 64
    elif schedule != "alternating":
        raise ValueError("prefix schedule must be full, alternating, or random")
    if minimum < 1 or maximum > 63 or minimum > maximum:
        raise ValueError("prefix range must satisfy 1 <= minimum <= maximum <= 63")
    return random.randint(minimum, maximum)


def normalize_objective_weights(
    full_weight: float, prefix_weight: float
) -> tuple[float, float]:
    full = float(full_weight)
    prefix = float(prefix_weight)
    if not math.isfinite(full) or not math.isfinite(prefix):
        raise ValueError("Objective weights must be finite")
    if full < 0.0 or prefix < 0.0:
        raise ValueError("Objective weights must be non-negative")
    total = full + prefix
    if total <= 0.0:
        raise ValueError("full_objective_weight and prefix_objective_weight cannot both be zero")
    return full / total, prefix / total


def sample_microbatch_prefixes(
    stage: str,
    optimizer_step: int,
    accumulation_steps: int,
    minimum: int = 1,
    maximum: int = 63,
    *,
    schedule: str | None = None,
    full_probability: float = 0.5,
    full_microbatches_per_step: int | None = None,
) -> tuple[int, ...]:
    """Build one rank-local prefix plan for a generator optimizer step."""

    if accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    resolved_schedule = schedule or (
        "full" if stage in ("stage1a", "stage2a") else "alternating"
    )
    if resolved_schedule != "mixed_accumulation":
        prefix = sample_prefix(
            stage,
            optimizer_step,
            minimum,
            maximum,
            schedule=resolved_schedule,
            full_probability=full_probability,
        )
        return (prefix,) * accumulation_steps

    full_count = (
        accumulation_steps // 2
        if full_microbatches_per_step is None
        else int(full_microbatches_per_step)
    )
    if accumulation_steps % 2 != 0 or full_count * 2 != accumulation_steps:
        raise ValueError(
            "mixed_accumulation currently requires an even accumulation count "
            "and exactly half full micro-batches"
        )
    if minimum < 1 or maximum > 63 or minimum > maximum:
        raise ValueError("prefix range must satisfy 1 <= minimum <= maximum <= 63")
    return tuple(
        64 if micro_step % 2 == 1 else random.randint(minimum, maximum)
        for micro_step in range(accumulation_steps)
    )


def objective_loss_scales(
    prefix_lengths: tuple[int, ...],
    full_weight: float,
    prefix_weight: float,
) -> tuple[tuple[float, ...], tuple[float, float]]:
    """Return per-micro loss scales and the effective full/prefix task weights."""

    normalized_full, normalized_prefix = normalize_objective_weights(
        full_weight, prefix_weight
    )
    full_count = sum(prefix == 64 for prefix in prefix_lengths)
    prefix_count = len(prefix_lengths) - full_count
    if full_count and prefix_count:
        scales = tuple(
            normalized_full / full_count
            if prefix == 64
            else normalized_prefix / prefix_count
            for prefix in prefix_lengths
        )
        return scales, (normalized_full, normalized_prefix)
    if full_count:
        return tuple(1.0 / full_count for _ in prefix_lengths), (1.0, 0.0)
    if prefix_count:
        return tuple(1.0 / prefix_count for _ in prefix_lengths), (0.0, 1.0)
    raise ValueError("At least one micro-batch is required")


def adversarial_factor(optimizer_step: int, start_step: int, ramp_steps: int) -> float:
    if start_step < 0 or ramp_steps < 0:
        raise ValueError("GAN start and ramp steps must be non-negative")
    if optimizer_step < start_step:
        return 0.0
    if ramp_steps == 0:
        return 1.0
    return min(1.0, max(0.0, (optimizer_step - start_step) / ramp_steps))

