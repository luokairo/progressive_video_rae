from __future__ import annotations

from dataclasses import dataclass
import math
import random

from torch import nn

from ..model.model import ProgressiveVideoRAE


STAGES = ("stage1a", "stage1b", "stage2a", "stage2b")
OBJECTIVE_MODES = {
    "stage1a": "full_repa",
    "stage1b": "full_repa_spatial_prefix",
    "stage2a": "full_repa",
    "stage2b": "full_repa_stateful",
}
PREFIX_CONFIG_KEYS = {
    "prefix_schedule",
    "prefix_min",
    "prefix_max",
    "prefix_objective_weight",
    "prefix_lpips_weight",
    "full_microbatches_per_step",
}

STAGE1B_PREFIX_SCHEDULE = "fixed_4_full_3_single_1_pair"
STAGE1A_PHASES = ("warmup", "interface", "full")


def stage1a_phase(
    optimizer_step: int,
    *,
    wan_interface_step: int = 2000,
    wan_full_step: int = 5000,
) -> str:
    if not 0 <= wan_interface_step <= wan_full_step:
        raise ValueError("Stage 1A boundaries must satisfy 0 <= interface <= full")
    if optimizer_step < wan_interface_step:
        return "warmup"
    if optimizer_step < wan_full_step:
        return "interface"
    return "full"


@dataclass(frozen=True)
class MicrobatchTask:
    kind: str  # full, single_prefix, paired_prefix
    endpoint: int = 47
    previous_endpoint: int | None = None

    @property
    def is_full(self) -> bool:
        return self.kind == "full"


def validate_stage_objective(config: dict) -> str:
    stage = str(config.get("stage"))
    if stage not in STAGES:
        raise ValueError(f"Unsupported stage {stage}; expected one of {STAGES}")
    expected = OBJECTIVE_MODES[stage]
    if config.get("objective_mode") != expected:
        raise ValueError(f"{stage} requires objective_mode={expected}")
    if stage in ("stage2a", "stage2b"):
        forbidden = sorted(PREFIX_CONFIG_KEYS.intersection(config))
        if forbidden:
            raise ValueError(f"{stage} forbids spatial-prefix configuration: {forbidden}")
    if stage == "stage1b":
        if config.get("prefix_schedule") != STAGE1B_PREFIX_SCHEDULE:
            raise ValueError(f"stage1b requires prefix_schedule={STAGE1B_PREFIX_SCHEDULE}")
        if int(config.get("gradient_accumulation_steps", 0)) != 8:
            raise ValueError("stage1b requires gradient_accumulation_steps=8")
        if int(config.get("full_microbatches_per_step", 0)) != 4:
            raise ValueError("stage1b requires full_microbatches_per_step=4")
    return expected


def _set_trainable(module: nn.Module, trainable: bool) -> None:
    module.requires_grad_(trainable)
    module.train(trainable)


def configure_stage(
    model: ProgressiveVideoRAE,
    stage: str,
    *,
    optimizer_step: int = 0,
    wan_interface_step: int = 2000,
    wan_full_step: int = 5000,
) -> None:
    if stage not in STAGES:
        raise ValueError(f"Unsupported stage {stage}; expected one of {STAGES}")
    _set_trainable(model.encoder, False)
    if stage in ("stage1a", "stage1b"):
        _set_trainable(model.projector, True)
        _set_trainable(model.repa_projection, True)
        if stage == "stage1a":
            model.projector.shared_mask_set.requires_grad_(False)
    else:
        _set_trainable(model.projector, False)
        _set_trainable(model.repa_projection, False)

    if stage != "stage1a":
        _set_trainable(model.decoder, True)
        return

    phase = stage1a_phase(
        optimizer_step,
        wan_interface_step=wan_interface_step,
        wan_full_step=wan_full_step,
    )
    _set_trainable(model.decoder, False)
    _set_trainable(model.decoder.temporal_adapter, True)
    if phase in ("interface", "full"):
        _set_trainable(model.decoder.pre_decoder, True)
        _set_trainable(model.decoder.decoder.conv1, True)
        for name, module in model.decoder.decoder.named_modules():
            if name.endswith("time_conv"):
                _set_trainable(module, True)
    if phase == "full":
        _set_trainable(model.decoder, True)


def _cosine_endpoint(minimum: int, maximum: int) -> int:
    endpoints = list(range(minimum, maximum + 1))
    denominator = max(1, maximum - minimum)
    weights = [1.5 + 0.5 * math.cos(math.pi * (s - minimum) / denominator) for s in endpoints]
    return random.choices(endpoints, weights=weights, k=1)[0]


def sample_microbatch_tasks(
    stage: str,
    accumulation_steps: int,
    *,
    optimizer_step: int = 0,
) -> tuple[MicrobatchTask, ...]:
    if stage in ("stage1a", "stage2a", "stage2b"):
        return tuple(MicrobatchTask("full") for _ in range(accumulation_steps))
    if stage != "stage1b":
        raise ValueError(f"Unsupported stage: {stage}")
    if accumulation_steps != 8:
        raise ValueError("Stage 1B requires exactly 8 microbatches per optimizer step")
    tasks = [MicrobatchTask("full") for _ in range(4)]
    tasks.extend(
        MicrobatchTask("single_prefix", _cosine_endpoint(0, 46))
        for _ in range(3)
    )
    endpoint = _cosine_endpoint(1, 47)
    tasks.append(MicrobatchTask("paired_prefix", endpoint, endpoint - 1))
    random.shuffle(tasks)
    return tuple(tasks)


def adversarial_factor(optimizer_step: int, start_step: int, ramp_steps: int) -> float:
    if start_step < 0 or ramp_steps < 0:
        raise ValueError("GAN start and ramp steps must be non-negative")
    if optimizer_step < start_step:
        return 0.0
    if ramp_steps == 0:
        return 1.0
    return min(1.0, max(0.0, (optimizer_step - start_step) / ramp_steps))
