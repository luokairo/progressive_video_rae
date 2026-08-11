from __future__ import annotations

from dataclasses import dataclass
import math
import random

import torch
from torch import Tensor
from torch import nn

from ..model.model import ProgressiveVideoRAE


STAGES = ("stage1a", "stage1b", "stage2a", "stage2b")
OBJECTIVE_MODES = {
    "stage1a": "full_repa",
    "stage1b": "full_repa_spatial_prefix",
    "stage2a": "full_repa",
    "stage2b": "full_repa_stateful",
}
STAGE_GEOMETRY = {
    "stage1a": (17, 5),
    "stage1b": (17, 5),
    "stage2a": (17, 5),
    "stage2b": (33, 9),
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
        forbidden = sorted(
            key
            for key in config
            if key in PREFIX_CONFIG_KEYS
            or key.startswith("prefix_")
            or "dct" in key.lower()
            or "mask_replacement" in key.lower()
            or "mask_replace" in key.lower()
        )
        if forbidden:
            raise ValueError(
                f"{stage} forbids prefix/DCT/mask-replacement configuration: "
                f"{forbidden}"
            )
    if stage == "stage1b":
        if config.get("prefix_schedule") != STAGE1B_PREFIX_SCHEDULE:
            raise ValueError(f"stage1b requires prefix_schedule={STAGE1B_PREFIX_SCHEDULE}")
        if int(config.get("gradient_accumulation_steps", 0)) != 8:
            raise ValueError("stage1b requires gradient_accumulation_steps=8")
        if int(config.get("full_microbatches_per_step", 0)) != 4:
            raise ValueError("stage1b requires full_microbatches_per_step=4")
    return expected

def validate_training_bundle(
    training: dict,
    model_config: dict,
    data_config: dict,
) -> tuple[int, int]:
    """Validate stage geometry after all command-line overrides are applied."""

    validate_stage_objective(training)
    stage = str(training["stage"])
    expected_frames, expected_latents = STAGE_GEOMETRY[stage]
    data_frames = int(data_config.get("num_frames", -1))
    if data_frames != expected_frames:
        raise ValueError(
            f"{stage} requires exactly {expected_frames} RGB frames, got {data_frames}"
        )
    if data_frames < 1 or (data_frames - 1) % 4:
        raise ValueError("Training clips must satisfy F=1+4*n")
    latent_frames = 1 + (data_frames - 1) // 4
    if latent_frames != expected_latents:
        raise ValueError(
            f"{stage} requires {expected_latents} latents, got {latent_frames}"
        )

    video = model_config["video"]
    state = model_config["state"]
    encoder = model_config["encoder"]
    decoder = model_config["decoder"]
    if int(video.get("num_frames", -1)) != expected_frames:
        raise ValueError(f"model.video.num_frames must be {expected_frames} for {stage}")
    if int(state.get("num_frames", -1)) != expected_latents:
        raise ValueError(f"model.state.num_frames must be {expected_latents} for {stage}")

    data_hw = (int(data_config["height"]), int(data_config["width"]))
    configured_sizes = {
        "model.video": (int(video["target_height"]), int(video["target_width"])),
        "encoder": (int(encoder["input_height"]), int(encoder["input_width"])),
        "decoder": (int(decoder["output_height"]), int(decoder["output_width"])),
    }
    for name, size in configured_sizes.items():
        if size != data_hw:
            raise ValueError(f"{name} spatial size {size} does not match data {data_hw}")
    spatial_upsample = int(decoder.get("spatial_upsample", 16))
    state_hw = (int(state["height"]), int(state["width"]))
    if (state_hw[0] * spatial_upsample, state_hw[1] * spatial_upsample) != data_hw:
        raise ValueError("State/decoder spatial geometry does not reconstruct the data size")
    if int(state["channels"]) != int(decoder["latent_channels"]):
        raise ValueError("State channels must match decoder latent_channels")

    if stage == "stage2b":
        maximum_prefix = data_frames + 1
        context = int(encoder.get("max_context_frames", 0))
        if maximum_prefix != 34:
            raise ValueError("Stage 2-B maximum V-JEPA prefix must be 34 frames")
        if maximum_prefix > context:
            raise ValueError(
                f"Stage 2-B maximum prefix {maximum_prefix} exceeds context {context}"
            )
    return expected_frames, expected_latents


def validate_training_batch(
    batch: dict,
    *,
    stage: str,
    expected_frames: int,
    expected_height: int,
    expected_width: int,
) -> None:
    pixel_values = batch.get("pixel_values")
    if not isinstance(pixel_values, Tensor) or pixel_values.ndim != 5:
        raise ValueError("Training batch pixel_values must be [B,3,F,H,W]")
    batch_size = int(pixel_values.shape[0])
    expected_shape = (3, expected_frames, expected_height, expected_width)
    if tuple(pixel_values.shape[1:]) != expected_shape:
        raise ValueError(
            f"Training batch geometry mismatch: got {tuple(pixel_values.shape[1:])}"
        )
    starts = batch.get("is_sequence_start")
    timestamps = batch.get("segment_start_timestamp")
    sequence_ids = batch.get("codec_sequence_id")
    origins = batch.get("sequence_origin")
    if (
        not isinstance(starts, Tensor)
        or starts.dtype != torch.bool
        or starts.shape != (batch_size,)
    ):
        raise ValueError("is_sequence_start must be BoolTensor[B]")
    if (
        not isinstance(timestamps, Tensor)
        or timestamps.dtype != torch.float64
        or timestamps.shape != (batch_size,)
    ):
        raise ValueError("segment_start_timestamp must be Float64Tensor[B]")
    if not isinstance(sequence_ids, list) or len(sequence_ids) != batch_size:
        raise ValueError("codec_sequence_id must be list[str] with batch length")
    if not all(isinstance(value, str) and value for value in sequence_ids):
        raise ValueError("codec_sequence_id entries must be non-empty strings")
    if not isinstance(origins, list) or len(origins) != batch_size:
        raise ValueError("sequence_origin must be list[str] with batch length")
    if stage == "stage2b":
        if not bool(starts.all()):
            raise ValueError("Stage 2-B requires every sample to declare sequence start")
        if any(origin != "sampled_segment" for origin in origins):
            raise ValueError("Stage 2-B requires sequence_origin=sampled_segment")


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
