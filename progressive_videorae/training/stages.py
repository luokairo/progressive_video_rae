from __future__ import annotations

from dataclasses import dataclass
import random

import torch
from torch import Tensor
from torch import nn

from ..model.model import ProgressiveVideoRAE


STAGES = ("stage1a", "stage1a_plus", "stage1b", "stage2a", "stage2b")
OBJECTIVE_MODES = {
    "stage1a": "full_repa",
    "stage1a_plus": "cross_clip_cache_reconstruction",
    "stage1b": "nested_spectral_hrepa_full_anchor",
    "stage2a": "full_repa",
    "stage2b": "full_repa_stateful",
}
STAGE_GEOMETRY = {
    "stage1a": (17, 5),
    "stage1a_plus": (33, 9),
    "stage1b": (17, 5),
    "stage2a": (17, 5),
    "stage2b": (33, 9),
}
PREFIX_CONFIG_KEYS = {
    "decoder_trainable_policy",
    "prefix_schedule",
    "prefix_min",
    "prefix_max",
    "prefix_objective_weight",
    "prefix_lpips_weight",
    "p47_full_microbatches_per_step",
    "p47_objective_weight",
    "prefix_repa_schedule",
    "prefix_repa_levels",
    "prefix_repa_global_weight",
    "prefix_repa_local_weight",
}

STAGE1B_PREFIX_SCHEDULE = "fixed_7_prefix_1_p47_full"
STAGE1B_REPA_SCHEDULE = "fixed_6level_spatial_pyramid"
STAGE1B_REPA_LEVELS = ((1, 1), (2, 3), (4, 6), (8, 12), (15, 24), (30, 48))
STAGE1B_DECODER_POLICY = "temporal_interface"
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
    if stage in ("stage1a_plus", "stage2a", "stage2b"):
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
    if stage == "stage1a_plus":
        required = {
            "gradient_accumulation_steps": 4,
            "global_batch_size": 32,
            "adversarial_weight": 0.0,
            "repa_max_factor": 0.0,
        }
        mismatches = {
            key: (config.get(key), value)
            for key, value in required.items()
            if config.get(key) != value
        }
        if mismatches:
            raise ValueError(f"stage1a_plus configuration mismatch: {mismatches}")
    if stage == "stage1b":
        if config.get("prefix_schedule") != STAGE1B_PREFIX_SCHEDULE:
            raise ValueError(f"stage1b requires prefix_schedule={STAGE1B_PREFIX_SCHEDULE}")
        if int(config.get("gradient_accumulation_steps", 0)) != 8:
            raise ValueError("stage1b requires gradient_accumulation_steps=8")
        if int(config.get("p47_full_microbatches_per_step", 0)) != 1:
            raise ValueError("stage1b requires p47_full_microbatches_per_step=1")
        if int(config.get("prefix_min", -1)) != 0 or int(config.get("prefix_max", -1)) != 46:
            raise ValueError("stage1b random prefix endpoints must cover [0,46]")
        if config.get("prefix_repa_schedule") != STAGE1B_REPA_SCHEDULE:
            raise ValueError(
                f"stage1b requires prefix_repa_schedule={STAGE1B_REPA_SCHEDULE}"
            )
        configured_levels = tuple(
            tuple(int(size) for size in level)
            for level in config.get("prefix_repa_levels", ())
        )
        if configured_levels != STAGE1B_REPA_LEVELS:
            raise ValueError(
                f"stage1b requires prefix_repa_levels={STAGE1B_REPA_LEVELS}"
            )
        for key in (
            "prefix_objective_weight",
            "p47_objective_weight",
            "prefix_repa_global_weight",
            "prefix_repa_local_weight",
        ):
            if float(config.get(key, -1.0)) != 1.0:
                raise ValueError(f"stage1b requires {key}=1.0")
        fixed_values = {
            "repa_local_weight": 1.0,
            "repa_global_weight": 1.0,
            "adversarial_weight": 0.05,
            "disc_start": 0,
            "adversarial_ramp_steps": 1000,
        }
        for key, value in fixed_values.items():
            if float(config.get(key, -1.0)) != value:
                raise ValueError(f"stage1b requires {key}={value}")
        if config.get("decoder_trainable_policy") != STAGE1B_DECODER_POLICY:
            raise ValueError(
                f"stage1b requires decoder_trainable_policy={STAGE1B_DECODER_POLICY}"
            )
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
    data_fps = float(data_config.get("target_fps", 0.0))
    model_fps = float(video.get("target_fps", 0.0))
    if data_fps <= 0.0 or model_fps <= 0.0:
        raise ValueError(
            "model.video.target_fps and data.target_fps must be positive"
        )
    if abs(data_fps - model_fps) > 1.0e-9:
        raise ValueError(
            f"model.video.target_fps={model_fps} does not match "
            f"data.target_fps={data_fps}"
        )
    ratio = float(data_config.get("min_native_fps_ratio", 0.0))
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("data.min_native_fps_ratio must be in [0,1]")
    model_frames, model_latents = (17, 5) if stage == "stage1a_plus" else (
        expected_frames, expected_latents
    )
    if int(video.get("num_frames", -1)) != model_frames:
        raise ValueError(f"model.video.num_frames must be {model_frames} for {stage}")
    if int(state.get("num_frames", -1)) != model_latents:
        raise ValueError(f"model.state.num_frames must be {model_latents} for {stage}")

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


def _configure_temporal_interface(decoder: nn.Module, *, pre_decoder: bool) -> None:
    """Freeze Wan weights selectively while keeping its checkpoint path in train mode."""

    decoder.requires_grad_(False)
    decoder.train(True)
    _set_trainable(decoder.temporal_adapter, True)
    if not pre_decoder:
        return
    _set_trainable(decoder.pre_decoder, True)
    _set_trainable(decoder.decoder.conv1, True)
    for name, module in decoder.decoder.named_modules():
        if name.endswith("time_conv"):
            _set_trainable(module, True)


def configure_stage(
    model: ProgressiveVideoRAE,
    stage: str,
    *,
    optimizer_step: int = 0,
    wan_interface_step: int = 2000,
    wan_full_step: int = 5000,
    repa_trainable: bool = True,
) -> None:
    if stage not in STAGES:
        raise ValueError(f"Unsupported stage {stage}; expected one of {STAGES}")
    _set_trainable(model.encoder, False)
    if stage == "stage1a":
        _set_trainable(model.projector, True)
        _set_trainable(model.repa_projection, repa_trainable)
        model.projector.shared_mask_set.requires_grad_(False)
    elif stage in ("stage1a_plus", "stage1b"):
        _set_trainable(model.projector, True)
        _set_trainable(model.repa_projection, False)
        if stage == "stage1a_plus":
            model.projector.shared_mask_set.requires_grad_(False)
    else:
        _set_trainable(model.projector, False)
        _set_trainable(model.repa_projection, False)

    if stage in ("stage1a_plus", "stage1b"):
        _configure_temporal_interface(model.decoder, pre_decoder=True)
        return
    if stage != "stage1a":
        _set_trainable(model.decoder, True)
        return

    phase = stage1a_phase(
        optimizer_step,
        wan_interface_step=wan_interface_step,
        wan_full_step=wan_full_step,
    )
    _configure_temporal_interface(
        model.decoder, pre_decoder=phase in ("interface", "full")
    )
    if phase == "full":
        _set_trainable(model.decoder, True)


def sample_microbatch_tasks(
    stage: str,
    accumulation_steps: int,
    *,
    optimizer_step: int = 0,
) -> tuple[MicrobatchTask, ...]:
    if stage in ("stage1a", "stage1a_plus", "stage2a", "stage2b"):
        return tuple(MicrobatchTask("full") for _ in range(accumulation_steps))
    if stage != "stage1b":
        raise ValueError(f"Unsupported stage: {stage}")
    if accumulation_steps != 8:
        raise ValueError("Stage 1B requires exactly 8 microbatches per optimizer step")
    tasks = [
        MicrobatchTask("single_prefix", random.randint(0, 46)) for _ in range(7)
    ]
    tasks.append(MicrobatchTask("full"))
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


def repa_factor(
    optimizer_step: int,
    start_step: int = 0,
    ramp_steps: int = 0,
    max_factor: float = 1.0,
) -> float:
    """Return the effective full-state REPA multiplier for one optimizer step."""

    if optimizer_step < 0:
        raise ValueError("REPA optimizer step must be non-negative")
    if start_step < 0 or ramp_steps < 0:
        raise ValueError("REPA start and ramp steps must be non-negative")
    if max_factor < 0.0:
        raise ValueError("REPA max factor must be non-negative")
    if optimizer_step < start_step:
        return 0.0
    if ramp_steps == 0:
        return float(max_factor)
    progress = min(
        1.0, max(0.0, (optimizer_step - start_step) / ramp_steps)
    )
    return float(max_factor) * progress
