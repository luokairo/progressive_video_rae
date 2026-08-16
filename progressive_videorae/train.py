from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from .config import load_training_bundle
from .data import DistributedFullDatasetSampler, VideoManifestDataset, collate_video_samples
from .model.factory import build_model
from .training.checkpoint import (
    load_checkpoint,
    representation_identity,
    save_checkpoint,
    verify_pretrained_checkpoint_hashes,
    unwrap,
)
from .training.logging import (
    GlobalMetricWindow,
    append_jsonl_record,
    gradient_norm,
    resolve_training_log_file,
)
from .training.paths import (
    resolve_training_run_paths,
    utc_run_id,
    validate_external_absolute_path,
    validate_resume_log_file,
)
from .training.losses import PatchDiscriminator, ProgressiveLosses
from .training.stages import (
    adversarial_factor,
    configure_stage,
    repa_factor,
    sample_microbatch_tasks,
    stage1a_phase,
    validate_stage_objective,
    validate_training_batch,
    validate_training_bundle,
)


def broadcast_rank0_object(value: Any, *, rank: int) -> Any:
    payload = [value if rank == 0 else None]
    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        torch.distributed.broadcast_object_list(payload, src=0)
    elif rank != 0:
        raise RuntimeError("Nonzero rank requires an initialized distributed process group")
    result = payload[0]
    if result is None:
        raise RuntimeError("Rank 0 broadcast returned no payload")
    return result


def distributed_context() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def distributed_run_id(*, rank: int, resume_checkpoint: str | None) -> str:
    if resume_checkpoint is not None:
        return Path(resume_checkpoint).expanduser().resolve().parent.name
    return str(broadcast_rank0_object(utc_run_id() if rank == 0 else None, rank=rank))


def distributed_verified_representation_identity(
    model: nn.Module, *, rank: int
) -> dict[str, Any]:
    result: dict[str, Any] | None = None
    if rank == 0:
        try:
            hashes = verify_pretrained_checkpoint_hashes(
                model, create_missing_sidecars=True
            )
        except Exception as exc:
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        else:
            result = {"ok": True, "hashes": hashes}
    result = broadcast_rank0_object(result, rank=rank)
    if not isinstance(result, dict) or result.get("ok") is not True:
        error = result.get("error") if isinstance(result, dict) else repr(result)
        raise RuntimeError(f"Rank 0 pretrained SHA256 preflight failed: {error}")
    hashes = result.get("hashes")
    if not isinstance(hashes, dict):
        raise RuntimeError("Rank 0 SHA256 preflight returned no verified hashes")
    return representation_identity(model, checkpoint_hashes=hashes)


def seed_everything(seed: int, rank: int) -> None:
    value = seed + rank
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def cosine_scheduler(
    optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
):
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in [0,1]")
    if warmup_steps < 0 or total_steps <= 0:
        raise ValueError("Scheduler requires warmup_steps >= 0 and total_steps > 0")

    def scale(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _set_nested(bundle: dict[str, Any], expression: str) -> None:
    if "=" not in expression:
        raise ValueError(f"Override must be key=value: {expression}")
    path, raw = expression.split("=", 1)
    try:
        import yaml

        value = yaml.safe_load(raw)
    except ImportError:
        value = raw
    keys = path.split(".")
    target: dict[str, Any] = bundle
    for key in keys[:-1]:
        child = target.get(key)
        if not isinstance(child, dict):
            raise KeyError(f"Override parent is not a mapping: {path}")
        target = child
    target[keys[-1]] = value


class CyclingLoader:
    def __init__(self, loader: DataLoader, sampler: DistributedFullDatasetSampler, epoch: int = 0):
        self.loader = loader
        self.sampler = sampler
        self.epoch = epoch
        self.sampler.set_epoch(epoch)
        self.iterator = iter(loader)

    def next(self) -> dict[str, Any]:
        try:
            return next(self.iterator)
        except StopIteration:
            self.epoch += 1
            self.sampler.set_epoch(self.epoch)
            self.iterator = iter(self.loader)
            return next(self.iterator)


class CudaEventLedger:
    """Collect CUDA timings only on logging steps, with one final synchronization."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = bool(enabled and torch.cuda.is_available())
        self.pairs: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}

    def start(self):
        if not self.enabled:
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def stop(self, name: str, start) -> None:
        if start is None:
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self.pairs.setdefault(name, []).append((start, end))

    def resolve(self) -> dict[str, float]:
        if not self.enabled:
            return {}
        return {
            name: sum(start.elapsed_time(end) for start, end in pairs) / 1000.0
            for name, pairs in self.pairs.items()
        }
def training_phase(training: dict[str, Any], optimizer_step: int) -> str:
    if training["stage"] != "stage1a":
        return str(training["stage"])
    return stage1a_phase(
        optimizer_step,
        wan_interface_step=int(training.get("wan_interface_step", 2000)),
        wan_full_step=int(training.get("wan_full_step", 5000)),
    )


def checkpointing_for_phase(training: dict[str, Any], phase: str) -> bool:
    if training["stage"] != "stage1a":
        return bool(training.get("gradient_checkpointing", True))
    schedule = training.get("gradient_checkpointing_by_phase", {})
    if phase not in schedule:
        raise ValueError(f"Missing Stage 1A checkpoint policy for phase {phase}")
    return bool(schedule[phase])


def _wrap_ddp(module: nn.Module, local_rank: int) -> DistributedDataParallel:
    return DistributedDataParallel(
        module,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )


def reconfigure_distributed_phase(
    model: nn.Module,
    *,
    training: dict[str, Any],
    optimizer_step: int,
    active_phase: str,
    local_rank: int,
    world_size: int,
) -> tuple[nn.Module, str]:
    requested_phase = training_phase(training, optimizer_step)
    if requested_phase == active_phase:
        return model, active_phase
    if world_size > 1:
        torch.distributed.barrier()
    module = unwrap(model)
    configure_stage(
        module,
        str(training["stage"]),
        optimizer_step=optimizer_step,
        wan_interface_step=int(training.get("wan_interface_step", 2000)),
        wan_full_step=int(training.get("wan_full_step", 5000)),
        repa_trainable=float(training.get("repa_max_factor", 1.0)) > 0.0,
    )
    module.decoder.enable_gradient_checkpointing(
        checkpointing_for_phase(training, requested_phase)
    )
    model = _wrap_ddp(module, local_rank) if world_size > 1 else module
    if world_size > 1:
        torch.distributed.barrier()
    return model, requested_phase


def assert_finite_distributed(loss: torch.Tensor, name: str) -> None:
    failed = (~torch.isfinite(loss.detach())).to(dtype=torch.int32)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(failed, op=torch.distributed.ReduceOp.MAX)
    if int(failed.item()):
        raise FloatingPointError(f"Non-finite distributed loss: {name}")


def verify_optimizer_gradients_synchronized(optimizer: torch.optim.Optimizer) -> None:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return
    world_size = torch.distributed.get_world_size()
    for group_index, group in enumerate(optimizer.param_groups):
        gradients = [
            parameter.grad.detach()
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        device = group["params"][0].device
        if gradients:
            checksum = torch.stack(
                (
                    torch.tensor(float(len(gradients)), device=device, dtype=torch.float64),
                    sum(gradient.double().sum() for gradient in gradients),
                    sum(gradient.double().square().sum() for gradient in gradients),
                )
            )
        else:
            checksum = torch.zeros(3, device=device, dtype=torch.float64)
        gathered = [torch.empty_like(checksum) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, checksum)
        reference = gathered[0]
        if any(
            not torch.allclose(value, reference, rtol=1e-6, atol=1e-6)
            for value in gathered[1:]
        ):
            name = group.get("name", str(group_index))
            raise RuntimeError(f"DDP gradient mismatch for optimizer group {name}")


def reduce_scalar_metrics(metrics: dict[str, float], device: torch.device) -> dict[str, float]:
    if not metrics:
        return {}
    keys = sorted(metrics)
    values = torch.tensor([metrics[key] for key in keys], device=device, dtype=torch.float64)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
        values.div_(torch.distributed.get_world_size())
    return {key: float(value) for key, value in zip(keys, values.cpu().tolist())}


def _unique(parameters):
    seen = set()
    result = []
    for parameter in parameters:
        if id(parameter) not in seen:
            seen.add(id(parameter))
            result.append(parameter)
    return result


def _has_gradient(parameters) -> int:
    return int(any(parameter.grad is not None for parameter in parameters))


def build_optimizers(model, discriminator, config: dict[str, Any]):
    module = unwrap(model)
    fast_parameters = list(module.projector.parameters()) + list(
        module.repa_projection.parameters()
    )
    interface = list(module.decoder.pre_decoder.parameters()) + list(
        module.decoder.decoder.conv1.parameters()
    )
    if config.get("stage") in ("stage1a_plus", "stage1b"):
        interface = list(module.decoder.temporal_adapter.parameters()) + interface
    else:
        fast_parameters += list(module.decoder.temporal_adapter.parameters())
    fast = _unique(fast_parameters)
    for name, child in module.decoder.decoder.named_modules():
        if name.endswith("time_conv"):
            interface.extend(child.parameters())
    interface = _unique(interface)
    used = {id(parameter) for parameter in fast + interface}
    spatial = [
        parameter
        for parameter in module.decoder.parameters()
        if id(parameter) not in used
    ]
    weight_decay_by_group = config.get("weight_decay_by_group", {})
    gradient_clip_by_group = config.get("gradient_clip_by_group", {})
    if not isinstance(weight_decay_by_group, dict):
        raise TypeError("weight_decay_by_group must be a mapping")
    if not isinstance(gradient_clip_by_group, dict):
        raise TypeError("gradient_clip_by_group must be a mapping")
    default_weight_decay = float(config["weight_decay"])
    default_gradient_clip = float(config["gradient_clip"])
    groups = [
        {
            "params": fast,
            "lr": float(config.get("projector_lr", 1e-4)),
            "name": "rae_fast",
            "weight_decay": float(weight_decay_by_group.get("rae_fast", default_weight_decay)),
            "max_grad_norm": float(gradient_clip_by_group.get("rae_fast", default_gradient_clip)),
        },
        {
            "params": interface,
            "lr": float(config.get("wan_temporal_lr", 2e-5)),
            "name": "wan_temporal",
            "weight_decay": float(
                weight_decay_by_group.get("wan_temporal", default_weight_decay)
            ),
            "max_grad_norm": float(
                gradient_clip_by_group.get("wan_temporal", default_gradient_clip)
            ),
        },
        {
            "params": spatial,
            "lr": float(config.get("wan_spatial_lr", 5e-6)),
            "name": "wan_spatial",
            "weight_decay": float(
                weight_decay_by_group.get("wan_spatial", default_weight_decay)
            ),
            "max_grad_norm": float(
                gradient_clip_by_group.get("wan_spatial", default_gradient_clip)
            ),
        },
    ]
    fused = bool(config.get("fused_optimizer", False))
    generator = torch.optim.AdamW(
        groups,
        betas=(0.9, 0.95),
        weight_decay=config["weight_decay"],
        fused=fused,
    )
    disc = torch.optim.AdamW(
        discriminator.parameters(),
        lr=config["discriminator_lr"],
        betas=(0.5, 0.9),
        weight_decay=0.0,
        fused=fused,
    )
    return generator, disc


def clip_optimizer_gradients(
    optimizer: torch.optim.Optimizer,
    *,
    default_max_norm: float,
) -> dict[str, float]:
    """Clip optimizer groups independently while preserving legacy norm logs."""

    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    total_pre = gradient_norm(parameters)
    metrics = {"grad_norm/generator": float(total_pre) if total_pre is not None else 0.0}
    for group_index, group in enumerate(optimizer.param_groups):
        name = str(group.get("name", group_index))
        max_norm = float(group.get("max_grad_norm", default_max_norm))
        if max_norm <= 0.0:
            raise ValueError(f"Gradient clip for {name} must be positive")
        pre = torch.nn.utils.clip_grad_norm_(
            group["params"],
            max_norm,
            error_if_nonfinite=True,
        )
        post = gradient_norm(group["params"])
        pre_value = float(pre)
        post_value = float(post) if post is not None else 0.0
        scale = 1.0 if pre_value <= max_norm else max_norm / max(pre_value, 1.0e-12)
        metrics[f"grad_norm/{name}"] = pre_value
        metrics[f"grad_norm_post/{name}"] = post_value
        metrics[f"grad_clip_scale/{name}"] = scale
        metrics[f"grad_clip_max/{name}"] = max_norm
    total_post = gradient_norm(parameters)
    metrics["grad_norm_post/generator"] = (
        float(total_post) if total_post is not None else 0.0
    )
    return metrics


def preflight_pretrained_weights(model: nn.Module, report_path: Path | None = None) -> dict:
    module = unwrap(model)
    module.assert_pretrained_ready()
    report = module.pretrained_load_report()
    if not report["ready"]:
        raise RuntimeError("Pretrained report is not ready")
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def verify_upstream_commit(source_root: str, expected_commit: str, component: str) -> None:
    root = Path(source_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"{component} source root is unavailable: {root}")
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    actual = result.stdout.strip()
    if actual != expected_commit:
        raise RuntimeError(f"{component} source commit mismatch: {actual} != {expected_commit}")


def _sequence_id(batch: dict[str, Any]) -> str:
    ids = []
    if "codec_sequence_id" in batch:
        return "|".join(batch["codec_sequence_id"])
    for sample_id, frame_indices, timestamps in zip(
        batch["sample_id"],
        batch["sampled_frame_indices"],
        batch["sampled_timestamps"],
    ):
        index_text = ",".join(str(int(value)) for value in frame_indices.tolist())
        timestamp_text = ",".join(
            f"{float(value):.9f}" for value in timestamps.tolist()
        )
        ids.append(f"{sample_id}|indices={index_text}|timestamps={timestamp_text}")
    return "|".join(ids)


def _prefix_loss(losses, output, pixel_values, endpoint, previous_prediction=None):
    full_target = pixel_values.mul(2.0).sub(1.0)
    with torch.autocast("cuda", enabled=False):
        return losses.prefix(
            output,
            full_target,
            endpoint=endpoint,
            previous_prediction=previous_prediction,
        )

PREFIX_WINDOW_TERMS = (
    "l1",
    "band",
    "leakage",
    "paired_delta",
    "prefix_repa_global",
    "prefix_repa_local",
    "prefix_repa_local_scaled",
)

PREFIX_WINDOW_STATISTICS = (
    "prefix_repa/level",
    "prefix_repa/grid_height",
    "prefix_repa/grid_width",
    "prefix_repa/local_scale",
    "prefix_repa/anchor_global_error",
    "prefix_repa/anchor_local_error",
) + tuple(
    f"prefix_repa/group_{group_index:02d}/{phase_name}_{scope}_error"
    for group_index in range(4)
    for phase_name in ("f0", "f1")
    for scope in ("global", "local")
)


def _prefix_window_metric_names() -> tuple[str, ...]:
    return tuple(
        f"prefix/{kind}/endpoint_{endpoint:02d}/{metric}"
        for kind in ("single", "paired")
        for endpoint in range(48)
        for metric in PREFIX_WINDOW_TERMS + PREFIX_WINDOW_STATISTICS
    )


def record_prefix_window(window, *, kind: str, endpoint: int, loss) -> None:
    window.add_prefix(endpoint, kind=kind)
    for term in PREFIX_WINDOW_TERMS:
        window.add_mean(
            f"prefix/{kind}/endpoint_{endpoint:02d}/{term}",
            loss.terms[term],
        )
    for name in PREFIX_WINDOW_STATISTICS:
        if name in loss.statistics:
            window.add_mean(
                f"prefix/{kind}/endpoint_{endpoint:02d}/{name}",
                loss.statistics[name],
            )


def forward_training_batch(
    model: nn.Module,
    pixel_values: torch.Tensor,
    *,
    task,
    stage: str,
    sequence_id: str | None = None,
):
    if stage in ("stage2a", "stage2b") and not task.is_full:
        raise RuntimeError("Stage 2 cannot execute a spatial-prefix task")
    endpoint = None if task.is_full else task.endpoint
    kwargs = {
        "endpoint": endpoint,
        "paired_previous_endpoint": task.previous_endpoint,
        "cache_mode": "reset" if stage == "stage2b" else "disabled",
        "return_decoder_features": (
            task.is_full or (stage == "stage1b" and task.kind == "single_prefix")
        ),
        "sequence_id": sequence_id,
    }
    return model(pixel_values, **kwargs)


def validate_resume_training_contract(
    checkpoint: dict[str, Any],
    training: dict[str, Any],
    data_config: dict[str, Any] | None = None,
) -> None:
    saved_bundle = checkpoint.get("config")
    saved = saved_bundle.get("training") if isinstance(saved_bundle, dict) else None
    if not isinstance(saved, dict):
        raise RuntimeError("Resume checkpoint is missing its resolved training config")
    saved_data = saved_bundle.get("data") if isinstance(saved_bundle, dict) else None
    if data_config is not None:
        if not isinstance(saved_data, dict):
            raise RuntimeError("Resume checkpoint is missing its resolved data config")
        data_keys = ("num_frames", "target_fps", "min_native_fps_ratio")
        data_mismatches = {
            key: (saved_data.get(key, 0.0), data_config.get(key, 0.0))
            for key in data_keys
            if saved_data.get(key, 0.0) != data_config.get(key, 0.0)
        }
        if data_mismatches:
            raise RuntimeError(
                f"Resume sampling contract mismatch: {data_mismatches}"
            )
    keys = ("micro_batch_size", "gradient_accumulation_steps", "global_batch_size", "max_steps")
    if training.get("stage") == "stage1a":
        keys += (
            "wan_interface_step",
            "wan_full_step",
            "projector_lr",
            "wan_temporal_lr",
            "wan_spatial_lr",
            "weight_decay",


            "weight_decay_by_group",
            "warmup_steps",
            "min_lr_ratio",
            "gradient_clip",
            "gradient_clip_by_group",
            "l1_weight",
            "lpips_weight",
            "temporal_l1_weight",
            "repa_local_weight",
            "repa_global_weight",
            "repa_start_step",
            "repa_ramp_steps",
            "repa_max_factor",
            "adversarial_weight",
            "disc_start",
            "adversarial_ramp_steps",
            "gradient_checkpointing_by_phase",
            "fused_optimizer",
        )
    if training.get("stage") == "stage1b":
        keys += (
            "prefix_schedule",
            "p47_full_microbatches_per_step",
            "prefix_min",
            "prefix_max",
            "prefix_objective_weight",
            "p47_objective_weight",
            "prefix_lpips_weight",
            "prefix_repa_schedule",
            "prefix_repa_levels",
            "prefix_repa_global_weight",
            "prefix_repa_local_weight",
            "repa_local_weight",
            "repa_global_weight",
            "band_weight",
            "leakage_weight",
            "adversarial_weight",
            "disc_start",
            "adversarial_ramp_steps",
            "decoder_trainable_policy",
            "fused_optimizer",
        )
    mismatches = {
        key: (saved.get(key), training.get(key))
        for key in keys
        if saved.get(key) != training.get(key)
    }
    if mismatches:
        raise RuntimeError(f"Resume training contract mismatch: {mismatches}")


def validate_init_sampling_contract(
    checkpoint: dict[str, Any], data_config: dict[str, Any]
) -> None:
    saved_bundle = checkpoint.get("config")
    saved_data = saved_bundle.get("data") if isinstance(saved_bundle, dict) else None
    if not isinstance(saved_data, dict):
        raise RuntimeError("Init checkpoint is missing its resolved data config")
    keys = ("target_fps", "min_native_fps_ratio")
    mismatches = {
        key: (saved_data.get(key, 0.0), data_config.get(key, 0.0))
        for key in keys
        if saved_data.get(key, 0.0) != data_config.get(key, 0.0)
    }
    if mismatches:
        raise RuntimeError(f"Init sampling contract mismatch: {mismatches}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ProgressiveVideoRAE Stage 1A/1B/2-A/2-B")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--init-from", default=None)
    parser.add_argument("--selection-certificate", default=None)
    parser.add_argument("--allow-smoke-checkpoint", action="store_true")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()
    if args.resume is not None and args.init_from is not None:
        parser.error("--resume and --init-from are mutually exclusive")
    if args.selection_certificate is not None and args.init_from is None:
        parser.error("--selection-certificate requires --init-from")
    if args.selection_certificate is not None and args.allow_smoke_checkpoint:
        parser.error("--selection-certificate is only valid for formal training")
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA")

    bundle = load_training_bundle(args.config, model_config_path=args.model_config)
    for expression in args.set:
        _set_nested(bundle, expression)
    training, model_config, data_config = bundle["training"], bundle["model"], bundle["data"]
    expected_frames, _ = validate_training_bundle(training, model_config, data_config)
    objective_mode = validate_stage_objective(training)
    verify_upstream_commit(
        model_config["encoder"]["source_root"],
        "204698b45b3712590f06245fbfba32d3be539812",
        "V-JEPA2",
    )
    verify_upstream_commit(
        model_config["decoder"]["source_root"],
        "42bf4cfaa384bc21833865abc2f9e6c0e67233dc",
        "Wan2.2",
    )
    stage = training["stage"]
    if stage == "stage1a_plus":
        parser.error("Stage 1-A-plus must use progressive_videorae.train_stage1a_plus")
    if stage == "stage1a" and args.init_from is not None:
        parser.error("Stage 1-A must start fresh; --init-from is not allowed")
    if stage != "stage1a" and args.resume is None and args.init_from is None:
        parser.error(f"{stage} requires --init-from from the previous completed stage")
    run_mode = "smoke" if args.allow_smoke_checkpoint else "formal"

    rank, local_rank, world_size = distributed_context()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    seed_everything(args.seed, rank)

    expected_global = (
        training["micro_batch_size"]
        * training["gradient_accumulation_steps"]
        * world_size
    )
    if expected_global != training["global_batch_size"]:
        raise ValueError("global_batch_size does not match the distributed topology")

    dataset = VideoManifestDataset(
        Path(data_config["manifest_dir"]) / "train.parquet",
        split="train",
        num_frames=data_config["num_frames"],
        target_fps=data_config["target_fps"],
        height=data_config["height"],
        width=data_config["width"],
        horizontal_flip=data_config.get("train_horizontal_flip", True),
        min_native_fps_ratio=float(data_config.get("min_native_fps_ratio", 0.0)),
    )
    sampler = DistributedFullDatasetSampler(
        dataset,
        seed=args.seed,
        rank=rank,
        world_size=world_size,
        samples_per_rank_multiple=training["micro_batch_size"],
    )
    loader_options: dict[str, Any] = {}
    if data_config["num_workers"] > 0:
        loader_options["prefetch_factor"] = int(data_config.get("prefetch_factor", 2))
    loader = DataLoader(
        dataset,
        batch_size=training["micro_batch_size"],
        sampler=sampler,
        num_workers=data_config["num_workers"],
        pin_memory=True,
        persistent_workers=data_config["num_workers"] > 0,
        collate_fn=collate_video_samples,
        drop_last=True,
        **loader_options,
    )
    cycling = CyclingLoader(loader, sampler)

    run_id = distributed_run_id(rank=rank, resume_checkpoint=args.resume)
    run_paths = resolve_training_run_paths(
        training,
        stage=stage,
        run_id=run_id,
        resume_checkpoint=args.resume,
    )
    checkpoint_dir = run_paths.checkpoint_dir
    source_checkpoint = (
        str(Path(args.resume or args.init_from).expanduser().resolve())
        if args.resume or args.init_from
        else None
    )
    if args.init_from is not None and not args.allow_smoke_checkpoint:
        source_path = validate_external_absolute_path(
            args.init_from, label="init-from checkpoint"
        )
        if not source_path.is_relative_to(
            run_paths.checkpoint_root
        ):
            raise ValueError("Formal init-from checkpoint must be under checkpoint_root")
        if args.selection_certificate is not None:
            certificate_path = validate_external_absolute_path(
                args.selection_certificate, label="selection certificate"
            )
            if not certificate_path.is_relative_to(run_paths.checkpoint_root):
                raise ValueError(
                    "Formal selection certificate must be under checkpoint_root"
                )
    if not args.resume and tuple(checkpoint_dir.glob("step_*.pt")):
        raise FileExistsError(f"Refusing to overwrite existing run: {checkpoint_dir}")
    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        torch.distributed.barrier()

    model = build_model(model_config, validate_pretrained=False)
    report = preflight_pretrained_weights(
        model, checkpoint_dir / "pretrained_load_report.json" if rank == 0 else None
    )
    if rank == 0:
        print(json.dumps({"pretrained": report}, ensure_ascii=False), flush=True)
    static_identity = distributed_verified_representation_identity(model, rank=rank)
    model = model.to(device)
    active_phase = training_phase(training, 0)
    configure_stage(
        model,
        stage,
        optimizer_step=0,
        wan_interface_step=int(training.get("wan_interface_step", 2000)),
        wan_full_step=int(training.get("wan_full_step", 5000)),
        repa_trainable=float(training.get("repa_max_factor", 1.0)) > 0.0,
    )
    model.decoder.enable_gradient_checkpointing(
        checkpointing_for_phase(training, active_phase)
    )
    discriminator: nn.Module = PatchDiscriminator().to(device)
    if world_size > 1:
        model = _wrap_ddp(model, local_rank)
        discriminator = DistributedDataParallel(
            discriminator, device_ids=[local_rank], output_device=local_rank
        )

    losses = ProgressiveLosses(
        l1_weight=training.get("l1_weight", 1.0),
        prefix_lpips_weight=training.get("prefix_lpips_weight", 0.5),
        lpips_weight=training.get("lpips_weight", 1.0),
        repa_local_weight=training.get("repa_local_weight", 1.0),
        repa_global_weight=training.get("repa_global_weight", 1.0),
        prefix_repa_local_weight=training.get("prefix_repa_local_weight", 0.0),
        prefix_repa_global_weight=training.get(
            "prefix_repa_global_weight", 0.0
        ),
        adversarial_weight=training.get("adversarial_weight", 0.1),
        temporal_l1_weight=training.get("temporal_l1_weight", 0.1),
        band_weight=training.get("band_weight", 1.0),
        leakage_weight=training.get("leakage_weight", 0.1),
        paired_delta_weight=training.get("paired_delta_weight", 1.0),
    ).to(device)
    generator_optimizer, discriminator_optimizer = build_optimizers(
        model, discriminator, training
    )
    generator_scheduler = cosine_scheduler(
        generator_optimizer,
        training["warmup_steps"],
        training["max_steps"],
        min_lr_ratio=float(training.get("min_lr_ratio", 0.0)),
    )
    discriminator_scheduler = cosine_scheduler(
        discriminator_optimizer,
        training["warmup_steps"],
        training["max_steps"],
        min_lr_ratio=float(training.get("min_lr_ratio", 0.0)),
    )

    optimizer_step = 0
    discriminator_update_count = 0
    checkpoint = None
    if args.init_from:
        checkpoint = load_checkpoint(
            args.init_from,
            model=model,
            discriminator=discriminator,
            restore_rng=False,
            static_identity=static_identity,
            target_stage=stage,
            target_objective_mode=objective_mode,
            load_mode="init",
            allow_smoke_checkpoint=args.allow_smoke_checkpoint,
            selection_certificate=args.selection_certificate,
        )
        validate_init_sampling_contract(checkpoint, data_config)
    if args.resume:
        checkpoint = load_checkpoint(
            args.resume,
            model=model,
            discriminator=discriminator,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            generator_scheduler=generator_scheduler,
            discriminator_scheduler=discriminator_scheduler,
            static_identity=static_identity,
            target_stage=stage,
            target_objective_mode=objective_mode,
            load_mode="resume",
            allow_smoke_checkpoint=args.allow_smoke_checkpoint,
        )
        validate_resume_training_contract(checkpoint, training, data_config)
        if checkpoint.get("run_id") != run_id:
            raise RuntimeError("Resume checkpoint run_id does not match its run directory")
        saved_checkpoint_dir = checkpoint.get("checkpoint_dir")
        if not isinstance(saved_checkpoint_dir, str) or (
            Path(saved_checkpoint_dir).expanduser().resolve() != checkpoint_dir
        ):
            raise RuntimeError("Resume checkpoint records a different checkpoint directory")
        if not isinstance(checkpoint.get("log_file"), str):
            raise RuntimeError("Resume checkpoint is missing its original log_file")

        optimizer_step = int(checkpoint["optimizer_step"])
        discriminator_update_count = int(checkpoint.get("discriminator_update_count", 0))
        cycling.epoch = int(checkpoint["epoch"])
        cycling.sampler.set_epoch(cycling.epoch)
        cycling.iterator = iter(cycling.loader)

    if args.resume:
        log_file = validate_resume_log_file(
            checkpoint["log_file"],
            log_root=run_paths.log_root,
        )
    else:
        log_file = resolve_training_log_file(
            training,
            checkpoint_dir,
            run_id=run_id,
        )
    resolved_bundle = copy.deepcopy(bundle)
    resolved_bundle["runtime"] = {
        "run_id": run_id,
        "checkpoint_dir": str(checkpoint_dir),
        "log_file": str(log_file),
        "source_checkpoint": source_checkpoint,
        "source_selection_certificate": (
            str(Path(args.selection_certificate).expanduser().resolve())
            if args.selection_certificate is not None
            else None
        ),
        "run_mode": run_mode,
    }
    bundle = resolved_bundle
    if rank == 0:
        (checkpoint_dir / "resolved_config.json").write_text(
            json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    if world_size > 1:
        torch.distributed.barrier()
    append_jsonl_record(
        log_file,
        {
            "event": "training_start" if not args.resume else "training_resume",
            "step": optimizer_step,
            "stage": stage,
            "objective_mode": objective_mode,
            "run_mode": run_mode,
            "data_filter_stats": dataset.filter_stats,
        },
        rank=rank,
    )

    autocast_dtype = torch.bfloat16 if training["precision"] == "bf16" else torch.float16
    accumulation = int(training["gradient_accumulation_steps"])
    prefix_objective_weight = float(training.get("prefix_objective_weight", 1.0))
    p47_objective_weight = float(training.get("p47_objective_weight", 1.0))
    prefix_window = GlobalMetricWindow(
        _prefix_window_metric_names(), device=device, prefix_bins=48
    )

    while optimizer_step < training["max_steps"]:
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        collect_timing = rank == 0 and (
            (optimizer_step + 1) % int(training["log_every"]) == 0
        )
        timing = CudaEventLedger(collect_timing)
        model, active_phase = reconfigure_distributed_phase(
            model,
            training=training,
            optimizer_step=optimizer_step,
            active_phase=active_phase,
            local_rank=local_rank,
            world_size=world_size,
        )
        checkpoint_enabled = checkpointing_for_phase(training, active_phase)
        unwrap(model).decoder.enable_gradient_checkpointing(checkpoint_enabled)
        unwrap(model).enable_runtime_timing(collect_timing)
        schedule_step = optimizer_step
        tasks = sample_microbatch_tasks(
            stage, accumulation, optimizer_step=schedule_step
        )
        if stage in ("stage2a", "stage2b") and any(not task.is_full for task in tasks):
            raise RuntimeError("Stage 2 must never construct a spatial-prefix task")
        generator_optimizer.zero_grad(set_to_none=True)
        discriminator_optimizer.zero_grad(set_to_none=True)
        metrics: dict[str, float] = {}
        full_count = sum(task.is_full for task in tasks)
        gan_factor = adversarial_factor(
            optimizer_step,
            int(training.get("disc_start", 0)),
            int(training.get("adversarial_ramp_steps", 0)),
        )
        current_repa_factor = repa_factor(
            optimizer_step,
            int(training.get("repa_start_step", 0)),
            int(training.get("repa_ramp_steps", 0)),
            float(training.get("repa_max_factor", 1.0)),
        )
        gan_active = (
            full_count > 0
            and gan_factor > 0.0
            and losses.adversarial_weight > 0.0
        )
        discriminator_forwards = 0

        data_seconds = 0.0

        for micro_index, task in enumerate(tasks):
            data_started = time.perf_counter()
            batch = cycling.next()
            validate_training_batch(
                batch,
                stage=stage,
                expected_frames=expected_frames,
                expected_height=int(data_config["height"]),
                expected_width=int(data_config["width"]),
            )
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            data_seconds += time.perf_counter() - data_started
            last_micro = micro_index == len(tasks) - 1
            sync = nullcontext() if last_micro or world_size == 1 else model.no_sync()
            with sync, torch.autocast("cuda", dtype=autocast_dtype):
                forward_event = timing.start()
                result = forward_training_batch(
                    model,
                    pixel_values,
                    task=task,
                    stage=stage,
                    sequence_id=_sequence_id(batch),
                )
                timing.stop("model_forward", forward_event)
                loss_event = timing.start()
                if task.is_full:
                    assert not isinstance(result, tuple)
                    if result.repa_reference is None or result.repa_features is None:
                        raise RuntimeError(f"{stage} full objective requires REPA")
                    if gan_active:
                        disc_loss = losses.discriminator(
                            discriminator,
                            result.reconstruction,
                            result.target,
                            adversarial_factor=gan_factor,
                        )
                        assert_finite_distributed(disc_loss.total, "discriminator")
                        (disc_loss.total / full_count).backward()
                        discriminator_forwards += 2
                        for name, value in disc_loss.terms.items():
                            metrics[name] = metrics.get(name, 0.0) + (
                                float(value.detach()) / full_count
                            )
                    for parameter in discriminator.parameters():
                        parameter.requires_grad_(False)
                    generator_loss = losses.full_generator(
                        result,
                        discriminator,
                        adversarial_factor=gan_factor,
                        repa_factor=current_repa_factor,
                    )
                    if gan_active:
                        discriminator_forwards += 1
                    if stage == "stage1b":
                        generator_loss.total = (
                            p47_objective_weight * generator_loss.total
                        )
                        for name, value in generator_loss.terms.items():
                            metrics[f"p47/{name}"] = float(value.detach())
                        for name, value in generator_loss.statistics.items():
                            metrics[f"p47/{name}"] = float(value.detach())
                    for parameter in discriminator.parameters():
                        parameter.requires_grad_(True)
                elif task.kind == "single_prefix":
                    assert not isinstance(result, tuple)
                    generator_loss = _prefix_loss(
                        losses, result, pixel_values, task.endpoint
                    )
                    generator_loss.total = prefix_objective_weight * generator_loss.total
                    record_prefix_window(
                        prefix_window,
                        kind="single",
                        endpoint=task.endpoint,
                        loss=generator_loss,
                    )
                else:
                    assert isinstance(result, tuple)
                    previous, current = result
                    previous_loss = _prefix_loss(
                        losses, previous, pixel_values, int(task.previous_endpoint)
                    )
                    current_loss = _prefix_loss(
                        losses,
                        current,
                        pixel_values,
                        task.endpoint,
                        previous_prediction=previous.reconstruction,
                    )
                    generator_loss = current_loss
                    generator_loss.total = prefix_objective_weight * 0.5 * (
                        previous_loss.total + current_loss.total
                    )
                    record_prefix_window(
                        prefix_window, kind="paired", endpoint=task.endpoint, loss=current_loss
                    )
                assert_finite_distributed(generator_loss.total, "generator")
                timing.stop("loss", loss_event)
                backward_event = timing.start()
                (generator_loss.total / len(tasks)).backward()
                timing.stop("backward", backward_event)

            metrics["generator/total"] = metrics.get("generator/total", 0.0) + (
                float(generator_loss.total.detach()) / len(tasks)
            )
            for name, value in generator_loss.terms.items():
                metrics[name] = metrics.get(name, 0.0) + float(value.detach()) / len(tasks)

            for name, value in generator_loss.statistics.items():
                metrics[name] = metrics.get(name, 0.0) + float(value.detach()) / len(tasks)

        optimizer_event = timing.start()
        prefix_window.step()
        if bool(training.get("verify_ddp_gradient_sync", False)):
            verify_optimizer_gradients_synchronized(generator_optimizer)
            module = unwrap(model)
            time_conv_parameters = [
                parameter
                for name, child in module.decoder.decoder.named_modules()
                if name.endswith("time_conv")
                for parameter in child.parameters()
            ]
            spatial_parameters = next(
                group["params"]
                for group in generator_optimizer.param_groups
                if group.get("name") == "wan_spatial"
            )
            metrics.update(
                {
                    "gradient_present/encoder": _has_gradient(module.encoder.parameters()),
                    "gradient_present/projector": _has_gradient(module.projector.parameters()),
                    "gradient_present/repa_projection": _has_gradient(module.repa_projection.parameters()),
                    "gradient_present/shared_mask": _has_gradient((module.projector.shared_mask_set,)),
                    "gradient_present/temporal_adapter": _has_gradient(module.decoder.temporal_adapter.parameters()),
                    "gradient_present/pre_decoder": _has_gradient(module.decoder.pre_decoder.parameters()),
                    "gradient_present/wan_conv1": _has_gradient(module.decoder.decoder.conv1.parameters()),
                    "gradient_present/wan_time_conv": _has_gradient(time_conv_parameters),
                    "gradient_present/wan_spatial": _has_gradient(spatial_parameters),
                    "gradient_present/discriminator": _has_gradient(discriminator.parameters()),
                    "system/ddp_gradient_sync_verified": 1,
                }
            )
        metrics.update(
            clip_optimizer_gradients(
                generator_optimizer,
                default_max_norm=float(training["gradient_clip"]),
            )
        )
        discriminator_grad_norm = 0.0
        if gan_active:
            discriminator_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    discriminator.parameters(),
                    training["gradient_clip"],
                    error_if_nonfinite=True,
                )
            )
        metrics["grad_norm/discriminator"] = discriminator_grad_norm
        generator_optimizer.step()
        generator_scheduler.step()
        if gan_active:
            discriminator_optimizer.step()
            discriminator_scheduler.step()
            discriminator_update_count += 1
        timing.stop("optimizer", optimizer_event)
        optimizer_step += 1

        if collect_timing:
            torch.cuda.synchronize(device)
        step_seconds = time.perf_counter() - started

        if optimizer_step % training["log_every"] == 0:
            metrics = reduce_scalar_metrics(metrics, device)
            prefix_metrics, prefix_histograms = prefix_window.reduce()
            metrics.update(prefix_metrics)

        if rank == 0 and optimizer_step % training["log_every"] == 0:
            cuda_timings = timing.resolve()
            model_timings = unwrap(model).runtime_timings()
            task_counts = {
                kind: sum(task.kind == kind for task in tasks)
                for kind in ("full", "single_prefix", "paired_prefix")
            }
            decoder_views = sum(2 if task.kind == "paired_prefix" else 1 for task in tasks)
            record = {
                "step": optimizer_step,
                "stage": stage,
                "objective_mode": objective_mode,
                "objective/prefix_active": int(stage == "stage1b"),
                "objective/repa_active": int(current_repa_factor > 0.0),
                "objective/repa_factor": current_repa_factor,
                "objective/gan_active": int(gan_active),
                "objective/gan_factor": gan_factor,
                "system/step_seconds": step_seconds,
                "system/clips_per_second": len(tasks) * world_size / step_seconds,
                "system/decoder_views": decoder_views,
                "system/decoder_views_per_second": decoder_views * world_size / step_seconds,
                "system/discriminator_forwards": discriminator_forwards,
                "time/data": data_seconds,
                "time/projector": model_timings.get("projector", 0.0),
                "time/wan_forward": model_timings.get("wan_forward", 0.0),
                "time/loss": cuda_timings.get("loss", 0.0),
                "time/backward": cuda_timings.get("backward", 0.0),
                "time/optimizer": cuda_timings.get("optimizer", 0.0),
                "system/max_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                "schedule/phase": active_phase,
                "system/gradient_checkpointing": int(checkpoint_enabled),
                "schedule/full": task_counts["full"],
                "schedule/single_prefix": task_counts["single_prefix"],
                "schedule/paired_prefix": task_counts["paired_prefix"],
                "schedule/task_counts": task_counts,
                "discriminator/update_count": discriminator_update_count,
                "prefix/single_endpoint_histogram": prefix_histograms["single"],
                "prefix/paired_endpoint_histogram": prefix_histograms["paired"],
                "lr": {
                    group.get("name", str(i)): group["lr"]
                    for i, group in enumerate(generator_optimizer.param_groups)
                },
                **metrics,
            }
            print(json.dumps(record, ensure_ascii=False), flush=True)
            append_jsonl_record(log_file, record, rank=rank)
        if optimizer_step % training["log_every"] == 0:
            prefix_window.reset()

        save_at_steps = {int(value) for value in training.get("save_at_steps", [])}
        should_save = (
            optimizer_step in save_at_steps
            or optimizer_step % int(training["save_every"]) == 0
            or optimizer_step == training["max_steps"]
        )
        if should_save:
            if world_size > 1:
                torch.distributed.barrier()
            if rank == 0:
                save_checkpoint(
                    checkpoint_dir / f"step_{optimizer_step:08d}.pt",
                    model=model,
                    discriminator=discriminator,
                    generator_optimizer=generator_optimizer,
                    discriminator_optimizer=discriminator_optimizer,
                    generator_scheduler=generator_scheduler,
                    discriminator_scheduler=discriminator_scheduler,
                    optimizer_step=optimizer_step,
                    discriminator_update_count=discriminator_update_count,
                    epoch=cycling.epoch,
                    config=bundle,
                    stage=stage,
                    objective_mode=objective_mode,
                    log_file=str(log_file),
                    run_id=run_id,
                    checkpoint_dir=str(checkpoint_dir),
                    source_checkpoint=source_checkpoint,
                    static_identity=static_identity,
                    update_latest=True,
                )
            if world_size > 1:
                torch.distributed.barrier()

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
