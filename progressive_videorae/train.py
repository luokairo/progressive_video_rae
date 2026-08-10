from __future__ import annotations

import argparse
from contextlib import nullcontext
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
from .model.dct import dct_lowpass_target
from .model.factory import build_model
from .training.checkpoint import load_checkpoint, load_decoder_from_checkpoint, save_checkpoint, unwrap
from .training.logging import append_jsonl_record, gradient_norm, resolve_training_log_file
from .training.losses import PatchDiscriminator, ProgressiveLosses
from .training.stages import (
    adversarial_factor,
    configure_stage,
    sample_microbatch_tasks,
    stage1a_phase,
    validate_stage_objective,
)


def distributed_context() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def seed_everything(seed: int, rank: int) -> None:
    value = seed + rank
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def cosine_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def scale(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

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
        gradients = [parameter.grad.detach() for parameter in group["params"] if parameter.grad is not None]
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
        if any(not torch.allclose(value, reference, rtol=1e-6, atol=1e-6) for value in gathered[1:]):
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


def build_optimizers(model, discriminator, config: dict[str, Any]):
    module = unwrap(model)
    fast = _unique(
        list(module.projector.parameters())
        + list(module.repa_projection.parameters())
        + list(module.decoder.temporal_adapter.parameters())
    )
    interface = list(module.decoder.pre_decoder.parameters()) + list(
        module.decoder.decoder.conv1.parameters()
    )
    for name, child in module.decoder.decoder.named_modules():
        if name.endswith("time_conv"):
            interface.extend(child.parameters())
    interface = _unique(interface)
    used = {id(parameter) for parameter in fast + interface}
    spatial = [parameter for parameter in module.decoder.parameters() if id(parameter) not in used]
    groups = [
        {"params": fast, "lr": float(config.get("projector_lr", 1e-4)), "name": "rae_fast"},
        {"params": interface, "lr": float(config.get("wan_temporal_lr", 2e-5)), "name": "wan_temporal"},
        {"params": spatial, "lr": float(config.get("wan_spatial_lr", 5e-6)), "name": "wan_spatial"},
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
        lowpass = dct_lowpass_target(full_target.float(), endpoint)
        return losses.prefix(
            output,
            lowpass,
            endpoint=endpoint,
            full_target=full_target,
            previous_prediction=previous_prediction,
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
        "return_decoder_features": task.is_full,
        "sequence_id": sequence_id,
    }
    return model(pixel_values, **kwargs)


def validate_resume_training_contract(
    checkpoint: dict[str, Any],
    training: dict[str, Any],
) -> None:
    saved_bundle = checkpoint.get("config")
    saved = saved_bundle.get("training") if isinstance(saved_bundle, dict) else None
    if not isinstance(saved, dict):
        raise RuntimeError("Resume checkpoint is missing its resolved training config")
    keys = ("micro_batch_size", "gradient_accumulation_steps", "global_batch_size")
    if training.get("stage") == "stage1a":
        keys += (
            "wan_interface_step",
            "wan_full_step",
            "gradient_checkpointing_by_phase",
            "fused_optimizer",
        )
    if training.get("stage") == "stage1b":
        keys += (
            "prefix_schedule",
            "full_microbatches_per_step",
        )
    mismatches = {
        key: (saved.get(key), training.get(key))
        for key in keys
        if saved.get(key) != training.get(key)
    }
    if mismatches:
        raise RuntimeError(f"Resume training contract mismatch: {mismatches}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ProgressiveVideoRAE Stage 1A/1B/2-A/2-B")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--init-from", default=None)
    parser.add_argument("--init-decoder-from", default=None)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()
    if sum(value is not None for value in (args.resume, args.init_from, args.init_decoder_from)) > 1:
        parser.error("--resume, --init-from and --init-decoder-from are mutually exclusive")
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA")

    bundle = load_training_bundle(args.config, model_config_path=args.model_config)
    for expression in args.set:
        _set_nested(bundle, expression)
    training, model_config, data_config = bundle["training"], bundle["model"], bundle["data"]
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
    rank, local_rank, world_size = distributed_context()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    seed_everything(args.seed, rank)

    expected_global = training["micro_batch_size"] * training["gradient_accumulation_steps"] * world_size
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

    output_dir = Path(training["output_dir"])
    existing_checkpoints = tuple(output_dir.glob("step_*.pt"))
    if not args.resume and existing_checkpoints:
        raise FileExistsError(
            f"Refusing to overwrite existing checkpoints in {output_dir}; use --resume"
        )
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "resolved_config.json").write_text(
            json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    model = build_model(model_config, validate_pretrained=False)
    report = preflight_pretrained_weights(
        model, output_dir / "pretrained_load_report.json" if rank == 0 else None
    )
    if rank == 0:
        print(json.dumps({"pretrained": report}, ensure_ascii=False), flush=True)
    model = model.to(device)
    active_phase = training_phase(training, 0)
    configure_stage(
        model,
        stage,
        optimizer_step=0,
        wan_interface_step=int(training.get("wan_interface_step", 2000)),
        wan_full_step=int(training.get("wan_full_step", 5000)),
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
        adversarial_weight=training.get("adversarial_weight", 0.1),
        temporal_l1_weight=training.get("temporal_l1_weight", 0.1),
        band_weight=training.get("band_weight", 1.0),
        leakage_weight=training.get("leakage_weight", 0.1),
        paired_delta_weight=training.get("paired_delta_weight", 1.0),
    ).to(device)
    generator_optimizer, discriminator_optimizer = build_optimizers(model, discriminator, training)
    generator_scheduler = cosine_scheduler(
        generator_optimizer, training["warmup_steps"], training["max_steps"]
    )
    discriminator_scheduler = cosine_scheduler(
        discriminator_optimizer, training["warmup_steps"], training["max_steps"]
    )

    optimizer_step = 0
    discriminator_update_count = 0
    if args.init_decoder_from:
        load_decoder_from_checkpoint(args.init_decoder_from, model=model)
    if args.init_from:
        load_checkpoint(args.init_from, model=model, discriminator=discriminator, restore_rng=False)
    checkpoint = None
    if args.resume:
        checkpoint = load_checkpoint(
            args.resume,
            model=model,
            discriminator=discriminator,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            generator_scheduler=generator_scheduler,
            discriminator_scheduler=discriminator_scheduler,
        )
        if checkpoint.get("stage") != stage or checkpoint.get("objective_mode") != objective_mode:
            raise RuntimeError("Resume checkpoint stage/objective mismatch")
        validate_resume_training_contract(checkpoint, training)
        optimizer_step = int(checkpoint["optimizer_step"])
        discriminator_update_count = int(checkpoint.get("discriminator_update_count", 0))
        cycling.epoch = int(checkpoint["epoch"])
        cycling.sampler.set_epoch(cycling.epoch)
        cycling.iterator = iter(cycling.loader)

    log_file = resolve_training_log_file(
        training,
        output_dir,
        resume_log_file=(checkpoint or {}).get("log_file") if args.resume else None,
    )
    append_jsonl_record(
        log_file,
        {
            "event": "training_start" if not args.resume else "training_resume",
            "step": optimizer_step,
            "stage": stage,
            "objective_mode": objective_mode,
        },
        rank=rank,
    )

    autocast_dtype = torch.bfloat16 if training["precision"] == "bf16" else torch.float16
    accumulation = int(training["gradient_accumulation_steps"])
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

        data_seconds = 0.0

        for micro_index, task in enumerate(tasks):
            data_started = time.perf_counter()
            batch = cycling.next()
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
                    if gan_factor > 0.0 and losses.adversarial_weight > 0.0:
                        disc_loss = losses.discriminator(
                            discriminator, result.reconstruction, result.target
                        )
                        assert_finite_distributed(disc_loss.total, "discriminator")
                        (disc_loss.total / max(1, full_count)).backward()
                        for name, value in disc_loss.terms.items():
                            metrics[name] = metrics.get(name, 0.0) + (
                                float(value.detach()) / max(1, full_count)
                            )
                    for parameter in discriminator.parameters():
                        parameter.requires_grad_(False)
                    generator_loss = losses.full_generator(
                        result, discriminator, adversarial_factor=gan_factor
                    )
                    for parameter in discriminator.parameters():
                        parameter.requires_grad_(True)
                elif task.kind == "single_prefix":
                    assert not isinstance(result, tuple)
                    generator_loss = _prefix_loss(losses, result, pixel_values, task.endpoint)
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
                    generator_loss.total = 0.5 * (previous_loss.total + current_loss.total)
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

        optimizer_event = timing.start()
        generator_parameters = [
            parameter
            for group in generator_optimizer.param_groups
            for parameter in group["params"]
        ]
        if bool(training.get("verify_ddp_gradient_sync", False)):
            verify_optimizer_gradients_synchronized(generator_optimizer)
        for group_index, group in enumerate(generator_optimizer.param_groups):
            value = gradient_norm(group["params"])
            if value is not None:
                name = group.get("name", str(group_index))
                metrics[f"grad_norm/{name}"] = float(value)
        generator_grad_norm = torch.nn.utils.clip_grad_norm_(
            generator_parameters,
            training["gradient_clip"],
            error_if_nonfinite=True,
        )
        discriminator_grad_norm = torch.nn.utils.clip_grad_norm_(
            discriminator.parameters(),
            training["gradient_clip"],
            error_if_nonfinite=True,
        )
        metrics["grad_norm/generator"] = float(generator_grad_norm)
        metrics["grad_norm/discriminator"] = float(discriminator_grad_norm)
        generator_optimizer.step()
        generator_scheduler.step()
        if full_count and gan_factor > 0:
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
                "objective/repa_active": 1,
                "system/step_seconds": step_seconds,
                "system/clips_per_second": len(tasks) * world_size / step_seconds,
                "system/decoder_views_per_second": decoder_views * world_size / step_seconds,
                "time/data": data_seconds,
                "time/projector": model_timings.get("projector", 0.0),
                "time/wan_forward": model_timings.get("wan_forward", 0.0),
                "time/loss": cuda_timings.get("loss", 0.0),
                "time/backward": cuda_timings.get("backward", 0.0),
                "time/optimizer": cuda_timings.get("optimizer", 0.0),
                "system/max_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                "schedule/phase": active_phase,
                "system/gradient_checkpointing": int(checkpoint_enabled),
                "schedule/task_counts": task_counts,
                "lr": {group.get("name", str(i)): group["lr"] for i, group in enumerate(generator_optimizer.param_groups)},
                **metrics,
            }
            print(json.dumps(record, ensure_ascii=False), flush=True)
            append_jsonl_record(log_file, record, rank=rank)

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
                    output_dir / f"step_{optimizer_step:08d}.pt",
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
                    update_latest=True,
                )
            if world_size > 1:
                torch.distributed.barrier()

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
