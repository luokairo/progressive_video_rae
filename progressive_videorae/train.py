from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from .config import load_training_bundle
from .data import BalancedHumanNonSpeechSampler, VideoManifestDataset, collate_video_samples
from .model.dct import dct_lowpass_target
from .model.factory import build_model
from .training.checkpoint import (
    load_checkpoint,
    load_decoder_from_checkpoint,
    save_checkpoint,
    unwrap,
)
from .training.logging import (
    GlobalMetricWindow,
    append_jsonl_record,
    gradient_norm,
    resolve_training_log_file,
)
from .training.losses import PatchDiscriminator, ProgressiveLosses
from .training.stages import (
    adversarial_factor,
    configure_stage,
    normalize_objective_weights,
    objective_loss_scales,
    sample_microbatch_prefixes,
)


FULL_LOSS_TERMS = (
    "l1",
    "lpips",
    "temporal_l1",
    "repa_local",
    "repa_global",
    "adversarial",
)
PREFIX_LOSS_TERMS = ("l1", "lpips", "temporal_l1")


def training_metric_names() -> tuple[str, ...]:
    names = [
        *(f"full/{term}" for term in FULL_LOSS_TERMS),
        *(f"prefix/{term}" for term in PREFIX_LOSS_TERMS),
        *(f"generator/full/weighted_{term}" for term in FULL_LOSS_TERMS),
        *(f"generator/prefix/weighted_{term}" for term in PREFIX_LOSS_TERMS),
        "generator/full/weighted_total",
        "generator/prefix/weighted_total",
        "generator/total",
        "disc/real_loss",
        "disc/fake_loss",
        "disc/total",
        "grad/generator_total_preclip",
        "grad/projector_preclip",
        "grad/decoder_preclip",
        "grad/repa_projection_preclip",
        "grad/discriminator_preclip",
        "system/step_seconds",
        "gan/ramp_factor",
        "gan/effective_weight",
    ]
    return tuple(names)


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


def cosine_scheduler(
    optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    def scale(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def validate_training_schedule(config: dict[str, Any]) -> int:
    """Validate optimizer-step schedules and return the planned D update count."""

    normalize_objective_weights(
        config.get("full_objective_weight", 1.0),
        config.get("prefix_objective_weight", 1.0),
    )
    temporal_weight = float(config.get("temporal_l1_weight", 0.0))
    if not math.isfinite(temporal_weight) or temporal_weight < 0.0:
        raise ValueError("temporal_l1_weight must be finite and non-negative")

    max_steps = int(config["max_steps"])
    disc_start = int(config.get("disc_start", 0))
    ramp_steps = int(config.get("adversarial_ramp_steps", 0))
    interval = int(config.get("discriminator_update_interval", 1))
    adversarial_factor(0, disc_start, ramp_steps)
    if disc_start > max_steps:
        raise ValueError("disc_start cannot exceed max_steps")
    if interval <= 0:
        raise ValueError("discriminator_update_interval must be positive")

    schedule = config.get("prefix_schedule") or (
        "full" if config["stage"] in ("stage1a", "stage2a") else "alternating"
    )
    accumulation = int(config["gradient_accumulation_steps"])
    if schedule == "mixed_accumulation":
        full_count = int(config.get("full_microbatches_per_step", accumulation // 2))
        if accumulation % 2 != 0 or full_count * 2 != accumulation:
            raise ValueError(
                "mixed_accumulation requires an even accumulation count and a 50/50 task split"
            )
    elif interval != 1:
        raise ValueError(
            "discriminator_update_interval > 1 is only supported by mixed_accumulation"
        )

    active_steps = max_steps - disc_start
    if schedule in ("full", "mixed_accumulation"):
        if active_steps % interval != 0:
            raise ValueError("GAN-active steps must end on a discriminator update boundary")
        if interval > 1:
            save_every = int(config["save_every"])
            if disc_start % interval != 0 or save_every % interval != 0:
                raise ValueError(
                    "disc_start and save_every must align with discriminator update boundaries"
                )
        return active_steps // interval

    if schedule == "alternating":
        return sum(step % 2 == 0 for step in range(disc_start, max_steps))
    if schedule == "random":
        probability = float(config.get("full_prefix_probability", 0.5))
        if not 0.0 <= probability <= 1.0:
            raise ValueError("full_prefix_probability must be in [0, 1]")
        return max(1, math.ceil(active_steps * probability))
    raise ValueError(
        "prefix_schedule must be full, alternating, random, or mixed_accumulation"
    )


def discriminator_cycle_flags(
    optimizer_step: int,
    start_step: int,
    update_interval: int,
    *,
    has_full_microbatch: bool,
) -> tuple[bool, bool, bool]:
    """Return whether D is active, starts a cycle, and steps after this G update."""

    if update_interval <= 0:
        raise ValueError("discriminator_update_interval must be positive")
    active = optimizer_step >= start_step and has_full_microbatch
    if not active:
        return False, False, False
    offset = optimizer_step - start_step
    return True, offset % update_interval == 0, (offset + 1) % update_interval == 0


def discriminator_accumulation_boundary(
    completed_generator_steps: int, start_step: int, update_interval: int
) -> bool:
    if update_interval <= 0:
        raise ValueError("discriminator_update_interval must be positive")
    return (
        completed_generator_steps <= start_step
        or (completed_generator_steps - start_step) % update_interval == 0
    )


def completed_discriminator_updates(completed_steps: int, config: dict[str, Any]) -> int:
    """Derive completed D updates for deterministic schedules and legacy checkpoints."""

    disc_start = int(config.get("disc_start", 0))
    if completed_steps <= disc_start:
        return 0
    interval = int(config.get("discriminator_update_interval", 1))
    schedule = config.get("prefix_schedule") or (
        "full" if config["stage"] in ("stage1a", "stage2a") else "alternating"
    )
    if schedule in ("full", "mixed_accumulation"):
        return (completed_steps - disc_start) // interval
    if schedule == "alternating":
        return sum(step % 2 == 0 for step in range(disc_start, completed_steps))
    return 0


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


class CyclingLoader:
    def __init__(self, loader: DataLoader, sampler: BalancedHumanNonSpeechSampler, epoch: int = 0):
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


def build_optimizers(model, discriminator, config: dict[str, Any]):
    module = unwrap(model)
    groups = []
    projector_parameters = [parameter for parameter in module.projector.parameters() if parameter.requires_grad]
    decoder_parameters = [
        parameter
        for submodule in (module.decoder, module.repa_projection)
        for parameter in submodule.parameters()
        if parameter.requires_grad
    ]
    if projector_parameters:
        groups.append({"params": projector_parameters, "lr": config.get("projector_lr", 1e-4)})
    if decoder_parameters:
        groups.append({"params": decoder_parameters, "lr": config["decoder_lr"]})
    generator_optimizer = torch.optim.AdamW(
        groups, betas=(0.9, 0.95), weight_decay=config["weight_decay"]
    )
    discriminator_optimizer = torch.optim.AdamW(
        discriminator.parameters(),
        lr=config["discriminator_lr"],
        betas=(0.5, 0.9),
        weight_decay=0.0,
    )
    return generator_optimizer, discriminator_optimizer


def preflight_pretrained_weights(model: nn.Module, report_path: Path | None = None) -> dict:
    """Block training before optimizer creation unless all required pretrained weights loaded."""

    module = unwrap(model)
    module.assert_pretrained_ready()
    report = module.pretrained_load_report()
    if not report["ready"]:
        raise RuntimeError("Pretrained report is not ready")
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return report


def forward_training_batch(
    model: nn.Module,
    pixel_values: torch.Tensor,
    *,
    prefix_len: int,
    full_state: bool,
):
    """Run one training forward pass with the sampled progressive prefix."""

    return model(
        pixel_values,
        prefix_len=prefix_len,
        cache_mode="disabled",
        return_decoder_features=full_state,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Progressive VideoRAE stages 1A, 1B, or 2A")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--init-from", default=None)
    parser.add_argument("--init-decoder-from", default=None)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()
    initializers = [args.resume, args.init_from, args.init_decoder_from]
    if sum(value is not None for value in initializers) > 1:
        parser.error(
            "--resume, --init-from, and --init-decoder-from are mutually exclusive"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA; run scripts/create_env.sh on the A100 node first")
    bundle = load_training_bundle(args.config, model_config_path=args.model_config)
    training = bundle["training"]
    model_config = bundle["model"]
    data_config = bundle["data"]
    rank, local_rank, world_size = distributed_context()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    seed_everything(args.seed, rank)

    expected_global = (
        training["micro_batch_size"] * training["gradient_accumulation_steps"] * world_size
    )
    if expected_global != training["global_batch_size"]:
        raise ValueError(
            f"Configured global_batch_size={training['global_batch_size']} but topology gives {expected_global}"
        )
    discriminator_total_updates = validate_training_schedule(training)

    train_manifest = Path(data_config["manifest_dir"]) / "train.parquet"
    dataset = VideoManifestDataset(
        train_manifest,
        split="train",
        num_frames=data_config["num_frames"],
        target_fps=data_config["target_fps"],
        height=data_config["height"],
        width=data_config["width"],
        horizontal_flip=data_config.get("train_horizontal_flip", True),
    )
    sampler = BalancedHumanNonSpeechSampler(
        dataset, seed=args.seed, rank=rank, world_size=world_size
    )
    loader = DataLoader(
        dataset,
        batch_size=training["micro_batch_size"],
        sampler=sampler,
        num_workers=data_config["num_workers"],
        pin_memory=True,
        persistent_workers=data_config["num_workers"] > 0,
        collate_fn=collate_video_samples,
        drop_last=True,
    )
    cycling = CyclingLoader(loader, sampler)

    output_dir = Path(training["output_dir"])
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "resolved_config.json").write_text(
            json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    model = build_model(model_config, validate_pretrained=False)
    report_path = output_dir / "pretrained_load_report.json" if rank == 0 else None
    pretrained_report = preflight_pretrained_weights(model, report_path)
    if rank == 0:
        print(json.dumps({"pretrained": pretrained_report}, ensure_ascii=False), flush=True)
    model = model.to(device)
    configure_stage(model, training["stage"])
    model.decoder.enable_gradient_checkpointing(True)
    discriminator: nn.Module = PatchDiscriminator().to(device)
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
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
        temporal_l1_weight=training.get("temporal_l1_weight", 0.0),
    ).to(device)
    generator_optimizer, discriminator_optimizer = build_optimizers(model, discriminator, training)
    generator_scheduler = cosine_scheduler(
        generator_optimizer, training["warmup_steps"], training["max_steps"]
    )
    discriminator_interval = int(training.get("discriminator_update_interval", 1))
    discriminator_warmup_updates = math.ceil(
        training["warmup_steps"] / discriminator_interval
    )
    discriminator_scheduler = cosine_scheduler(
        discriminator_optimizer, discriminator_warmup_updates, discriminator_total_updates
    )

    optimizer_step = 0
    discriminator_update_count = 0
    if args.init_decoder_from:
        load_decoder_from_checkpoint(args.init_decoder_from, model=model)
    if args.init_from:
        load_checkpoint(
            args.init_from,
            model=model,
            discriminator=discriminator,
            restore_rng=False,
        )
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
        optimizer_step = int(checkpoint["optimizer_step"])
        discriminator_update_count = int(
            checkpoint.get(
                "discriminator_update_count",
                completed_discriminator_updates(optimizer_step, training),
            )
        )
        if not discriminator_accumulation_boundary(
            optimizer_step,
            int(training.get("disc_start", 0)),
            discriminator_interval,
        ):
            raise ValueError(
                "Checkpoint was saved inside a discriminator accumulation cycle"
            )
        cycling.epoch = int(checkpoint["epoch"])
        cycling.sampler.set_epoch(cycling.epoch)
        cycling.iterator = iter(cycling.loader)

    log_file: Path | None = None
    if rank == 0:
        log_file = resolve_training_log_file(
            training,
            output_dir,
            resume_log_file=checkpoint.get("log_file") if args.resume else None,
        )
        log_file.parent.mkdir(parents=True, exist_ok=True)
        print(json.dumps({"log_file": str(log_file)}, ensure_ascii=False), flush=True)

    autocast_dtype = torch.bfloat16 if training["precision"] == "bf16" else torch.float16
    accumulation = training["gradient_accumulation_steps"]
    disc_start = int(training.get("disc_start", 0))
    adversarial_ramp_steps = int(training.get("adversarial_ramp_steps", 0))
    full_objective_weight = float(training.get("full_objective_weight", 1.0))
    prefix_objective_weight = float(training.get("prefix_objective_weight", 1.0))
    metric_window = GlobalMetricWindow(training_metric_names(), device=device)
    while optimizer_step < training["max_steps"]:
        started = time.perf_counter()
        prefix_lengths = sample_microbatch_prefixes(
            training["stage"],
            optimizer_step,
            accumulation,
            training.get("prefix_min", 1),
            training.get("prefix_max", 63),
            schedule=training.get("prefix_schedule"),
            full_probability=training.get("full_prefix_probability", 0.5),
            full_microbatches_per_step=training.get("full_microbatches_per_step"),
        )
        generator_loss_scales, effective_objective_weights = objective_loss_scales(
            prefix_lengths,
            full_objective_weight,
            prefix_objective_weight,
        )
        full_microbatch_count = sum(prefix == 64 for prefix in prefix_lengths)
        last_full_micro = max(
            (index for index, prefix in enumerate(prefix_lengths) if prefix == 64),
            default=-1,
        )
        gan_factor = adversarial_factor(
            optimizer_step, disc_start, adversarial_ramp_steps
        )
        metric_window.add_mean("gan/ramp_factor", gan_factor)
        metric_window.add_mean("gan/effective_weight", losses.adversarial_weight * gan_factor)
        (
            discriminator_active,
            discriminator_cycle_start,
            discriminator_step_due,
        ) = discriminator_cycle_flags(
            optimizer_step,
            disc_start,
            discriminator_interval,
            has_full_microbatch=full_microbatch_count > 0,
        )

        generator_optimizer.zero_grad(set_to_none=True)
        if discriminator_cycle_start:
            discriminator_optimizer.zero_grad(set_to_none=True)
        weighted_step_terms: dict[str, torch.Tensor] = {}

        for micro_step, prefix_len in enumerate(prefix_lengths):
            full_state = prefix_len == 64
            batch = cycling.next()
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            sample_count = int(pixel_values.shape[0])
            metric_window.add_prefix(prefix_len, count=sample_count)
            last_micro = micro_step == accumulation - 1
            model_sync = nullcontext() if last_micro or world_size == 1 else model.no_sync()
            disc_sync = (
                nullcontext()
                if (
                    world_size == 1
                    or not discriminator_active
                    or not full_state
                    or micro_step == last_full_micro
                )
                else discriminator.no_sync()
            )
            disc_loss = None
            with model_sync, torch.autocast("cuda", dtype=autocast_dtype):
                output = forward_training_batch(
                    model,
                    pixel_values,
                    prefix_len=prefix_len,
                    full_state=full_state,
                )
                if full_state:
                    if discriminator_active:
                        with disc_sync:
                            disc_loss = losses.discriminator(
                                discriminator, output.reconstruction, output.target
                            )
                            disc_scale = (
                                training.get("discriminator_weight", 1.0)
                                / (full_microbatch_count * discriminator_interval)
                            )
                            (disc_scale * disc_loss.total).backward()
                    set_requires_grad(discriminator, False)
                    generator_loss = losses.full_generator(
                        output,
                        discriminator,
                        adversarial_factor=gan_factor,
                    )
                    set_requires_grad(discriminator, True)
                    task_name = "full"
                else:
                    with torch.autocast("cuda", enabled=False):
                        prefix_target = dct_lowpass_target(output.target.float(), prefix_len)
                    generator_loss = losses.prefix(output, prefix_target)
                    task_name = "prefix"
                (generator_loss_scales[micro_step] * generator_loss.total).backward()

            for key, value in generator_loss.terms.items():
                metric_window.add_mean(f"{task_name}/{key}", value, count=sample_count)
            scale = generator_loss_scales[micro_step]
            for key, value in generator_loss.weighted_terms.items():
                metric_key = f"generator/{task_name}/weighted_{key}"
                contribution = value.detach() * scale
                prior = weighted_step_terms.get(metric_key)
                weighted_step_terms[metric_key] = (
                    contribution if prior is None else prior + contribution
                )
            total_key = f"generator/{task_name}/weighted_total"
            total_contribution = generator_loss.total.detach() * scale
            prior_total = weighted_step_terms.get(total_key)
            weighted_step_terms[total_key] = (
                total_contribution if prior_total is None else prior_total + total_contribution
            )
            prior_generator_total = weighted_step_terms.get("generator/total")
            weighted_step_terms["generator/total"] = (
                total_contribution
                if prior_generator_total is None
                else prior_generator_total + total_contribution
            )
            if disc_loss is not None:
                for key, value in disc_loss.terms.items():
                    metric_window.add_mean(
                        f"disc/{key.removeprefix('disc_')}", value, count=sample_count
                    )
                metric_window.add_logits(
                    disc_loss.statistics["real_logits"],
                    disc_loss.statistics["fake_logits"],
                )

        for key, value in weighted_step_terms.items():
            metric_window.add_mean(key, value)

        generator_parameters = [
            parameter
            for group in generator_optimizer.param_groups
            for parameter in group["params"]
        ]
        generator_module = unwrap(model)
        for name, module in (
            ("projector", generator_module.projector),
            ("decoder", generator_module.decoder),
            ("repa_projection", generator_module.repa_projection),
        ):
            module_norm = gradient_norm(module.parameters())
            if module_norm is not None:
                metric_window.add_mean(f"grad/{name}_preclip", module_norm)
        generator_norm = torch.nn.utils.clip_grad_norm_(
            generator_parameters, training["gradient_clip"]
        )
        metric_window.add_mean("grad/generator_total_preclip", generator_norm)
        generator_optimizer.step()
        generator_scheduler.step()
        if discriminator_step_due:
            discriminator_norm = torch.nn.utils.clip_grad_norm_(
                discriminator.parameters(), training["gradient_clip"]
            )
            metric_window.add_mean("grad/discriminator_preclip", discriminator_norm)
            discriminator_optimizer.step()
            discriminator_scheduler.step()
            discriminator_update_count += 1
        optimizer_step += 1
        metric_window.add_mean("system/step_seconds", time.perf_counter() - started)
        metric_window.step()

        if optimizer_step % training["log_every"] == 0:
            global_metrics, prefix_histogram = metric_window.reduce()
            window_steps = metric_window.steps
            metric_window.reset()
            if rank == 0:
                assert log_file is not None
                full_samples = prefix_histogram.get("64", 0)
                prefix_samples = sum(prefix_histogram.values()) - full_samples
                record = {
                    "step": optimizer_step,
                    "epoch": cycling.epoch,
                    "world_size": world_size,
                    "window/optimizer_steps": window_steps,
                    "generator_lr": generator_scheduler.get_last_lr(),
                    "discriminator_lr": discriminator_scheduler.get_last_lr(),
                    "generator/lr": generator_scheduler.get_last_lr(),
                    "disc/lr": discriminator_scheduler.get_last_lr(),
                    "weights/full_objective_config": full_objective_weight,
                    "weights/prefix_objective_config": prefix_objective_weight,
                    "weights/full_effective": effective_objective_weights[0],
                    "weights/prefix_effective": effective_objective_weights[1],
                    "weights/l1": losses.l1_weight,
                    "objective/full_config_weight": full_objective_weight,
                    "objective/prefix_config_weight": prefix_objective_weight,
                    "objective/full_effective_weight": effective_objective_weights[0],
                    "objective/prefix_effective_weight": effective_objective_weights[1],
                    "weights/lpips": losses.lpips_weight,
                    "weights/prefix_lpips": losses.prefix_lpips_weight,
                    "weights/temporal_l1": losses.temporal_l1_weight,
                    "weights/repa_local": losses.repa_local_weight,
                    "weights/repa_global": losses.repa_global_weight,
                    "gan/base_weight": losses.adversarial_weight,
                    "gan/ramp_factor": gan_factor,
                    "gan/adaptive_enabled": False,
                    "gan/adaptive_weight": 1.0,
                    "gan/effective_weight": losses.adversarial_weight * gan_factor,
                    "disc/update_count": discriminator_update_count,
                    "disc/weight": float(training.get("discriminator_weight", 1.0)),
                    "grad/clip_threshold": training["gradient_clip"],
                    "prefix/histogram": prefix_histogram,
                    "prefix/full_samples": full_samples,
                    "prefix/prefix_samples": prefix_samples,
                    **global_metrics,
                }
                print(json.dumps(record, ensure_ascii=False), flush=True)
                append_jsonl_record(log_file, record, rank=rank)

        if optimizer_step % training["save_every"] == 0 or optimizer_step == training["max_steps"]:
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
                    log_file=str(log_file) if log_file is not None else None,
                )
            if world_size > 1:
                torch.distributed.barrier()

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
