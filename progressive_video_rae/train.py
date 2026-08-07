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
from .training.checkpoint import load_checkpoint, save_checkpoint, unwrap
from .training.losses import PatchDiscriminator, ProgressiveLosses, scalar_terms
from .training.stages import configure_stage, sample_prefix


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Progressive VideoRAE stages 1A, 1B, or 2A")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--init-from", default=None)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()
    if args.resume and args.init_from:
        parser.error("--resume and --init-from are mutually exclusive")

    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA; run scripts/create_env.sh on the A100 node first")
    bundle = load_training_bundle(args.config)
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
    losses = ProgressiveLosses().to(device)
    generator_optimizer, discriminator_optimizer = build_optimizers(model, discriminator, training)
    generator_scheduler = cosine_scheduler(
        generator_optimizer, training["warmup_steps"], training["max_steps"]
    )
    discriminator_scheduler = cosine_scheduler(
        discriminator_optimizer, training["warmup_steps"], training["max_steps"]
    )

    optimizer_step = 0
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
        cycling.epoch = int(checkpoint["epoch"])
        cycling.sampler.set_epoch(cycling.epoch)
        cycling.iterator = iter(cycling.loader)

    autocast_dtype = torch.bfloat16 if training["precision"] == "bf16" else torch.float16
    accumulation = training["gradient_accumulation_steps"]
    while optimizer_step < training["max_steps"]:
        started = time.perf_counter()
        prefix_len = sample_prefix(
            training["stage"],
            optimizer_step,
            training.get("prefix_min", 1),
            training.get("prefix_max", 63),
        )
        full_state = prefix_len == 64
        generator_optimizer.zero_grad(set_to_none=True)
        discriminator_optimizer.zero_grad(set_to_none=True)
        accumulated_terms: dict[str, float] = {}

        for micro_step in range(accumulation):
            batch = cycling.next()
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            last_micro = micro_step == accumulation - 1
            model_sync = nullcontext() if last_micro or world_size == 1 else model.no_sync()
            disc_sync = (
                nullcontext()
                if last_micro or world_size == 1 or not full_state
                else discriminator.no_sync()
            )
            with model_sync, torch.autocast("cuda", dtype=autocast_dtype):
                output = model(
                    pixel_values,
                    prefix_len=prefix_len,
                    cache_mode="disabled",
                    return_decoder_features=full_state,
                )
                if full_state:
                    with disc_sync:
                        disc_loss = losses.discriminator(
                            discriminator, output.reconstruction, output.target
                        )
                        (disc_loss.total / accumulation).backward()
                    set_requires_grad(discriminator, False)
                    generator_loss = losses.full_generator(output, discriminator)
                    set_requires_grad(discriminator, True)
                    terms = {**generator_loss.terms, **disc_loss.terms}
                else:
                    with torch.autocast("cuda", enabled=False):
                        prefix_target = dct_lowpass_target(output.target.float(), prefix_len)
                    generator_loss = losses.prefix(output, prefix_target)
                    terms = generator_loss.terms
                (generator_loss.total / accumulation).backward()
            for key, value in scalar_terms(terms).items():
                accumulated_terms[key] = accumulated_terms.get(key, 0.0) + value / accumulation

        torch.nn.utils.clip_grad_norm_(
            [parameter for group in generator_optimizer.param_groups for parameter in group["params"]],
            training["gradient_clip"],
        )
        generator_optimizer.step()
        generator_scheduler.step()
        if full_state:
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), training["gradient_clip"])
            discriminator_optimizer.step()
        discriminator_scheduler.step()
        optimizer_step += 1

        if rank == 0 and optimizer_step % training["log_every"] == 0:
            record = {
                "step": optimizer_step,
                "epoch": cycling.epoch,
                "prefix_len": prefix_len,
                "seconds": time.perf_counter() - started,
                "generator_lr": generator_scheduler.get_last_lr(),
                **accumulated_terms,
            }
            print(json.dumps(record, ensure_ascii=False), flush=True)
            with (output_dir / "train.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

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
                    epoch=cycling.epoch,
                    config=bundle,
                )
            if world_size > 1:
                torch.distributed.barrier()

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
