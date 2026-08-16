from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import json
from pathlib import Path
import time
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from .config import load_training_bundle
from .data import DistributedFullDatasetSampler, VideoManifestDataset, collate_video_samples
from .model.factory import build_model
from .training.checkpoint import load_checkpoint, save_checkpoint, unwrap
from .training.logging import append_jsonl_record, resolve_training_log_file
from .training.losses import FrozenLPIPS, PatchDiscriminator, temporal_l1
from .training.paths import resolve_training_run_paths, validate_resume_log_file
from .training.stages import (
    configure_stage,
    validate_stage_objective,
    validate_training_batch,
    validate_training_bundle,
)
from .train import (
    CyclingLoader,
    _has_gradient,
    _sequence_id,
    _set_nested,
    _wrap_ddp,
    build_optimizers,
    clip_optimizer_gradients,
    cosine_scheduler,
    distributed_context,
    distributed_run_id,
    distributed_verified_representation_identity,
    preflight_pretrained_weights,
    seed_everything,
    validate_init_sampling_contract,
    validate_resume_training_contract,
    verify_optimizer_gradients_synchronized,
    verify_upstream_commit,
)


def _no_sync(model: nn.Module):
    return model.no_sync() if isinstance(model, DistributedDataParallel) else nullcontext()


def _autocast(precision: str):
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def _chunk_loss(
    output,
    lpips: FrozenLPIPS,
    *,
    l1_weight: float,
    lpips_weight: float,
    temporal_weight: float,
    frame_fraction: float,
    transition_fraction: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prediction = output.reconstruction
    target = output.target
    l1 = F.l1_loss(prediction, target)
    perceptual = lpips(prediction, target)
    temporal = temporal_l1(prediction, target)
    total = frame_fraction * (l1_weight * l1 + lpips_weight * perceptual)
    total = total + transition_fraction * temporal_weight * temporal
    return total, {"l1": l1, "lpips": perceptual, "temporal_l1": temporal}


def _boundary_temporal_l1(first, second, first_target, second_target):
    predicted_delta = second[:, :, 0] - first[:, :, -1]
    target_delta = second_target[:, :, 0] - first_target[:, :, -1]
    return F.l1_loss(predicted_delta, target_delta)


def _gradient_ownership(model: nn.Module) -> dict[str, int]:
    module = unwrap(model)
    time_conv = [
        parameter
        for name, child in module.decoder.decoder.named_modules()
        if name.endswith("time_conv")
        for parameter in child.parameters()
    ]
    spatial = [
        parameter
        for name, parameter in module.decoder.decoder.named_parameters()
        if "time_conv" not in name and not name.startswith("conv1.")
    ]
    return {
        "gradient_present/encoder": _has_gradient(module.encoder.parameters()),
        "gradient_present/projector": _has_gradient(module.projector.parameters()),
        "gradient_present/shared_mask": _has_gradient((module.projector.shared_mask_set,)),
        "gradient_present/repa_projection": _has_gradient(module.repa_projection.parameters()),
        "gradient_present/temporal_adapter": _has_gradient(
            module.decoder.temporal_adapter.parameters()
        ),
        "gradient_present/pre_decoder": _has_gradient(
            module.decoder.pre_decoder.parameters()
        ),
        "gradient_present/wan_conv1": _has_gradient(
            module.decoder.decoder.conv1.parameters()
        ),
        "gradient_present/wan_time_conv": _has_gradient(time_conv),
        "gradient_present/wan_spatial": _has_gradient(spatial),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train optional Stage 1-A-plus cross-clip cache adaptation"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--init-from", default=None)
    parser.add_argument("--selection-certificate", default=None)
    parser.add_argument("--allow-smoke-checkpoint", action="store_true")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()
    if args.resume and args.init_from:
        parser.error("--resume and --init-from are mutually exclusive")
    if not args.resume and not args.init_from:
        parser.error("Stage 1-A-plus requires --init-from or --resume")
    if args.selection_certificate and not args.init_from:
        parser.error("--selection-certificate requires --init-from")
    if args.selection_certificate and args.allow_smoke_checkpoint:
        parser.error("--selection-certificate is only valid for formal training")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 1-A-plus requires CUDA")

    bundle = load_training_bundle(args.config)
    for expression in args.set:
        _set_nested(bundle, expression)
    training, model_config, data_config = (
        bundle["training"],
        bundle["model"],
        bundle["data"],
    )
    if training.get("stage") != "stage1a_plus":
        raise ValueError("Plus trainer requires stage=stage1a_plus")
    validate_training_bundle(training, model_config, data_config)
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

    rank, local_rank, world_size = distributed_context()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    seed_everything(args.seed, rank)
    expected_global = (
        int(training["micro_batch_size"])
        * int(training["gradient_accumulation_steps"])
        * world_size
    )
    if expected_global != int(training["global_batch_size"]):
        raise ValueError("global_batch_size does not match the distributed topology")

    dataset = VideoManifestDataset(
        Path(data_config["manifest_dir"]) / "train.parquet",
        split="train",
        num_frames=33,
        target_fps=float(data_config["target_fps"]),
        height=int(data_config["height"]),
        width=int(data_config["width"]),
        horizontal_flip=bool(data_config.get("train_horizontal_flip", True)),
        min_native_fps_ratio=float(data_config.get("min_native_fps_ratio", 0.0)),
    )
    sampler = DistributedFullDatasetSampler(
        dataset,
        seed=args.seed,
        rank=rank,
        world_size=world_size,
        samples_per_rank_multiple=int(training["micro_batch_size"]),
    )
    loader_options: dict[str, Any] = {}
    if int(data_config["num_workers"]) > 0:
        loader_options["prefetch_factor"] = int(data_config.get("prefetch_factor", 2))
    loader = DataLoader(
        dataset,
        batch_size=int(training["micro_batch_size"]),
        sampler=sampler,
        num_workers=int(data_config["num_workers"]),
        pin_memory=True,
        persistent_workers=int(data_config["num_workers"]) > 0,
        collate_fn=collate_video_samples,
        drop_last=True,
        **loader_options,
    )
    cycling = CyclingLoader(loader, sampler)

    run_id = distributed_run_id(rank=rank, resume_checkpoint=args.resume)
    run_paths = resolve_training_run_paths(
        training,
        stage="stage1a_plus",
        run_id=run_id,
        resume_checkpoint=args.resume,
    )
    checkpoint_dir = run_paths.checkpoint_dir
    source_checkpoint = str(Path(args.resume or args.init_from).expanduser().resolve())
    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        torch.distributed.barrier()

    model = build_model(model_config, validate_pretrained=False)
    report = preflight_pretrained_weights(
        model,
        checkpoint_dir / "pretrained_load_report.json" if rank == 0 else None,
    )
    static_identity = distributed_verified_representation_identity(model, rank=rank)
    model = model.to(device)
    configure_stage(model, "stage1a_plus", repa_trainable=False)
    model.decoder.enable_gradient_checkpointing(bool(training["gradient_checkpointing"]))
    discriminator = PatchDiscriminator().to(device).requires_grad_(False)
    if world_size > 1:
        model = _wrap_ddp(model, local_rank)
        discriminator = DistributedDataParallel(
            discriminator, device_ids=[local_rank], output_device=local_rank
        )

    optimizer, discriminator_optimizer = build_optimizers(
        model, discriminator, training
    )
    scheduler = cosine_scheduler(
        optimizer,
        int(training["warmup_steps"]),
        int(training["max_steps"]),
        min_lr_ratio=float(training.get("min_lr_ratio", 0.0)),
    )
    discriminator_scheduler = cosine_scheduler(
        discriminator_optimizer,
        int(training["warmup_steps"]),
        int(training["max_steps"]),
        min_lr_ratio=float(training.get("min_lr_ratio", 0.0)),
    )
    lpips = FrozenLPIPS().to(device)

    optimizer_step = 0
    checkpoint = None
    if args.init_from:
        checkpoint = load_checkpoint(
            args.init_from,
            model=model,
            discriminator=discriminator,
            restore_rng=False,
            static_identity=static_identity,
            target_stage="stage1a_plus",
            target_objective_mode=objective_mode,
            load_mode="init",
            allow_smoke_checkpoint=args.allow_smoke_checkpoint,
            selection_certificate=args.selection_certificate,
        )
        validate_init_sampling_contract(checkpoint, data_config)
    else:
        checkpoint = load_checkpoint(
            args.resume,
            model=model,
            discriminator=discriminator,
            generator_optimizer=optimizer,
            discriminator_optimizer=discriminator_optimizer,
            generator_scheduler=scheduler,
            discriminator_scheduler=discriminator_scheduler,
            static_identity=static_identity,
            target_stage="stage1a_plus",
            target_objective_mode=objective_mode,
            load_mode="resume",
            allow_smoke_checkpoint=args.allow_smoke_checkpoint,
        )
        validate_resume_training_contract(checkpoint, training, data_config)
        optimizer_step = int(checkpoint["optimizer_step"])
        cycling.epoch = int(checkpoint["epoch"])
        cycling.sampler.set_epoch(cycling.epoch)
        cycling.iterator = iter(cycling.loader)

    if args.resume:
        log_file = validate_resume_log_file(
            checkpoint["log_file"], log_root=run_paths.log_root
        )
    else:
        log_file = resolve_training_log_file(
            training, checkpoint_dir, run_id=run_id
        )
    resolved_bundle = copy.deepcopy(bundle)
    resolved_bundle["runtime"] = {
        "run_id": run_id,
        "checkpoint_dir": str(checkpoint_dir),
        "log_file": str(log_file),
        "source_checkpoint": source_checkpoint,
        "run_mode": "smoke" if args.allow_smoke_checkpoint else "formal",
    }
    if rank == 0:
        (checkpoint_dir / "resolved_config.json").write_text(
            json.dumps(resolved_bundle, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    append_jsonl_record(
        log_file,
        {
            "event": "training_start" if not args.resume else "training_resume",
            "step": optimizer_step,
            "stage": "stage1a_plus",
            "objective_mode": objective_mode,
            "data_filter_stats": dataset.filter_stats,
            "pretrained_ready": bool(report["ready"]),
        },
        rank=rank,
    )

    accumulation = int(training["gradient_accumulation_steps"])
    l1_weight = float(training["l1_weight"])
    lpips_weight = float(training["lpips_weight"])
    temporal_weight = float(training["temporal_l1_weight"])

    while optimizer_step < int(training["max_steps"]):
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad(set_to_none=True)
        metrics: dict[str, float] = {}

        for micro_index in range(accumulation):
            batch = cycling.next()
            validate_training_batch(
                batch,
                stage="stage1a_plus",
                expected_frames=33,
                expected_height=int(data_config["height"]),
                expected_width=int(data_config["width"]),
            )
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            first_window = pixels[:, :, :17]
            second_window = pixels[:, :, 16:33]
            sequence_id = _sequence_id(batch)
            first_timestamp = float(batch["sampled_timestamps"][0, 0])
            second_timestamp = float(batch["sampled_timestamps"][0, 16])

            with _no_sync(model), _autocast(str(training["precision"])):
                first = model(
                    first_window,
                    sequence_id=sequence_id,
                    chunk_latent_start=0,
                    chunk_is_sequence_start=True,
                    chunk_target_fps=float(data_config["target_fps"]),
                    chunk_start_timestamp=first_timestamp,
                )
                cache_state = first.decoder_output.cache_state
                if cache_state is None:
                    raise RuntimeError("First Plus chunk returned no cache")
                continuation_cache = unwrap(model).decoder.detached_cache_state(
                    cache_state
                )
                first_loss, first_terms = _chunk_loss(
                    first,
                    lpips,
                    l1_weight=l1_weight,
                    lpips_weight=lpips_weight,
                    temporal_weight=temporal_weight,
                    frame_fraction=17.0 / 33.0,
                    transition_fraction=16.0 / 32.0,
                )
            (first_loss / accumulation).backward()
            first_prediction = first.reconstruction.detach()
            first_target = first.target.detach()
            del first, first_loss

            final_microbatch = micro_index == accumulation - 1
            sync_context = nullcontext() if final_microbatch else _no_sync(model)
            with sync_context, _autocast(str(training["precision"])):
                second = model(
                    second_window,
                    cache_state=continuation_cache,
                    sequence_id=sequence_id,
                    chunk_latent_start=5,
                    chunk_is_sequence_start=False,
                    chunk_target_fps=float(data_config["target_fps"]),
                    chunk_start_timestamp=second_timestamp,
                )
                second_loss, second_terms = _chunk_loss(
                    second,
                    lpips,
                    l1_weight=l1_weight,
                    lpips_weight=lpips_weight,
                    temporal_weight=temporal_weight,
                    frame_fraction=16.0 / 33.0,
                    transition_fraction=15.0 / 32.0,
                )
                boundary = _boundary_temporal_l1(
                    first_prediction,
                    second.reconstruction,
                    first_target,
                    second.target,
                )
                total_second = second_loss + temporal_weight * boundary / 32.0
            (total_second / accumulation).backward()

            values = {
                "loss/first_total": first_terms["l1"] + first_terms["lpips"],
                "loss/second_total": second_terms["l1"] + second_terms["lpips"],
                "loss/first_l1": first_terms["l1"],
                "loss/second_l1": second_terms["l1"],
                "loss/first_lpips": first_terms["lpips"],
                "loss/second_lpips": second_terms["lpips"],
                "loss/first_temporal": first_terms["temporal_l1"],
                "loss/second_temporal": second_terms["temporal_l1"],
                "loss/boundary_temporal": boundary,
            }
            for name, value in values.items():
                if not torch.isfinite(value):
                    raise FloatingPointError(f"Non-finite Plus loss: {name}")
                metrics[name] = metrics.get(name, 0.0) + float(value.detach()) / accumulation

        if bool(training.get("verify_ddp_gradient_sync", False)):
            verify_optimizer_gradients_synchronized(optimizer)
        metrics.update(_gradient_ownership(model))
        metrics.update(
            clip_optimizer_gradients(
                optimizer,
                default_max_norm=float(training["gradient_clip"]),
            )
        )
        optimizer.step()
        scheduler.step()
        optimizer_step += 1
        torch.cuda.synchronize(device)
        metrics.update(
            {
                "event": "train_step",
                "step": optimizer_step,
                "stage": "stage1a_plus",
                "system/step_seconds": time.perf_counter() - started,
                "system/peak_allocated_gb": torch.cuda.max_memory_allocated(device)
                / (1024**3),
                "system/peak_reserved_gb": torch.cuda.max_memory_reserved(device)
                / (1024**3),
                "lr/rae_fast": float(optimizer.param_groups[0]["lr"]),
                "lr/wan_temporal": float(optimizer.param_groups[1]["lr"]),
            }
        )
        if optimizer_step % int(training["log_every"]) == 0 or optimizer_step <= 2:
            append_jsonl_record(log_file, metrics, rank=rank)

        save_steps = {int(value) for value in training.get("save_at_steps", [])}
        should_save = (
            optimizer_step in save_steps
            or optimizer_step % int(training["save_every"]) == 0
            or optimizer_step == int(training["max_steps"])
        )
        if should_save:
            if world_size > 1:
                torch.distributed.barrier()
            if rank == 0:
                save_checkpoint(
                    checkpoint_dir / f"step_{optimizer_step:08d}.pt",
                    model=model,
                    discriminator=discriminator,
                    generator_optimizer=optimizer,
                    discriminator_optimizer=discriminator_optimizer,
                    generator_scheduler=scheduler,
                    discriminator_scheduler=discriminator_scheduler,
                    optimizer_step=optimizer_step,
                    discriminator_update_count=0,
                    epoch=cycling.epoch,
                    config=resolved_bundle,
                    stage="stage1a_plus",
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
