from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import resource
import time
import traceback
from typing import Any

import torch
from torch.utils.data import DataLoader

from progressive_videorae.config import load_yaml
from progressive_videorae.data import (
    BalancedHumanNonSpeechSampler,
    VideoManifestDataset,
    collate_video_samples,
)
from progressive_videorae.data.manifest import merge_source_rows, parse_csv_spec


def distributed_context() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="gloo")
    return rank, local_rank, world_size


def collect_existing_rows(source, limit: int) -> list[dict[str, str]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("Install pandas to build a smoke manifest") from exc

    rows: list[dict[str, str]] = []
    for chunk in pd.read_csv(
        source.path,
        usecols=lambda column: column in {"path", "caption"},
        chunksize=4096,
    ):
        if "path" not in chunk or "caption" not in chunk:
            raise ValueError(f"CSV must contain path and caption columns: {source.path}")
        for row in chunk[["path", "caption"]].itertuples(index=False):
            path = str(row.path).strip()
            if path and path.lower() not in {"nan", "none"} and Path(path).is_file():
                rows.append({"path": path, "caption": "" if row.caption is None else str(row.caption)})
                if len(rows) >= limit:
                    return rows
    return rows


def build_smoke_manifest(csv_spec: Path, output: Path, samples_per_category: int) -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("Install pandas and pyarrow to build a smoke manifest") from exc

    sources = parse_csv_spec(csv_spec)
    source_rows = [
        (source, collect_existing_rows(source, samples_per_category)) for source in sources
    ]
    merged = merge_source_rows(source_rows)
    human = [row for row in merged if row["category"] == "human"]
    non_speech = [row for row in merged if row["category"] == "non_speech"]
    if len(human) < samples_per_category or len(non_speech) < samples_per_category:
        raise RuntimeError(
            "Could not collect enough existing files for the smoke manifest: "
            f"human={len(human)}, non_speech={len(non_speech)}, "
            f"required={samples_per_category}"
        )
    selected = human[:samples_per_category] + non_speech[:samples_per_category]
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selected).to_parquet(output, index=False)
    return {
        "path": str(output),
        "records": len(selected),
        "human": samples_per_category,
        "non_speech": samples_per_category,
        "sources": {source.source_tag: len(rows) for source, rows in source_rows},
    }


def validate_batch(
    batch: dict[str, Any],
    *,
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
) -> tuple[float, float]:
    pixels = batch["pixel_values"]
    expected = (batch_size, 3, num_frames, height, width)
    if tuple(pixels.shape) != expected:
        raise AssertionError(f"pixel_values shape={tuple(pixels.shape)}, expected={expected}")
    if not torch.is_floating_point(pixels):
        raise AssertionError(f"pixel_values dtype must be floating point, got {pixels.dtype}")
    if not torch.isfinite(pixels).all():
        raise AssertionError("pixel_values contains NaN or Inf")
    minimum = float(pixels.min())
    maximum = float(pixels.max())
    if minimum < 0.0 or maximum > 1.0:
        raise AssertionError(f"pixel_values range must be [0,1], got [{minimum}, {maximum}]")

    timestamps = batch["sampled_timestamps"]
    frame_indices = batch["sampled_frame_indices"]
    if tuple(timestamps.shape) != (batch_size, num_frames):
        raise AssertionError(f"sampled_timestamps has invalid shape {tuple(timestamps.shape)}")
    if tuple(frame_indices.shape) != (batch_size, num_frames):
        raise AssertionError(f"sampled_frame_indices has invalid shape {tuple(frame_indices.shape)}")
    if not torch.all(timestamps[:, 1:] > timestamps[:, :-1]):
        raise AssertionError("sampled timestamps are not strictly increasing")
    if not torch.all(frame_indices[:, 1:] >= frame_indices[:, :-1]):
        raise AssertionError("sampled frame indices are not monotonic")
    if any(category not in {"human", "non_speech"} for category in batch["category"]):
        raise AssertionError(f"Unexpected category in batch: {batch['category']}")
    if any(not Path(path).is_file() for path in batch["path"]):
        raise AssertionError("Dataset returned a path that is not a file")
    return minimum, maximum


def run_rank(
    args: argparse.Namespace,
    manifest: Path,
    data_config: dict[str, Any],
    rank: int,
    local_rank: int,
    world_size: int,
) -> dict[str, Any]:
    dataset = VideoManifestDataset(
        manifest,
        split="train",
        num_frames=int(data_config["num_frames"]),
        target_fps=float(data_config["target_fps"]),
        height=int(data_config["height"]),
        width=int(data_config["width"]),
        horizontal_flip=bool(data_config.get("train_horizontal_flip", True)),
        max_decode_retries=args.max_decode_retries,
    )
    sampler = BalancedHumanNonSpeechSampler(
        dataset,
        seed=args.seed,
        rank=rank,
        world_size=world_size,
    )
    sampler.set_epoch(args.epoch)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_video_samples,
        drop_last=True,
    )

    available_batches = len(loader)
    target_batches = min(args.batches, available_batches)
    if target_batches < args.batches:
        raise RuntimeError(
            f"rank {rank} has only {available_batches} complete batches, requested {args.batches}"
        )

    device = None
    if args.check_device_transfer:
        if not torch.cuda.is_available():
            raise RuntimeError("--check-device-transfer requires CUDA")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

    iterator_started = time.perf_counter()
    iterator = iter(loader)
    first_batch_seconds = None
    categories: Counter[str] = Counter()
    sample_ids: list[str] = []
    paths: list[str] = []
    minimum = float("inf")
    maximum = float("-inf")
    pinned_batches = 0
    h2d_seconds = 0.0

    for batch_index in range(target_batches):
        batch = next(iterator)
        if first_batch_seconds is None:
            first_batch_seconds = time.perf_counter() - iterator_started
        batch_min, batch_max = validate_batch(
            batch,
            batch_size=args.batch_size,
            num_frames=int(data_config["num_frames"]),
            height=int(data_config["height"]),
            width=int(data_config["width"]),
        )
        minimum = min(minimum, batch_min)
        maximum = max(maximum, batch_max)
        pinned_batches += int(batch["pixel_values"].is_pinned())
        categories.update(batch["category"])
        sample_ids.extend(batch["sample_id"])
        paths.extend(batch["path"])

        if device is not None:
            transfer_started = time.perf_counter()
            device_pixels = batch["pixel_values"].to(device, non_blocking=True)
            torch.cuda.synchronize(device)
            h2d_seconds += time.perf_counter() - transfer_started
            del device_pixels

    total_seconds = time.perf_counter() - iterator_started
    clips = target_batches * args.batch_size
    return {
        "ok": True,
        "rank": rank,
        "batches": target_batches,
        "clips": clips,
        "categories": dict(categories),
        "sample_ids": sample_ids,
        "paths": paths,
        "first_batch_seconds": first_batch_seconds,
        "total_seconds": total_seconds,
        "clips_per_second": clips / total_seconds,
        "pinned_batches": pinned_batches,
        "pixel_min": minimum,
        "pixel_max": maximum,
        "mean_h2d_ms": 1000.0 * h2d_seconds / target_batches if device is not None else None,
        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }


def summarize(results: list[dict[str, Any]], manifest_info: dict[str, Any] | None) -> bool:
    failures = [result for result in results if not result.get("ok")]
    if failures:
        print(json.dumps({"status": "FAILED", "failures": failures}, indent=2), flush=True)
        return False

    all_ids = [sample_id for result in results for sample_id in result["sample_ids"]]
    duplicate_ids = sorted(
        sample_id for sample_id, count in Counter(all_ids).items() if count > 1
    )
    global_categories = Counter()
    for result in results:
        global_categories.update(result["categories"])

    compact_ranks = [
        {
            key: result[key]
            for key in (
                "rank",
                "batches",
                "clips",
                "categories",
                "first_batch_seconds",
                "total_seconds",
                "clips_per_second",
                "pinned_batches",
                "pixel_min",
                "pixel_max",
                "mean_h2d_ms",
                "max_rss_mb",
            )
        }
        for result in results
    ]
    balanced = global_categories["human"] == global_categories["non_speech"]
    ok = not duplicate_ids and balanced
    report = {
        "status": "PASS" if ok else "FAILED",
        "manifest": manifest_info,
        "world_size": len(results),
        "global_clips": len(all_ids),
        "global_categories": dict(global_categories),
        "unique_sample_ids": len(set(all_ids)),
        "cross_rank_duplicate_ids": duplicate_ids[:20],
        "rank_results": compact_ranks,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test the exact Progressive VideoRAE training data-loading path"
    )
    parser.add_argument("--data-config", default="configs/data/default.yaml")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--csv-spec", default=None)
    parser.add_argument(
        "--smoke-manifest",
        type=Path,
        default=Path("/tmp/progressive_videorae_dataloader_smoke/train.parquet"),
    )
    parser.add_argument("--samples-per-category", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-decode-retries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument(
        "--check-device-transfer", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if args.batch_size <= 0 or args.batches <= 0 or args.num_workers < 0:
        parser.error("batch-size and batches must be positive; num-workers must be non-negative")

    rank, local_rank, world_size = distributed_context()
    data_config = load_yaml(args.data_config)
    configured_manifest = Path(data_config["manifest_dir"]) / "train.parquet"
    manifest = Path(args.manifest) if args.manifest is not None else configured_manifest
    manifest_info: dict[str, Any] | None = None

    if not manifest.is_file():
        manifest = args.smoke_manifest
        csv_spec = Path(args.csv_spec or data_config["csv_spec"])
        if rank == 0:
            manifest_info = build_smoke_manifest(
                csv_spec, manifest, args.samples_per_category
            )
            print(json.dumps({"smoke_manifest": manifest_info}, ensure_ascii=False), flush=True)
        if world_size > 1:
            torch.distributed.barrier()
    elif rank == 0:
        manifest_info = {"path": str(manifest), "source": "existing training manifest"}

    try:
        local_result = run_rank(
            args, manifest, data_config, rank, local_rank, world_size
        )
    except Exception as exc:
        local_result = {
            "ok": False,
            "rank": rank,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }

    if world_size > 1:
        results: list[dict[str, Any] | None] = [None] * world_size
        torch.distributed.all_gather_object(results, local_result)
        gathered = [result for result in results if result is not None]
    else:
        gathered = [local_result]

    success = summarize(gathered, manifest_info) if rank == 0 else True
    if world_size > 1:
        status = [success]
        torch.distributed.broadcast_object_list(status, src=0)
        success = bool(status[0])
        torch.distributed.destroy_process_group()
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
