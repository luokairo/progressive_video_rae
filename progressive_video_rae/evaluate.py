from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import load_yaml, resolve_config_path
from .data import VideoManifestDataset, collate_video_samples
from .evaluation.io import save_comparison_video, save_prefix_curve
from .evaluation.metrics import (
    I3DFeatureExtractor,
    MetricSuite,
    encoder_cosine,
    frechet_distance,
    psnr,
    state_set_statistics,
)
from .model.dct import dct_lowpass_target
from .model.factory import build_model
from .training.checkpoint import load_checkpoint


def average_numeric(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = sorted(
        {key for row in rows for key, value in row.items() if isinstance(value, (int, float))}
    )
    return {
        key: float(np.mean([row[key] for row in rows if key in row]))
        for key in keys
        if key != "prefix_len"
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate full and prefix Progressive VideoRAE states")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    evaluation = load_yaml(args.config)
    model_config = load_yaml(resolve_config_path(evaluation["model_config"]))
    data_config = load_yaml(resolve_config_path(evaluation["data_config"]))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_config).to(device).eval()
    load_checkpoint(args.checkpoint, model=model, restore_rng=False)
    model.decoder.enable_gradient_checkpointing(False)

    manifest = Path(data_config["manifest_dir"]) / f"{evaluation['split']}.parquet"
    dataset_split = "test" if str(evaluation["split"]).startswith("test") else "val"
    dataset = VideoManifestDataset(
        manifest,
        split=dataset_split,
        num_frames=data_config["num_frames"],
        target_fps=data_config["target_fps"],
        height=data_config["height"],
        width=data_config["width"],
        horizontal_flip=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=evaluation["batch_size"],
        shuffle=False,
        num_workers=data_config["num_workers"],
        pin_memory=device.type == "cuda",
        persistent_workers=data_config["num_workers"] > 0,
        collate_fn=collate_video_samples,
    )
    if evaluation["batch_size"] != 1:
        raise ValueError("Per-sample CSV reporting currently requires eval batch_size=1")
    metric_suite = MetricSuite().to(device).eval()
    i3d = None
    if evaluation.get("compute_fvd", False):
        i3d = I3DFeatureExtractor(evaluation["i3d_checkpoint"]).to(device).eval()

    rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    real_fvd: list[np.ndarray] = []
    fake_fvd: list[np.ndarray] = []
    saved_videos = 0
    cache_error = None
    prefixes = [int(value) for value in evaluation["prefixes"]]
    precision = torch.bfloat16 if evaluation["precision"] == "bf16" else torch.float16

    for clip_index, batch in enumerate(loader):
        if clip_index >= evaluation["max_clips"]:
            break
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        target_rgb = pixel_values.mul(2).sub(1)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        encoder_started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(
            device.type, dtype=precision, enabled=device.type == "cuda"
        ):
            encoder_output = model.encoder(pixel_values)
        if device.type == "cuda":
            torch.cuda.synchronize()
        encoder_seconds = time.perf_counter() - encoder_started

        for prefix_len in prefixes:
            with torch.inference_mode(), torch.autocast(
                device.type, dtype=precision, enabled=device.type == "cuda"
            ):
                state = model.projector(encoder_output, prefix_len=prefix_len)
                decoder_started = time.perf_counter()
                decoder_output = model.decoder.decode(
                    state, prefix_len=prefix_len, cache_mode="disabled"
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                decoder_seconds = time.perf_counter() - decoder_started
            with torch.autocast(device.type, enabled=False):
                dct_target = dct_lowpass_target(target_rgb.float(), prefix_len)
            prediction = decoder_output.video.float()
            prefix_metrics = metric_suite.reconstruction(
                prediction, dct_target.float(), prefix_len=prefix_len
            )
            row = {
                "sample_id": batch["sample_id"][0],
                "path": batch["path"][0],
                "category": batch["category"][0],
                "source_tags": "+".join(batch["source_tags"][0]),
                "prefix_len": prefix_len,
                "psnr_rgb": float(psnr(prediction, target_rgb.float()).mean().cpu()),
                "encoder_seconds": encoder_seconds,
                "decoder_seconds": decoder_seconds,
                "peak_memory_gb": (
                    torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
                ),
                **prefix_metrics,
            }
            rows.append(row)
            if prefix_len == 64:
                with torch.inference_mode(), torch.autocast(
                    device.type, dtype=precision, enabled=device.type == "cuda"
                ):
                    reconstructed_features = model.encoder(
                        prediction.add(1).mul(0.5).clamp(0, 1)
                    )
                local, global_score = encoder_cosine(encoder_output, reconstructed_features)
                row["vjepa_local_cosine"] = float(local.cpu())
                row["vjepa_global_cosine"] = float(global_score.cpu())
                for stats in state_set_statistics(state):
                    state_rows.append({"sample_id": batch["sample_id"][0], **stats})
                if i3d is not None:
                    real_fvd.append(i3d(target_rgb).cpu().numpy())
                    fake_fvd.append(i3d(prediction).cpu().numpy())
                if evaluation.get("check_cache_equivalence", False) and cache_error is None:
                    with torch.autocast(
                        device.type, dtype=precision, enabled=device.type == "cuda"
                    ):
                        cached = model.decoder.decode(state, prefix_len=64, cache_mode="reset")
                    cache_error = float((cached.video.float() - prediction).abs().max().cpu())

            if (
                evaluation.get("save_videos", False)
                and saved_videos < evaluation.get("save_video_limit", 8)
                and prefix_len in (1, 8, 32, 64)
            ):
                save_comparison_video(
                    output_dir / "videos" / f"{batch['sample_id'][0]}_p{prefix_len:02d}.mp4",
                    [target_rgb[0], dct_target[0], decoder_output.video[0]],
                    fps=data_config["target_fps"],
                )
                if prefix_len == 64:
                    saved_videos += 1

    grouped = {
        str(prefix): average_numeric([row for row in rows if row["prefix_len"] == prefix])
        for prefix in prefixes
    }
    category_groups = {
        category: average_numeric([row for row in rows if row["category"] == category])
        for category in sorted({row["category"] for row in rows})
    }
    source_tag_groups = {
        source_tags: average_numeric([row for row in rows if row["source_tags"] == source_tags])
        for source_tags in sorted({row["source_tags"] for row in rows})
    }
    monotonic_values = []
    for sample_id in {row["sample_id"] for row in rows}:
        sample_rows = sorted(
            (row for row in rows if row["sample_id"] == sample_id),
            key=lambda row: row["prefix_len"],
        )
        comparisons = [
            later["psnr_rgb"] >= earlier["psnr_rgb"]
            for earlier, later in zip(sample_rows, sample_rows[1:])
        ]
        if comparisons:
            monotonic_values.append(sum(comparisons) / len(comparisons))
    summary: dict[str, Any] = {
        "num_clips": len({row["sample_id"] for row in rows}),
        "prefix_metrics": grouped,
        "category_metrics": category_groups,
        "source_tag_metrics": source_tag_groups,
        "mean_monotonic_improvement_rate": (
            float(np.mean(monotonic_values)) if monotonic_values else None
        ),
        "cache_max_abs_error": cache_error,
    }
    if len(real_fvd) >= 2 and len(fake_fvd) >= 2:
        summary["rfvd"] = frechet_distance(np.concatenate(real_fvd), np.concatenate(fake_fvd))
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(output_dir / "metrics.csv", rows)
    write_csv(output_dir / "state_statistics.csv", state_rows)
    save_prefix_curve(rows, output_dir / "quality_prefix_curve.png", metric="psnr_rgb")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
