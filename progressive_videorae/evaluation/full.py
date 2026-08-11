from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..data.dataset import VideoSamplingConfig, decode_contiguous_clip
from ..model.types import ProgressiveState


RESULT_SCHEMA_VERSION = 2
FVD_FEATURE_DIM = 400
FVD_PROTOCOL = "stylegan_v_i3d_224_center_crop_features_v1"
PSNR_PROTOCOL = "frame_mean_then_sample_mean"
QUALITY_RANGE_PROTOCOL = "clamped_rgb_minus1_plus1_v1"
PERFORMANCE_PROTOCOL = "batch1_synchronized_cuda_diagnostic_v1"

RECONSTRUCTION_METRICS = (
    "rgb_mse",
    "rgb_psnr",
    "rgb_ssim",
    "rgb_lpips",
    "temporal_difference_l1",
    "vjepa_local_cosine",
    "vjepa_global_cosine",
    "out_of_range_fraction",
    "mean_overshoot",
    "max_overshoot",
)
PERFORMANCE_METRICS = (
    "encoder_seconds",
    "projector_seconds",
    "decoder_seconds",
    "model_forward_seconds",
    "encoder_peak_allocated_gb",
    "encoder_incremental_peak_gb",
    "projector_peak_allocated_gb",
    "projector_incremental_peak_gb",
    "decoder_peak_allocated_gb",
    "decoder_incremental_peak_gb",
)
SUMMARY_METRICS = RECONSTRUCTION_METRICS + PERFORMANCE_METRICS

_ALLOWED_EVALUATION_KEYS = {
    "model_config",
    "data_config",
    "split",
    "device",
    "batch_size",
    "precision",
    "max_clips",
    "sampling_seed",
    "save_videos",
    "save_video_limit",
    "compute_fvd",
    "i3d_checkpoint",
    "compute_state_statistics",
    "check_cache_equivalence",
    "cache_check_samples",
    "cache_split_patterns",
    "cache_equivalence_atol",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_sample_key(sample_id: str, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}\0{sample_id}".encode("utf-8")).digest()


def sample_id_digest(sample_ids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for sample_id in sample_ids:
        digest.update(str(sample_id).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def validate_evaluation_config(
    evaluation: dict[str, Any],
    model_config: dict[str, Any],
    data_config: dict[str, Any],
    *,
    require_runtime_device: bool,
    check_paths: bool = True,
) -> dict[str, Any]:
    unknown = sorted(set(evaluation) - _ALLOWED_EVALUATION_KEYS)
    if unknown:
        raise ValueError(f"Unknown full-evaluation config fields: {unknown}")
    required = {
        "model_config",
        "data_config",
        "split",
        "device",
        "batch_size",
        "precision",
        "max_clips",
        "sampling_seed",
        "save_videos",
        "save_video_limit",
        "compute_fvd",
        "compute_state_statistics",
        "check_cache_equivalence",
        "cache_check_samples",
        "cache_split_patterns",
    }
    missing = sorted(required - set(evaluation))
    if missing:
        raise ValueError(f"Missing full-evaluation config fields: {missing}")
    if evaluation["split"] != "test":
        raise ValueError("Formal full evaluation requires split='test'")
    if evaluation["device"] != "cuda":
        raise ValueError("Formal full evaluation requires device='cuda'")
    if int(evaluation["batch_size"]) != 1:
        raise ValueError("Formal full evaluation requires batch_size=1")
    if evaluation["precision"] not in {"bf16", "fp16"}:
        raise ValueError("precision must be exactly 'bf16' or 'fp16'")
    if int(evaluation["max_clips"]) < 2:
        raise ValueError("max_clips must be at least 2")
    if int(evaluation["sampling_seed"]) < 0:
        raise ValueError("sampling_seed must be non-negative")
    if int(evaluation["save_video_limit"]) < 0:
        raise ValueError("save_video_limit must be non-negative")
    if int(evaluation["cache_check_samples"]) < 0:
        raise ValueError("cache_check_samples must be non-negative")
    if not isinstance(evaluation["cache_split_patterns"], list):
        raise TypeError("cache_split_patterns must be a list of integer lists")
    expected_latents = int(model_config["state"]["num_frames"])
    patterns: list[list[int]] = []
    for value in evaluation["cache_split_patterns"]:
        if not isinstance(value, list) or len(value) < 2:
            raise ValueError("Every cache split pattern must contain at least two chunks")
        pattern = [int(size) for size in value]
        if any(size <= 0 for size in pattern) or sum(pattern) != expected_latents:
            raise ValueError(
                f"Cache split {pattern} must contain positive chunks summing to {expected_latents}"
            )
        patterns.append(pattern)
    if evaluation["check_cache_equivalence"] and not patterns:
        raise ValueError("Cache equivalence requires at least one split pattern")
    atol = float(evaluation.get("cache_equivalence_atol", 1e-5))
    if not np.isfinite(atol) or atol <= 0:
        raise ValueError("cache_equivalence_atol must be finite and positive")
    if int(data_config["num_frames"]) != 17:
        raise ValueError("The formal full_480p protocol requires exactly 17 RGB frames")
    if float(data_config["target_fps"]) != 12.0:
        raise ValueError("The formal full_480p protocol requires target_fps=12")
    if (int(data_config["height"]), int(data_config["width"])) != (480, 768):
        raise ValueError("The formal full_480p protocol requires 480x768 RGB clips")
    if bool(evaluation["compute_fvd"]):
        checkpoint = evaluation.get("i3d_checkpoint")
        if not isinstance(checkpoint, str) or not checkpoint.strip():
            raise ValueError("compute_fvd=true requires i3d_checkpoint")
        if check_paths and not Path(checkpoint).expanduser().is_file():
            raise FileNotFoundError(f"I3D checkpoint is unavailable: {checkpoint}")
    if require_runtime_device:
        if not torch.cuda.is_available():
            raise RuntimeError("Formal full evaluation requires an available CUDA device")
        if evaluation["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("The selected CUDA device does not support bf16")
    normalized = dict(evaluation)
    normalized["batch_size"] = 1
    normalized["max_clips"] = int(evaluation["max_clips"])
    normalized["sampling_seed"] = int(evaluation["sampling_seed"])
    normalized["save_video_limit"] = int(evaluation["save_video_limit"])
    normalized["cache_check_samples"] = int(evaluation["cache_check_samples"])
    normalized["cache_split_patterns"] = patterns
    normalized["cache_equivalence_atol"] = atol
    return normalized


def load_ranked_records(
    manifest_path: str | Path,
    *,
    num_frames: int,
    target_fps: float,
    sampling_seed: int,
) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("Install pandas and pyarrow to read evaluation manifests") from exc
    frame = pd.read_parquet(manifest_path).reset_index(drop=True)
    minimum_duration = (num_frames - 1) / target_fps
    if "path_exists" in frame:
        frame = frame[frame["path_exists"].fillna(False)]
    if "decode_valid" in frame:
        frame = frame[frame["decode_valid"].fillna(False)]
    if "duration" in frame:
        duration = frame["duration"]
        frame = frame[duration.isna() | (duration >= minimum_duration - 1e-6)]
    records = frame.to_dict("records")
    records.sort(key=lambda row: stable_sample_key(str(row["sample_id"]), sampling_seed))
    if not records:
        raise ValueError("No usable records remain after evaluation manifest filtering")
    return records


class ExactEvaluationDataset(Dataset):
    """Decode exactly the ranked record; never substitute a different sample."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        num_frames: int,
        target_fps: float,
        height: int,
        width: int,
    ) -> None:
        self.records = records
        self.config = VideoSamplingConfig(
            num_frames=num_frames,
            target_fps=target_fps,
            height=height,
            width=width,
            split="test",
            horizontal_flip=False,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        sample_id = str(row["sample_id"])
        path = str(row["path"])
        base = {
            "candidate_index": int(index),
            "sample_id": sample_id,
            "path": path,
            "category": str(row.get("category", "")),
            "source_tags": normalize_source_tags(row.get("source_tags", [])),
        }
        try:
            pixel_values, metadata = decode_contiguous_clip(path, self.config)
        except Exception as exc:
            return {
                **base,
                "decode_ok": False,
                "decode_error": f"{type(exc).__name__}: {exc}",
            }
        sampled_indices = metadata["sampled_frame_indices"]
        sampled_timestamps = metadata["sampled_timestamps"]
        identity_payload = (
            f"{sample_id}|indices="
            + ",".join(str(int(value)) for value in sampled_indices.tolist())
            + "|timestamps="
            + ",".join(f"{float(value):.9f}" for value in sampled_timestamps.tolist())
        )
        return {
            **base,
            "decode_ok": True,
            "pixel_values": pixel_values,
            "codec_sequence_id": identity_payload,
            **metadata,
        }


def normalize_source_tags(value: Any) -> list[str]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def collate_exact_evaluation(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if len(samples) != 1:
        raise ValueError("Exact evaluation collate requires batch_size=1")
    sample = samples[0]
    if sample["decode_ok"]:
        sample = dict(sample)
        sample["pixel_values"] = sample["pixel_values"].unsqueeze(0)
    return sample


def slice_progressive_state(state: ProgressiveState, start: int, end: int) -> ProgressiveState:
    if not 0 <= start < end <= state.tokens.shape[1]:
        raise ValueError(f"Invalid state slice [{start}:{end}]")
    latent_types = None if state.latent_types is None else state.latent_types[start:end]
    return ProgressiveState(
        tokens=state.tokens[:, start:end],
        layout_version=state.layout_version,
        layout_checksum=state.layout_checksum,
        latent_types=latent_types,
        contract=state.contract,
    )


@dataclass(frozen=True)
class PhaseMeasurement:
    seconds: float
    peak_allocated_gb: float
    incremental_peak_gb: float


def measure_cuda_phase(function):
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    started = torch.cuda.Event(enable_timing=True)
    finished = torch.cuda.Event(enable_timing=True)
    started.record()
    value = function()
    finished.record()
    torch.cuda.synchronize()
    seconds = float(started.elapsed_time(finished) / 1000.0)
    peak = torch.cuda.max_memory_allocated()
    gib = float(1024**3)
    return value, PhaseMeasurement(
        seconds=seconds,
        peak_allocated_gb=float(peak / gib),
        incremental_peak_gb=float(max(0, peak - baseline) / gib),
    )


def summary_statistics(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty metric")
    if not np.isfinite(array).all():
        raise FloatingPointError("Cannot summarize non-finite metric values")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "median": float(np.median(array)),
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def summarize_rows(
    rows: list[dict[str, Any]], metric_names: Iterable[str] = SUMMARY_METRICS
) -> dict[str, dict[str, float | int]]:
    result = {}
    for name in metric_names:
        values = [float(row[name]) for row in rows if name in row]
        if values:
            result[name] = summary_statistics(values)
    return result


def atomic_write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_jsonl(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl_repair(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    valid_end = 0
    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                break
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Corrupt JSONL record before file tail: {path}") from exc
            if not isinstance(value, dict):
                raise TypeError(f"JSONL records must be objects: {path}")
            records.append(value)
            valid_end = handle.tell()
    if path.stat().st_size != valid_end:
        with path.open("r+b") as handle:
            handle.truncate(valid_end)
            handle.flush()
            os.fsync(handle.fileno())
    return records


_PROVENANCE_SOURCE_ROOTS = (
    "configs/",
    "docs/",
    "progressive_videorae/",
    "scripts/",
    "tests/",
)


def git_identity(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root)

    def run(*arguments: str) -> bytes:
        return subprocess.check_output(
            ["git", *arguments], cwd=root, stderr=subprocess.DEVNULL
        ).strip()

    try:
        commit = run("rev-parse", "HEAD").decode("ascii")
        tracked_diff = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD"], cwd=root, stderr=subprocess.DEVNULL
        )
        fingerprint = hashlib.sha256(tracked_diff)
        untracked = run("ls-files", "--others", "--exclude-standard").decode("utf-8")
        relevant_untracked = []
        for relative in sorted(filter(None, untracked.splitlines())):
            if not relative.startswith(_PROVENANCE_SOURCE_ROOTS):
                continue
            relevant_untracked.append(relative)
            path = root / relative
            fingerprint.update(relative.encode("utf-8"))
            if path.is_file():
                with path.open("rb") as handle:
                    while chunk := handle.read(1024 * 1024):
                        fingerprint.update(chunk)
        return {
            "commit": commit,
            "dirty": bool(tracked_diff or relevant_untracked),
            "working_tree_fingerprint": fingerprint.hexdigest(),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "working_tree_fingerprint": None}


def environment_identity() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def run_identity(
    *,
    evaluation: dict[str, Any],
    model_config: dict[str, Any],
    data_config: dict[str, Any],
    checkpoint_path: str,
    checkpoint_sha256: str,
    manifest_path: str,
    manifest_sha256: str,
    i3d_sha256: str | None,
    project_root: str | Path,
) -> dict[str, Any]:
    return {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "evaluation_config": evaluation,
        "model_config": model_config,
        "data_config": data_config,
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "manifest_path": str(Path(manifest_path).resolve()),
        "manifest_sha256": manifest_sha256,
        "i3d": (
            {
                "path": str(Path(evaluation["i3d_checkpoint"]).resolve()),
                "sha256": i3d_sha256,
                "protocol": FVD_PROTOCOL,
                "feature_dim": FVD_FEATURE_DIM,
            }
            if evaluation["compute_fvd"]
            else None
        ),
        "git": git_identity(project_root),
        "environment": environment_identity(),
        "protocols": {
            "psnr": PSNR_PROTOCOL,
            "quality_range": QUALITY_RANGE_PROTOCOL,
            "performance": PERFORMANCE_PROTOCOL,
        },
    }


def prepare_run_manifest(
    output_dir: str | Path,
    identity: dict[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    output = Path(output_dir)
    manifest_path = output / "run_manifest.json"
    if resume:
        if not manifest_path.is_file():
            raise FileNotFoundError("--resume requires an existing run_manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "completed":
            raise RuntimeError("Completed evaluations cannot be resumed")
        if manifest.get("run_identity") != identity:
            raise RuntimeError("Resume provenance does not match the existing evaluation")
        manifest["status"] = "running"
        manifest["resumed_at"] = utc_now()
        manifest.pop("error", None)
        atomic_write_json(manifest_path, manifest)
        return manifest
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Evaluation output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "running",
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "run_identity": identity,
        "representation_identity": None,
        "completed_samples": 0,
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def update_run_manifest(output_dir: str | Path, manifest: dict[str, Any], **updates: Any) -> None:
    manifest.update(updates)
    manifest["updated_at"] = utc_now()
    atomic_write_json(Path(output_dir) / "run_manifest.json", manifest)
