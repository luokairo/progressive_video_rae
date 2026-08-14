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
from .benchmarks import (
    DAVIS_NAME,
    DAVIS_PROTOCOL_ID,
    TOKENBENCH_COUNTS,
    TOKENBENCH_NAME,
    TOKENBENCH_PROTOCOL_ID,
    decode_benchmark_record,
)


RESULT_SCHEMA_VERSION = 2
FVD_FEATURE_DIM = 400
FVD_PROTOCOL = "stylegan_v_i3d_224_center_crop_features_v1"
PSNR_PROTOCOL = "frame_mean_then_sample_mean"
QUALITY_RANGE_PROTOCOL = "clamped_rgb_minus1_plus1_v1"
PERFORMANCE_PROTOCOL = "batch1_synchronized_cuda_diagnostic_v1"
SAMPLING_PROTOCOL = "sha256_seed_nul_sample_id_v1"

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
    "encoder_start_allocated_gb",
    "encoder_peak_allocated_gb",
    "encoder_incremental_peak_gb",
    "projector_start_allocated_gb",
    "projector_peak_allocated_gb",
    "projector_incremental_peak_gb",
    "decoder_start_allocated_gb",
    "decoder_peak_allocated_gb",
    "decoder_incremental_peak_gb",
)
SUMMARY_METRICS = RECONSTRUCTION_METRICS + PERFORMANCE_METRICS

_ALLOWED_EVALUATION_KEYS = {
    "purpose",
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
    "compute_vjepa",
    "i3d_checkpoint",
    "compute_state_statistics",
    "check_cache_equivalence",
    "cache_check_samples",
    "cache_split_patterns",
    "cache_equivalence_atol",
    "benchmark",
}

_ALLOWED_BENCHMARK_KEYS = {
    "name",
    "protocol_id",
    "sampling_mode",
    "manifest_schema_version",
    "expected_samples",
    "source_counts",
    "num_frames",
    "target_fps",
    "height",
    "width",
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


def consume_reserve_replacement(
    candidate_index: int,
    initial_selection_count: int,
    unresolved_initial_failures: list[str],
) -> str | None:
    if candidate_index < initial_selection_count or not unresolved_initial_failures:
        return None
    return unresolved_initial_failures.pop(0)


def require_exact_sample_count(actual: int, expected: int) -> None:
    if actual != expected:
        raise RuntimeError(f"Only {actual}/{expected} samples decoded successfully")


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
    purpose = evaluation.get("purpose", "full_reconstruction")
    if purpose not in {
        "full_reconstruction",
        "checkpoint_selection",
        "checkpoint_selection_smoke",
    }:
        raise ValueError(f"Unsupported evaluation purpose: {purpose}")
    selection_mode = purpose in {
        "checkpoint_selection",
        "checkpoint_selection_smoke",
    }
    benchmark = evaluation.get("benchmark")
    if benchmark is None:
        expected_split = "val" if selection_mode else "test"
        if evaluation["split"] != expected_split:
            raise ValueError(
                f"{purpose} evaluation requires split='{expected_split}'"
            )
    else:
        if not isinstance(benchmark, dict):
            raise TypeError("benchmark must be a mapping")
        unknown_benchmark = sorted(set(benchmark) - _ALLOWED_BENCHMARK_KEYS)
        missing_benchmark = sorted(_ALLOWED_BENCHMARK_KEYS - set(benchmark))
        if unknown_benchmark or missing_benchmark:
            raise ValueError(
                "Invalid benchmark config fields: "
                f"unknown={unknown_benchmark}, missing={missing_benchmark}"
            )
        if evaluation["split"] != "benchmark":
            raise ValueError("Benchmark evaluation requires split='benchmark'")
        if benchmark["sampling_mode"] != "exhaustive":
            raise ValueError("Benchmark evaluation requires exhaustive sampling")
        if int(benchmark["manifest_schema_version"]) != 1:
            raise ValueError("Benchmark evaluation requires manifest schema v1")
        if int(benchmark["expected_samples"]) != int(evaluation["max_clips"]):
            raise ValueError("max_clips must equal benchmark.expected_samples")
        if not isinstance(benchmark["name"], str) or not benchmark["name"].strip():
            raise ValueError("benchmark.name must be non-empty")
        if not isinstance(benchmark["protocol_id"], str) or not benchmark["protocol_id"].strip():
            raise ValueError("benchmark.protocol_id must be non-empty")
        if not isinstance(benchmark["source_counts"], dict):
            raise TypeError("benchmark.source_counts must be a mapping")
        benchmark_geometry = (
            int(benchmark["num_frames"]),
            float(benchmark["target_fps"]),
            int(benchmark["height"]),
            int(benchmark["width"]),
        )
        if benchmark_geometry != (17, 12.0, 480, 768):
            raise ValueError("Benchmark protocol must use 17 frames at 12 FPS and 480x768")
        fixed_protocols = {
            TOKENBENCH_NAME: {
                "protocol_id": TOKENBENCH_PROTOCOL_ID,
                "expected_samples": 500,
                "source_counts": TOKENBENCH_COUNTS,
                "compute_fvd": True,
            },
            DAVIS_NAME: {
                "protocol_id": DAVIS_PROTOCOL_ID,
                "expected_samples": 30,
                "source_counts": {"davis2017_val": 30},
                "compute_fvd": False,
            },
        }
        if benchmark["name"] not in fixed_protocols:
            raise ValueError(f"Unsupported formal benchmark: {benchmark['name']}")
        fixed = fixed_protocols[benchmark["name"]]
        normalized_counts = {
            str(key): int(value) for key, value in benchmark["source_counts"].items()
        }
        if benchmark["protocol_id"] != fixed["protocol_id"]:
            raise ValueError("Benchmark protocol_id is not the fixed PVR protocol")
        if int(benchmark["expected_samples"]) != fixed["expected_samples"]:
            raise ValueError("Benchmark sample count is not the fixed PVR protocol")
        if normalized_counts != fixed["source_counts"]:
            raise ValueError("Benchmark source counts are not the fixed PVR protocol")
        if bool(evaluation["compute_fvd"]) is not fixed["compute_fvd"]:
            raise ValueError("Benchmark compute_fvd does not match the fixed PVR protocol")
    if evaluation["device"] != "cuda":
        raise ValueError("Formal full evaluation requires device='cuda'")
    if int(evaluation["batch_size"]) != 1:
        raise ValueError("Formal full evaluation requires batch_size=1")
    if evaluation["precision"] != "bf16":
        raise ValueError("Formal full evaluation requires precision='bf16'")
    if int(evaluation["max_clips"]) < 2:
        raise ValueError("max_clips must be at least 2")
    if int(evaluation["sampling_seed"]) < 0:
        raise ValueError("sampling_seed must be non-negative")
    if int(evaluation["save_video_limit"]) < 0:
        raise ValueError("save_video_limit must be non-negative")
    if int(evaluation["cache_check_samples"]) < 0:
        raise ValueError("cache_check_samples must be non-negative")
    for key in (
        "save_videos",
        "compute_fvd",
        "compute_state_statistics",
        "check_cache_equivalence",
    ):
        if type(evaluation[key]) is not bool:
            raise TypeError(f"{key} must be a boolean")
    compute_vjepa = evaluation.get("compute_vjepa", True)
    if type(compute_vjepa) is not bool:
        raise TypeError("compute_vjepa must be a boolean")
    if selection_mode:
        if benchmark is not None:
            raise ValueError("Checkpoint selection cannot use a benchmark config")
        required_selection_values = {
            "sampling_seed": 20260807,
            "compute_fvd": False,
            "compute_vjepa": False,
            "compute_state_statistics": False,
            "check_cache_equivalence": False,
            "save_videos": False,
        }
        mismatches = {
            key: (evaluation.get(key), expected)
            for key, expected in required_selection_values.items()
            if evaluation.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"Checkpoint-selection protocol mismatch: {mismatches}")
        if purpose == "checkpoint_selection" and int(evaluation["max_clips"]) != 5000:
            raise ValueError("Formal checkpoint selection requires exactly 5000 clips")
        if purpose == "checkpoint_selection_smoke" and not 2 <= int(
            evaluation["max_clips"]
        ) <= 8:
            raise ValueError("Checkpoint-selection smoke requires 2 to 8 clips")
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
    rgb_frames = int(data_config["num_frames"])
    latent_frames = int(model_config["state"]["num_frames"])
    allowed_geometry = {(17, 5), (33, 9)} if selection_mode else {(17, 5)}
    if (rgb_frames, latent_frames) not in allowed_geometry:
        raise ValueError(
            f"{purpose} does not allow RGB/state geometry "
            f"{rgb_frames} frames/{latent_frames} latents"
        )
    if float(data_config["target_fps"]) != 12.0:
        raise ValueError("The formal full_480p protocol requires target_fps=12")
    if (int(data_config["height"]), int(data_config["width"])) != (480, 768):
        raise ValueError("The formal full_480p protocol requires 480x768 RGB clips")
    expected_state = {
        "num_frames": latent_frames,
        "height": 30,
        "width": 48,
        "channels": 48,
        "num_sets": 48,
        "tokens_per_set": 30,
    }
    actual_state = model_config["state"]
    for key, expected in expected_state.items():
        if benchmark is not None and key not in actual_state:
            raise ValueError(f"Benchmark model config is missing state.{key}")
        if int(actual_state.get(key, expected)) != expected:
            raise ValueError(f"Formal P_47 evaluation requires state.{key}={expected}")
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
    normalized["purpose"] = purpose
    normalized["compute_vjepa"] = compute_vjepa
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
    sample_ids = [str(row["sample_id"]) for row in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Evaluation manifest contains duplicate sample_id values")
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
        benchmark_mode: bool = False,
    ) -> None:
        self.records = records
        self.benchmark_mode = benchmark_mode
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
            if self.benchmark_mode:
                pixel_values, metadata = decode_benchmark_record(
                    row,
                    num_frames=self.config.num_frames,
                    target_fps=self.config.target_fps,
                    height=self.config.height,
                    width=self.config.width,
                )
            else:
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
    start_allocated_gb: float
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
        start_allocated_gb=float(baseline / gib),
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
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            is_last_line = handle.tell() == file_size
            if not line.endswith(b"\n"):
                break
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                if is_last_line:
                    break
                raise ValueError(f"Corrupt JSONL record before file tail: {path}") from exc
            if not isinstance(value, dict):
                if is_last_line:
                    break
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
    "evaluation/",
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
    pretrained_checkpoint_sha256: dict[str, str],
    sampling: dict[str, Any],
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
        "pretrained_checkpoint_sha256": pretrained_checkpoint_sha256,
        "sampling": {
            "protocol": (
                "benchmark_manifest_exhaustive_v1"
                if evaluation.get("benchmark") is not None
                else (
                    "checkpoint_selection_frozen_manifest_v1"
                    if str(evaluation.get("purpose", "")).startswith(
                        "checkpoint_selection"
                    )
                    else SAMPLING_PROTOCOL
                )
            ),
            **sampling,
        },
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
