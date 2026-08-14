from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
from typing import Any, Iterable

import numpy as np

from ..checksums import sha256_file
from ..config import load_yaml, project_root, resolve_config_path
from ..data.dataset import VideoSamplingConfig, decode_contiguous_clip
from ..model.types import StateContract
from ..training.checkpoint import CHECKPOINT_SCHEMA_VERSION, OBJECTIVE_BY_STAGE
from ..training.stages import STAGES
from .full import (
    atomic_write_json,
    sample_id_digest,
    stable_sample_key,
    utc_now,
    validate_evaluation_config,
)


SELECTION_SCHEMA_VERSION = 1
SELECTION_PROTOCOL = "pvr_validation_5000_rgb_lpips_v1"
SELECTION_SMOKE_PROTOCOL = "pvr_validation_smoke_rgb_lpips_v1"
SELECTION_COUNT = 5000
SELECTION_SEED = 20260807
_CHECKPOINT_PATTERN = re.compile(r"^step_(\d{8})\.pt$")


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def stable_validation_candidates(
    records: Iterable[dict[str, Any]], *, seed: int = SELECTION_SEED
) -> list[dict[str, Any]]:
    values = [dict(row) for row in records]
    sample_ids = [str(row["sample_id"]) for row in values]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Validation manifest contains duplicate sample IDs")
    values.sort(key=lambda row: stable_sample_key(str(row["sample_id"]), seed))
    return values


def derive_expected_checkpoint_steps(training: dict[str, Any]) -> list[int]:
    maximum = int(training["max_steps"])
    save_every = int(training["save_every"])
    if maximum <= 0 or save_every <= 0:
        raise ValueError("max_steps and save_every must be positive")
    steps = set(range(save_every, maximum + 1, save_every))
    steps.update(int(value) for value in training.get("save_at_steps", ()))
    steps.add(maximum)
    if any(step <= 0 or step > maximum for step in steps):
        raise ValueError("Checkpoint save steps must be in (0, max_steps]")
    return sorted(steps)


def discover_checkpoints(checkpoint_dir: str | Path) -> dict[int, Path]:
    directory = Path(checkpoint_dir).expanduser().resolve()
    discovered: dict[int, Path] = {}
    for path in directory.iterdir():
        match = _CHECKPOINT_PATTERN.fullmatch(path.name)
        if match is None or not path.is_file() or path.is_symlink():
            continue
        step = int(match.group(1))
        if step in discovered:
            raise RuntimeError(f"Duplicate checkpoint step {step}")
        discovered[step] = path.resolve()
    return dict(sorted(discovered.items()))


def validate_resolved_stage_config(
    resolved: dict[str, Any], *, stage: str, checkpoint_dir: Path
) -> tuple[dict[str, Any], list[int]]:
    if stage not in STAGES:
        raise ValueError(f"Unsupported stage: {stage}")
    if set(resolved) != {"training", "model", "data", "runtime"}:
        raise ValueError("resolved_config.json must contain training/model/data/runtime")
    training = resolved["training"]
    runtime = resolved["runtime"]
    if training.get("stage") != stage:
        raise ValueError("Requested stage does not match resolved training config")
    if training.get("objective_mode") != OBJECTIVE_BY_STAGE[stage]:
        raise ValueError("Resolved stage objective is invalid")
    recorded = Path(runtime["checkpoint_dir"]).expanduser().resolve()
    if recorded != checkpoint_dir:
        raise ValueError("resolved_config checkpoint_dir does not match the selected directory")
    if not isinstance(runtime.get("run_id"), str) or not runtime["run_id"]:
        raise ValueError("Resolved config is missing runtime.run_id")
    return training, derive_expected_checkpoint_steps(training)


def validate_selection_evaluation_config(
    path: str | Path, *, expected_geometry: tuple[int, int]
) -> dict[str, Any]:
    evaluation_path = resolve_config_path(path)
    evaluation = load_yaml(evaluation_path)
    model_path = resolve_config_path(
        evaluation["model_config"], relative_to=evaluation_path.parent
    )
    data_path = resolve_config_path(
        evaluation["data_config"], relative_to=evaluation_path.parent
    )
    model = load_yaml(model_path)
    data = load_yaml(data_path)
    normalized = dict(evaluation)
    normalized["model_config"] = str(model_path)
    normalized["data_config"] = str(data_path)
    normalized = validate_evaluation_config(
        normalized,
        model,
        data,
        require_runtime_device=False,
        check_paths=False,
    )
    geometry = (int(data["num_frames"]), int(model["state"]["num_frames"]))
    if geometry != expected_geometry:
        raise ValueError(
            f"Selection config {evaluation_path} has geometry {geometry}, "
            f"expected {expected_geometry}"
        )
    return normalized


def validate_selection_candidate_metadata(
    checkpoint: dict[str, Any],
    *,
    resolved_config: dict[str, Any],
    stage: str,
    expected_step: int,
) -> dict[str, Any]:
    if int(checkpoint.get("checkpoint_schema_version", 0)) != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("Checkpoint selection requires schema v4")
    if checkpoint.get("stage") != stage:
        raise RuntimeError("Checkpoint stage does not match selector stage")
    if checkpoint.get("objective_mode") != OBJECTIVE_BY_STAGE[stage]:
        raise RuntimeError("Checkpoint objective does not match selector stage")
    if checkpoint.get("optimizer_step") != expected_step:
        raise RuntimeError("Checkpoint filename step does not match optimizer_step")
    maximum = int(resolved_config["training"]["max_steps"])
    if checkpoint.get("stage_max_steps") != maximum:
        raise RuntimeError("Checkpoint stage_max_steps does not match resolved config")
    complete = checkpoint.get("stage_complete")
    if type(complete) is not bool or complete != (expected_step >= maximum):
        raise RuntimeError("Checkpoint stage_complete is inconsistent")
    if checkpoint.get("state_contract") != StateContract().to_dict():
        raise RuntimeError("Checkpoint StateContract is missing or incompatible")
    if checkpoint.get("run_id") != resolved_config["runtime"]["run_id"]:
        raise RuntimeError("Checkpoint run_id does not match resolved config")
    recorded_dir = Path(str(checkpoint.get("checkpoint_dir", ""))).expanduser().resolve()
    expected_dir = Path(resolved_config["runtime"]["checkpoint_dir"]).expanduser().resolve()
    if recorded_dir != expected_dir:
        raise RuntimeError("Checkpoint directory identity mismatch")
    if checkpoint.get("config") != resolved_config:
        raise RuntimeError("Checkpoint resolved config does not match resolved_config.json")
    if not isinstance(checkpoint.get("representation_identity"), dict):
        raise RuntimeError("Checkpoint representation identity is missing")
    return {
        "stage": stage,
        "objective": OBJECTIVE_BY_STAGE[stage],
        "optimizer_step": expected_step,
        "stage_complete": complete,
        "run_id": checkpoint["run_id"],
    }


def load_selection_candidate_metadata(
    path: str | Path,
    *,
    resolved_config: dict[str, Any],
    stage: str,
    expected_step: int,
) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(
        str(Path(path).expanduser().resolve()),
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError("Training checkpoint must contain a dictionary")
    return validate_selection_candidate_metadata(
        checkpoint,
        resolved_config=resolved_config,
        stage=stage,
        expected_step=expected_step,
    )


def _selection_rows_digest(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        payload = {
            key: row[key]
            for key in (
                "selection_index",
                "sample_id",
                "path",
                "file_size",
                "file_mtime_ns",
                "center_timestamp",
                "replacement_for",
            )
        }
        digest.update(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _atomic_write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("Install pandas and pyarrow to write selection manifests") from exc
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    pd.DataFrame(rows).to_parquet(temporary, index=False)
    os.replace(temporary, path)


def prepare_selection_manifest(
    *,
    validation_manifest: str | Path,
    output_dir: str | Path,
    count: int = SELECTION_COUNT,
    seed: int = SELECTION_SEED,
    protocol: str = SELECTION_PROTOCOL,
) -> tuple[Path, dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("Install pandas and pyarrow to prepare selection manifests") from exc
    source = Path(validation_manifest).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    manifest_path = output / "selection_manifest.parquet"
    report_path = output / "selection_manifest.json"
    if manifest_path.exists() or report_path.exists():
        raise FileExistsError("Selection manifest already exists; use --resume")
    frame = pd.read_parquet(source).reset_index(drop=True)
    minimum_duration = 32 / 12.0
    frame = frame[
        frame["path_exists"].fillna(False)
        & frame["decode_valid"].fillna(False)
        & (frame["duration"].isna() | (frame["duration"] >= minimum_duration - 1e-6))
    ]
    records = stable_validation_candidates(frame.to_dict("records"), seed=seed)
    decode_config = VideoSamplingConfig(
        num_frames=33,
        target_fps=12.0,
        height=480,
        width=768,
        split="val",
        horizontal_flip=False,
    )
    selected: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    unresolved_failures: list[str] = []
    failures_path = output / "selection_failures.jsonl"
    output.mkdir(parents=True, exist_ok=True)
    for candidate_index, row in enumerate(records):
        if len(selected) >= count:
            break
        path = Path(str(row["path"])).expanduser().resolve()
        try:
            _, metadata = decode_contiguous_clip(str(path), decode_config)
        except Exception as exc:
            failure = {
                "candidate_index": candidate_index,
                "sample_id": str(row["sample_id"]),
                "path": str(path),
                "error": f"{type(exc).__name__}: {exc}",
            }
            with failures_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            failures.append(failure)
            unresolved_failures.append(str(row["sample_id"]))
            continue
        stat = path.stat()
        timestamps = metadata["sampled_timestamps"].tolist()
        selected.append(
            {
                **row,
                "path": str(path),
                "selection_index": len(selected),
                "candidate_index": candidate_index,
                "file_size": stat.st_size,
                "file_mtime_ns": stat.st_mtime_ns,
                "center_timestamp": float(timestamps[len(timestamps) // 2]),
                "replacement_for": (
                    unresolved_failures.pop(0) if unresolved_failures else None
                ),
            }
        )
    if len(selected) != count:
        raise RuntimeError(f"Only {len(selected)}/{count} validation samples decoded")
    digest = sample_id_digest(str(row["sample_id"]) for row in selected)
    rows_digest = _selection_rows_digest(selected)
    source_sha = sha256_file(source)
    report = {
        "selection_schema_version": SELECTION_SCHEMA_VERSION,
        "protocol": protocol,
        "created_at": utc_now(),
        "source_manifest": str(source),
        "source_manifest_sha256": source_sha,
        "selection_count": count,
        "selection_seed": seed,
        "minimum_geometry": {"rgb_frames": 33, "fps": 12.0},
        "sample_id_digest": digest,
        "selection_rows_digest": rows_digest,
        "decode_failures": len(failures),
    }
    _atomic_write_parquet(manifest_path, selected)
    report["selection_manifest_sha256"] = sha256_file(manifest_path)
    atomic_write_json(report_path, report)
    return manifest_path, report


def load_selection_manifest(
    manifest_path: str | Path,
    *,
    expected_count: int = SELECTION_COUNT,
    expected_protocol: str = SELECTION_PROTOCOL,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("Install pandas and pyarrow to read selection manifests") from exc
    path = Path(manifest_path).expanduser().resolve()
    report_path = path.with_suffix(".json")
    if not report_path.is_file():
        raise FileNotFoundError(f"Selection manifest report is missing: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("selection_schema_version") != SELECTION_SCHEMA_VERSION:
        raise RuntimeError("Unsupported selection manifest schema")
    if report.get("protocol") != expected_protocol:
        raise RuntimeError("Selection manifest protocol mismatch")
    if report.get("selection_count") != expected_count:
        raise RuntimeError("Selection manifest sample count mismatch")
    if report.get("selection_manifest_sha256") != sha256_file(path):
        raise RuntimeError("Selection manifest SHA256 mismatch")
    rows = pd.read_parquet(path).to_dict("records")
    rows.sort(key=lambda row: int(row["selection_index"]))
    if [int(row["selection_index"]) for row in rows] != list(range(expected_count)):
        raise RuntimeError("Selection indices are incomplete or duplicated")
    if sample_id_digest(str(row["sample_id"]) for row in rows) != report["sample_id_digest"]:
        raise RuntimeError("Selection sample ID digest mismatch")
    if _selection_rows_digest(rows) != report["selection_rows_digest"]:
        raise RuntimeError("Selection row digest mismatch")
    for row in rows:
        path_value = Path(str(row["path"])).expanduser().resolve()
        stat = path_value.stat()
        if stat.st_size != int(row["file_size"]) or stat.st_mtime_ns != int(
            row["file_mtime_ns"]
        ):
            raise RuntimeError(f"Selected validation source changed: {path_value}")
    return rows, report


def checkpoint_rank_key(row: dict[str, Any]) -> tuple[float, float, float, int]:
    return (
        float(row["rgb_lpips"]),
        -float(row["rgb_psnr"]),
        -float(row["rgb_ssim"]),
        -int(row["optimizer_step"]),
    )


def rank_checkpoint_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    values = [dict(row) for row in rows]
    for row in values:
        for name in ("rgb_lpips", "rgb_psnr", "rgb_ssim"):
            if not np.isfinite(float(row[name])):
                raise FloatingPointError(f"Non-finite selection metric {name}")
    values.sort(key=checkpoint_rank_key)
    for rank, row in enumerate(values, 1):
        row["rank"] = rank
    return values


def _atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_symlink(path: Path, target: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(target.resolve())
    os.replace(temporary, path)


def _checkpoint_sha_with_inventory(
    output: Path, checkpoint: Path, *, step: int
) -> str:
    inventory_path = output / "checkpoint_inventory.json"
    inventory = (
        json.loads(inventory_path.read_text(encoding="utf-8"))
        if inventory_path.is_file()
        else {"checkpoints": {}}
    )
    key = str(step)
    stat = checkpoint.stat()
    previous = inventory["checkpoints"].get(key)
    if previous is not None:
        if (
            previous["path"] != str(checkpoint)
            or int(previous["size"]) != stat.st_size
            or int(previous["mtime_ns"]) != stat.st_mtime_ns
        ):
            raise RuntimeError(f"Checkpoint file identity changed for step {step}")
        return str(previous["sha256"])
    value = sha256_file(checkpoint)
    inventory["checkpoints"][key] = {
        "path": str(checkpoint),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": value,
    }
    atomic_write_json(inventory_path, inventory)
    return value


def _verify_checkpoint_inventory(output: Path) -> None:
    inventory_path = output / "checkpoint_inventory.json"
    if not inventory_path.is_file():
        return
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    for step, recorded in inventory.get("checkpoints", {}).items():
        path = Path(str(recorded["path"])).expanduser().resolve()
        stat = path.stat()
        if (
            stat.st_size != int(recorded["size"])
            or stat.st_mtime_ns != int(recorded["mtime_ns"])
            or sha256_file(path) != recorded["sha256"]
        ):
            raise RuntimeError(f"Checkpoint file identity changed for step {step}")


def _candidate_result(
    directory: Path,
    *,
    step: int,
    checkpoint: Path,
    geometry: str,
    stage: str,
    selection_report: dict[str, Any],
) -> dict[str, Any] | None:
    manifest_path = directory / "run_manifest.json"
    metrics_path = directory / "metrics.json"
    if not manifest_path.is_file() or not metrics_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        return None
    run_identity = manifest.get("run_identity", {})
    selection_identity = run_identity.get("checkpoint_selection", {})
    expected_geometry = (
        {"rgb_frames": 17, "temporal_latents": 5}
        if geometry == "17f"
        else {"rgb_frames": 33, "temporal_latents": 9}
    )
    expected_identity = {
        "protocol": SELECTION_PROTOCOL,
        "selection_manifest_sha256": selection_report["selection_manifest_sha256"],
        "sample_id_digest": selection_report["sample_id_digest"],
        "sample_count": selection_report["selection_count"],
        "stage": stage,
        "objective": OBJECTIVE_BY_STAGE[stage],
        "optimizer_step": step,
        "evaluation_geometry": expected_geometry,
        "full_endpoint": 47,
        "exhaustive": True,
    }
    mismatches = {
        key: (selection_identity.get(key), expected)
        for key, expected in expected_identity.items()
        if selection_identity.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"Candidate evaluation provenance mismatch: {mismatches}")
    if Path(run_identity.get("checkpoint_path", "")).resolve() != checkpoint:
        raise RuntimeError("Candidate evaluation checkpoint path mismatch")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    summaries = metrics["metrics"]
    return {
        "optimizer_step": step,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": manifest["run_identity"]["checkpoint_sha256"],
        "geometry": geometry,
        "sample_id_digest": metrics["sample_id_digest"],
        "sample_count": int(metrics["num_clips"]),
        "rgb_lpips": float(summaries["rgb_lpips"]["mean"]),
        "rgb_psnr": float(summaries["rgb_psnr"]["mean"]),
        "rgb_ssim": float(summaries["rgb_ssim"]["mean"]),
        "run_dir": str(directory),
    }


def _evaluation_command(
    *,
    config: Path,
    selection_manifest: Path,
    checkpoint: Path,
    stage: str,
    output_dir: Path,
    resume: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "progressive_videorae.evaluate",
        "--config",
        str(config),
        "--selection-manifest",
        str(selection_manifest),
        "--checkpoint",
        str(checkpoint),
        "--expected-stage",
        stage,
        "--output-dir",
        str(output_dir),
    ]
    if resume:
        command.append("--resume")
    return command


def _write_leaderboard(
    output: Path,
    rows: list[dict[str, Any]],
    *,
    selection_report: dict[str, Any],
) -> list[dict[str, Any]]:
    primary = [row for row in rows if row["geometry"] == "17f"]
    ranked = rank_checkpoint_rows(primary)
    rank_by_step = {int(row["optimizer_step"]): int(row["rank"]) for row in ranked}
    combined = [
        {
            **row,
            "rank": (
                rank_by_step.get(int(row["optimizer_step"]))
                if row["geometry"] == "17f"
                else None
            ),
        }
        for row in sorted(rows, key=lambda value: (int(value["optimizer_step"]), value["geometry"]))
    ]
    payload = {
        "selection_schema_version": SELECTION_SCHEMA_VERSION,
        "protocol": SELECTION_PROTOCOL,
        "ranking": "min(rgb_lpips), max(rgb_psnr), max(rgb_ssim), max(step)",
        "sample_id_digest": selection_report["sample_id_digest"],
        "updated_at": utc_now(),
        "rows": combined,
    }
    atomic_write_json(output / "provisional_leaderboard.json", payload)
    _atomic_write_csv(output / "provisional_leaderboard.csv", combined)
    if ranked:
        atomic_write_json(
            output / "provisional_best.json",
            {
                "status": "provisional",
                "protocol": SELECTION_PROTOCOL,
                "updated_at": utc_now(),
                "winner": ranked[0],
                "completed_primary_candidates": len(ranked),
            },
        )
    return ranked


def _finalize_selection(
    *,
    output: Path,
    stage: str,
    resolved: dict[str, Any],
    expected_steps: list[int],
    checkpoints: dict[int, Path],
    rows: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    selection_report: dict[str, Any],
) -> dict[str, Any] | None:
    required_geometries = {"17f", "33f"} if stage == "stage2b" else {"17f"}
    completed = {
        (int(row["optimizer_step"]), str(row["geometry"])) for row in rows
    }
    required = {(step, geometry) for step in expected_steps for geometry in required_geometries}
    if not required.issubset(completed) or not ranked:
        return None
    final_step = expected_steps[-1]
    final_path = checkpoints.get(final_step)
    if final_path is None:
        return None
    metadata = load_selection_candidate_metadata(
        final_path,
        resolved_config=resolved,
        stage=stage,
        expected_step=final_step,
    )
    if not metadata["stage_complete"]:
        return None
    primary = [row for row in rows if row["geometry"] == "17f"]
    final_ranked = rank_checkpoint_rows(primary)
    rank_by_step = {
        int(row["optimizer_step"]): int(row["rank"]) for row in final_ranked
    }
    combined = [
        {
            **row,
            "rank": (
                rank_by_step.get(int(row["optimizer_step"]))
                if row["geometry"] == "17f"
                else None
            ),
        }
        for row in sorted(
            rows,
            key=lambda value: (int(value["optimizer_step"]), value["geometry"]),
        )
    ]
    leaderboard_payload = {
        "selection_schema_version": SELECTION_SCHEMA_VERSION,
        "protocol": SELECTION_PROTOCOL,
        "ranking": "min(rgb_lpips), max(rgb_psnr), max(rgb_ssim), max(step)",
        "sample_id_digest": selection_report["sample_id_digest"],
        "completed_at": utc_now(),
        "rows": combined,
    }
    atomic_write_json(output / "leaderboard.json", leaderboard_payload)
    _atomic_write_csv(output / "leaderboard.csv", combined)
    winner = ranked[0]
    winner_path = Path(winner["checkpoint_path"]).resolve()
    final_sha = sha256_file(final_path)
    certificate = {
        "selection_certificate_schema_version": SELECTION_SCHEMA_VERSION,
        "status": "completed",
        "protocol": SELECTION_PROTOCOL,
        "completed_at": utc_now(),
        "stage": stage,
        "objective": OBJECTIVE_BY_STAGE[stage],
        "run_id": resolved["runtime"]["run_id"],
        "resolved_config_sha256": canonical_json_sha256(resolved),
        "selection_manifest_sha256": selection_report["selection_manifest_sha256"],
        "leaderboard_sha256": sha256_file(output / "leaderboard.json"),
        "sample_id_digest": selection_report["sample_id_digest"],
        "expected_steps": expected_steps,
        "completed_steps": expected_steps,
        "ranking": "min(rgb_lpips), max(rgb_psnr), max(rgb_ssim), max(step)",
        "winner": winner,
        "stage_final": {
            "optimizer_step": final_step,
            "checkpoint_path": str(final_path),
            "checkpoint_sha256": final_sha,
        },
    }
    atomic_write_json(output / "evaluation_best.json", certificate)
    atomic_write_json(
        output / "stage_final.json",
        {key: certificate[key] for key in ("status", "protocol", "stage", "objective", "run_id")}
        | certificate["stage_final"],
    )
    _atomic_symlink(output / "evaluation_best.pt", winner_path)
    _atomic_symlink(output / "stage_final.pt", final_path)
    return certificate


def run_selector(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    resolved_path = checkpoint_dir / "resolved_config.json"
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Missing resolved_config.json: {resolved_path}")
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    training, expected_steps = validate_resolved_stage_config(
        resolved, stage=args.stage, checkpoint_dir=checkpoint_dir
    )
    config17 = resolve_config_path(args.config)
    config33 = resolve_config_path(args.stage2b_config) if args.stage == "stage2b" else None
    validate_selection_evaluation_config(config17, expected_geometry=(17, 5))
    if config33 is not None:
        validate_selection_evaluation_config(config33, expected_geometry=(33, 9))
    validation_manifest = (
        Path(resolved["data"]["manifest_dir"]).expanduser().resolve() / "val.parquet"
    )
    identity = {
        "selection_schema_version": SELECTION_SCHEMA_VERSION,
        "protocol": SELECTION_PROTOCOL,
        "stage": args.stage,
        "objective": training["objective_mode"],
        "run_id": resolved["runtime"]["run_id"],
        "checkpoint_dir": str(checkpoint_dir),
        "resolved_config_sha256": canonical_json_sha256(resolved),
        "validation_manifest": str(validation_manifest),
        "validation_manifest_sha256": sha256_file(validation_manifest),
        "selection_count": SELECTION_COUNT,
        "selection_seed": SELECTION_SEED,
        "expected_steps": expected_steps,
        "config_17f": str(config17),
        "config_17f_sha256": sha256_file(config17),
        "config_33f": str(config33) if config33 is not None else None,
        "config_33f_sha256": sha256_file(config33) if config33 is not None else None,
    }
    outer_manifest_path = output / "run_manifest.json"
    if args.resume:
        if not outer_manifest_path.is_file():
            raise FileNotFoundError("--resume requires selection run_manifest.json")
        outer = json.loads(outer_manifest_path.read_text(encoding="utf-8"))
        if outer.get("status") == "completed":
            raise RuntimeError("Completed checkpoint selections cannot be resumed")
        if outer.get("identity") != identity:
            raise RuntimeError("Selection resume provenance mismatch")
        outer["status"] = "watching" if args.watch else "running"
        outer["resumed_at"] = utc_now()
        atomic_write_json(outer_manifest_path, outer)
        selection_manifest = output / "selection_manifest.parquet"
        _, selection_report = load_selection_manifest(selection_manifest)
        _verify_checkpoint_inventory(output)
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Selection output directory is not empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
        outer = {
            "status": "watching" if args.watch else "running",
            "started_at": utc_now(),
            "identity": identity,
        }
        atomic_write_json(outer_manifest_path, outer)
        selection_manifest, selection_report = prepare_selection_manifest(
            validation_manifest=validation_manifest,
            output_dir=output,
        )
    lock_path = output / "selector.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another selector process owns this output directory") from exc
        lock.seek(0)
        lock.truncate()
        lock.write(json.dumps({"pid": os.getpid(), "host": socket.gethostname()}))
        lock.flush()
        while True:
            checkpoints = discover_checkpoints(checkpoint_dir)
            unexpected = sorted(set(checkpoints) - set(expected_steps))
            if unexpected:
                raise RuntimeError(f"Unexpected checkpoint steps: {unexpected}")
            rows: list[dict[str, Any]] = []
            for step in expected_steps:
                checkpoint = checkpoints.get(step)
                if checkpoint is None:
                    continue
                load_selection_candidate_metadata(
                    checkpoint,
                    resolved_config=resolved,
                    stage=args.stage,
                    expected_step=step,
                )
                checkpoint_sha = _checkpoint_sha_with_inventory(
                    output, checkpoint, step=step
                )
                geometries = [("17f", config17)]
                if args.stage == "stage2b":
                    geometries.append(("33f", config33))
                for geometry, config in geometries:
                    candidate_dir = output / "checkpoints" / f"step_{step:08d}" / geometry
                    result = _candidate_result(
                        candidate_dir,
                        step=step,
                        checkpoint=checkpoint,
                        geometry=geometry,
                        stage=args.stage,
                        selection_report=selection_report,
                    )
                    if result is None:
                        resume_candidate = (candidate_dir / "run_manifest.json").is_file()
                        subprocess.run(
                            _evaluation_command(
                                config=config,
                                selection_manifest=selection_manifest,
                                checkpoint=checkpoint,
                                stage=args.stage,
                                output_dir=candidate_dir,
                                resume=resume_candidate,
                            ),
                            cwd=project_root(),
                            check=True,
                        )
                        result = _candidate_result(
                            candidate_dir,
                            step=step,
                            checkpoint=checkpoint,
                            geometry=geometry,
                            stage=args.stage,
                            selection_report=selection_report,
                        )
                        if result is None:
                            raise RuntimeError(f"Candidate evaluation did not complete: {step}/{geometry}")
                    if result["sample_id_digest"] != selection_report["sample_id_digest"]:
                        raise RuntimeError("Candidate sample digest differs from frozen selection")
                    if result["checkpoint_sha256"] != checkpoint_sha:
                        raise RuntimeError(f"Checkpoint SHA256 changed for step {step}")
                    if result["sample_count"] != SELECTION_COUNT:
                        raise RuntimeError("Candidate did not evaluate exactly 5000 samples")
                    rows.append(result)
            ranked = _write_leaderboard(output, rows, selection_report=selection_report)
            certificate = _finalize_selection(
                output=output,
                stage=args.stage,
                resolved=resolved,
                expected_steps=expected_steps,
                checkpoints=checkpoints,
                rows=rows,
                ranked=ranked,
                selection_report=selection_report,
            )
            outer.update(
                {
                    "updated_at": utc_now(),
                    "completed_candidate_geometries": len(rows),
                    "status": (
                        "completed"
                        if certificate is not None
                        else ("watching" if args.watch else "incomplete")
                    ),
                }
            )
            atomic_write_json(outer_manifest_path, outer)
            if certificate is not None or not args.watch:
                return {"run_manifest": outer, "certificate": certificate, "leaderboard": ranked}
            time.sleep(int(args.poll_seconds))


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    resolved_path = checkpoint_dir / "resolved_config.json"
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Missing resolved_config.json: {resolved_path}")
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    _, expected_steps = validate_resolved_stage_config(
        resolved, stage=args.stage, checkpoint_dir=checkpoint_dir
    )
    step = int(args.smoke_step)
    if step not in expected_steps:
        raise ValueError(f"Smoke step {step} is not in the checkpoint schedule")
    checkpoint = discover_checkpoints(checkpoint_dir).get(step)
    if checkpoint is None:
        raise FileNotFoundError(f"Smoke checkpoint step {step} is unavailable")
    load_selection_candidate_metadata(
        checkpoint,
        resolved_config=resolved,
        stage=args.stage,
        expected_step=step,
    )
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Smoke output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    base_path = resolve_config_path(args.config)
    normalized = validate_selection_evaluation_config(
        base_path, expected_geometry=(17, 5)
    )
    smoke_config = {
        **normalized,
        "purpose": "checkpoint_selection_smoke",
        "max_clips": int(args.smoke_samples),
    }
    smoke_config_path = output / "smoke_evaluation_config.yaml"
    atomic_write_json(smoke_config_path, smoke_config)
    validation_manifest = (
        Path(resolved["data"]["manifest_dir"]).expanduser().resolve()
        / "val.parquet"
    )
    selection_manifest, selection_report = prepare_selection_manifest(
        validation_manifest=validation_manifest,
        output_dir=output,
        count=int(args.smoke_samples),
        protocol=SELECTION_SMOKE_PROTOCOL,
    )
    identity = {
        "status": "running",
        "protocol": SELECTION_SMOKE_PROTOCOL,
        "stage": args.stage,
        "optimizer_step": step,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "sample_count": int(args.smoke_samples),
        "sample_id_digest": selection_report["sample_id_digest"],
        "started_at": utc_now(),
    }
    smoke_manifest_path = output / "smoke_manifest.json"
    atomic_write_json(smoke_manifest_path, identity)
    try:
        subprocess.run(
            _evaluation_command(
                config=smoke_config_path,
                selection_manifest=selection_manifest,
                checkpoint=checkpoint,
                stage=args.stage,
                output_dir=output / "checkpoint",
                resume=False,
            ),
            cwd=project_root(),
            check=True,
        )
    except Exception as exc:
        identity.update(
            status="failed", error=f"{type(exc).__name__}: {exc}", updated_at=utc_now()
        )
        atomic_write_json(smoke_manifest_path, identity)
        raise
    metrics = json.loads((output / "checkpoint" / "metrics.json").read_text(encoding="utf-8"))
    if (
        int(metrics["num_clips"]) != int(args.smoke_samples)
        or metrics["sample_id_digest"] != selection_report["sample_id_digest"]
    ):
        raise RuntimeError("Smoke evaluation did not preserve the frozen samples")
    identity.update(status="completed", completed_at=utc_now())
    atomic_write_json(smoke_manifest_path, identity)
    return {"smoke_manifest": identity, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Incrementally select the best full-state PVR checkpoint"
    )
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--config", default="configs/eval/checkpoint_selection_17f.yaml")
    parser.add_argument(
        "--stage2b-config", default="configs/eval/checkpoint_selection_33f.yaml"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-samples", type=int, default=None)
    parser.add_argument("--smoke-step", type=int, default=100)
    args = parser.parse_args()
    if args.poll_seconds < 10:
        parser.error("--poll-seconds must be at least 10")
    if args.smoke_samples is not None:
        if not 2 <= args.smoke_samples <= 8:
            parser.error("--smoke-samples must be between 2 and 8")
        if args.watch or args.resume:
            parser.error("Smoke mode cannot use --watch or --resume")
        result = run_smoke(args)
    else:
        result = run_selector(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
