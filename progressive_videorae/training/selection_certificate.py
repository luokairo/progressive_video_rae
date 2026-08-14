from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from ..checksums import sha256_file
from ..model.types import StateContract


SELECTION_CERTIFICATE_SCHEMA_VERSION = 1
SELECTION_PROTOCOL = "pvr_validation_5000_rgb_lpips_v1"
SELECTION_RANKING = "min(rgb_lpips), max(rgb_psnr), max(rgb_ssim), max(step)"


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _load_checkpoint_metadata(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(
        str(path), map_location="cpu", mmap=True, weights_only=False
    )
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint is not a dictionary: {path}")
    return checkpoint


def validate_selection_certificate(
    certificate_path: str | Path,
    *,
    source_checkpoint: str | Path,
    target_stage: str,
    previous_stage: str,
    previous_objective: str,
    checkpoint_schema_version: int,
) -> dict[str, Any]:
    path = Path(certificate_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Selection certificate is unavailable: {path}")
    certificate = json.loads(path.read_text(encoding="utf-8"))
    if certificate.get("selection_certificate_schema_version") != (
        SELECTION_CERTIFICATE_SCHEMA_VERSION
    ):
        raise RuntimeError("Unsupported selection certificate schema")
    required = {
        "status": "completed",
        "protocol": SELECTION_PROTOCOL,
        "stage": previous_stage,
        "objective": previous_objective,
        "ranking": SELECTION_RANKING,
    }
    mismatches = {
        key: (certificate.get(key), expected)
        for key, expected in required.items()
        if certificate.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"Selection certificate identity mismatch: {mismatches}")

    expected_steps = [int(value) for value in certificate.get("expected_steps", ())]
    completed_steps = [int(value) for value in certificate.get("completed_steps", ())]
    if not expected_steps or expected_steps != sorted(set(expected_steps)):
        raise RuntimeError("Selection certificate has an invalid candidate set")
    if completed_steps != expected_steps:
        raise RuntimeError("Selection certificate does not cover every candidate")

    winner = certificate.get("winner")
    final = certificate.get("stage_final")
    if not isinstance(winner, dict) or not isinstance(final, dict):
        raise RuntimeError("Selection certificate is missing winner/final identities")
    winner_step = int(winner.get("optimizer_step", -1))
    if winner_step not in expected_steps or int(winner.get("rank", -1)) != 1:
        raise RuntimeError("Selection winner is not a ranked candidate")
    if winner.get("geometry") != "17f" or int(winner.get("sample_count", -1)) != 5000:
        raise RuntimeError("Selection winner is not the 17-frame 5000-sample result")
    if winner.get("sample_id_digest") != certificate.get("sample_id_digest"):
        raise RuntimeError("Selection winner sample digest mismatch")
    if int(final.get("optimizer_step", -1)) != expected_steps[-1]:
        raise RuntimeError("Selection certificate final step is invalid")

    source = Path(source_checkpoint).expanduser().resolve()
    winner_path = Path(str(winner.get("checkpoint_path", ""))).expanduser().resolve()
    final_path = Path(str(final.get("checkpoint_path", ""))).expanduser().resolve()
    if source != winner_path:
        raise RuntimeError(
            f"--init-from is not the certified winner for {target_stage}"
        )
    if sha256_file(source) != winner.get("checkpoint_sha256"):
        raise RuntimeError("Certified winner checkpoint SHA256 mismatch")
    if not final_path.is_file() or sha256_file(final_path) != final.get("checkpoint_sha256"):
        raise RuntimeError("Certified stage-final checkpoint SHA256 mismatch")

    leaderboard_path = path.parent / "leaderboard.json"
    if not leaderboard_path.is_file():
        raise FileNotFoundError("Selection certificate leaderboard.json is missing")
    if sha256_file(leaderboard_path) != certificate.get("leaderboard_sha256"):
        raise RuntimeError("Selection leaderboard SHA256 mismatch")
    leaderboard = json.loads(leaderboard_path.read_text(encoding="utf-8"))
    if (
        leaderboard.get("protocol") != SELECTION_PROTOCOL
        or leaderboard.get("sample_id_digest") != certificate.get("sample_id_digest")
    ):
        raise RuntimeError("Selection leaderboard provenance mismatch")
    rows = leaderboard.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError("Selection leaderboard rows are missing")
    required_geometries = {"17f", "33f"} if previous_stage == "stage2b" else {"17f"}
    completed = {
        (int(row["optimizer_step"]), str(row["geometry"])) for row in rows
    }
    required_rows = {
        (step, geometry) for step in expected_steps for geometry in required_geometries
    }
    if completed != required_rows:
        raise RuntimeError("Selection leaderboard does not exactly cover all candidates")
    for row in rows:
        run_dir = Path(str(row.get("run_dir", ""))).expanduser().resolve()
        if not run_dir.is_relative_to((path.parent / "checkpoints").resolve()):
            raise RuntimeError("Selection candidate run directory escaped the certificate")
        run_manifest_path = run_dir / "run_manifest.json"
        metrics_path = run_dir / "metrics.json"
        if not run_manifest_path.is_file() or not metrics_path.is_file():
            raise FileNotFoundError("A certified candidate result is missing")
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        run_identity = run_manifest.get("run_identity", {})
        identity = run_identity.get("checkpoint_selection", {})
        expected_geometry = (
            {"rgb_frames": 17, "temporal_latents": 5}
            if row["geometry"] == "17f"
            else {"rgb_frames": 33, "temporal_latents": 9}
        )
        if run_manifest.get("status") != "completed":
            raise RuntimeError("A certified candidate run is incomplete")
        if (
            identity.get("protocol") != SELECTION_PROTOCOL
            or identity.get("stage") != previous_stage
            or identity.get("objective") != previous_objective
            or int(identity.get("optimizer_step", -1)) != int(row["optimizer_step"])
            or identity.get("sample_id_digest") != certificate.get("sample_id_digest")
            or identity.get("selection_manifest_sha256")
            != certificate.get("selection_manifest_sha256")
            or int(identity.get("sample_count", -1)) != 5000
            or identity.get("evaluation_geometry") != expected_geometry
            or identity.get("full_endpoint") != 47
            or run_identity.get("checkpoint_sha256") != row["checkpoint_sha256"]
        ):
            raise RuntimeError("A certified candidate run has mismatched provenance")
        if (
            metrics.get("sample_id_digest") != certificate.get("sample_id_digest")
            or int(metrics.get("num_clips", -1)) != 5000
        ):
            raise RuntimeError("A certified candidate has mismatched samples")
        summaries = metrics.get("metrics", {})
        for name in ("rgb_lpips", "rgb_psnr", "rgb_ssim"):
            if float(summaries[name]["mean"]) != float(row[name]):
                raise RuntimeError(f"Certified candidate {name} differs from metrics.json")
    ranked_primary = sorted(
        (row for row in rows if row["geometry"] == "17f"),
        key=lambda row: int(row["rank"]),
    )
    if not ranked_primary or ranked_primary[0]["checkpoint_sha256"] != winner["checkpoint_sha256"]:
        raise RuntimeError("Selection certificate winner differs from leaderboard rank 1")
    recomputed = sorted(
        ranked_primary,
        key=lambda row: (
            float(row["rgb_lpips"]),
            -float(row["rgb_psnr"]),
            -float(row["rgb_ssim"]),
            -int(row["optimizer_step"]),
        ),
    )
    if [int(row["optimizer_step"]) for row in recomputed] != [
        int(row["optimizer_step"]) for row in ranked_primary
    ]:
        raise RuntimeError("Selection leaderboard ranking is invalid")

    winner_checkpoint = _load_checkpoint_metadata(source)
    final_checkpoint = _load_checkpoint_metadata(final_path)
    for label, checkpoint in (("winner", winner_checkpoint), ("final", final_checkpoint)):
        if checkpoint.get("checkpoint_schema_version") != checkpoint_schema_version:
            raise RuntimeError(f"Certified {label} checkpoint schema mismatch")
        if checkpoint.get("stage") != previous_stage:
            raise RuntimeError(f"Certified {label} checkpoint stage mismatch")
        if checkpoint.get("objective_mode") != previous_objective:
            raise RuntimeError(f"Certified {label} checkpoint objective mismatch")
        if checkpoint.get("run_id") != certificate.get("run_id"):
            raise RuntimeError(f"Certified {label} checkpoint run ID mismatch")
        if checkpoint.get("state_contract") != StateContract().to_dict():
            raise RuntimeError(f"Certified {label} StateContract mismatch")
        if _canonical_json_sha256(checkpoint.get("config")) != certificate.get(
            "resolved_config_sha256"
        ):
            raise RuntimeError(f"Certified {label} resolved config mismatch")
    if int(winner_checkpoint.get("optimizer_step", -1)) != winner_step:
        raise RuntimeError("Certified winner optimizer step mismatch")
    if bool(final_checkpoint.get("stage_complete")) is not True:
        raise RuntimeError("Certified stage-final checkpoint is incomplete")
    if int(final_checkpoint.get("optimizer_step", -1)) != expected_steps[-1]:
        raise RuntimeError("Certified final checkpoint optimizer step mismatch")
    training = final_checkpoint["config"]["training"]
    maximum = int(training["max_steps"])
    save_every = int(training["save_every"])
    derived_steps = set(range(save_every, maximum + 1, save_every))
    derived_steps.update(int(value) for value in training.get("save_at_steps", ()))
    derived_steps.add(maximum)
    if expected_steps != sorted(derived_steps):
        raise RuntimeError("Certificate candidate set differs from the training schedule")
    return certificate


__all__ = ["validate_selection_certificate"]
