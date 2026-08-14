from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from progressive_videorae.checksums import sha256_file
from progressive_videorae.config import load_yaml
from progressive_videorae.evaluation.full import validate_evaluation_config
from progressive_videorae.evaluation.selection import (
    SELECTION_PROTOCOL,
    canonical_json_sha256,
    derive_expected_checkpoint_steps,
    discover_checkpoints,
    load_selection_manifest,
    prepare_selection_manifest,
    rank_checkpoint_rows,
    stable_validation_candidates,
    validate_selection_candidate_metadata,
    _checkpoint_sha_with_inventory,
    _verify_checkpoint_inventory,
)
from progressive_videorae.model.types import StateContract
from progressive_videorae.training.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    validate_checkpoint_transition,
)
from progressive_videorae.training.selection_certificate import (
    validate_selection_certificate,
)


def test_stable_5000_selection_ignores_input_order() -> None:
    rows = [{"sample_id": f"sample-{index:05d}"} for index in range(10591)]
    forward = stable_validation_candidates(rows)[:5000]
    reverse = stable_validation_candidates(reversed(rows))[:5000]
    assert [row["sample_id"] for row in forward] == [
        row["sample_id"] for row in reverse
    ]


def test_expected_steps_and_checkpoint_discovery(tmp_path: Path) -> None:
    assert derive_expected_checkpoint_steps(
        {"max_steps": 10000, "save_every": 1000, "save_at_steps": [100, 500]}
    ) == [100, 500, *range(1000, 10001, 1000)]
    for name in (
        "step_00001000.pt",
        "step_00000100.pt",
        ".step_00000500.pt.12.tmp",
        "latest.pt",
    ):
        (tmp_path / name).write_bytes(b"checkpoint")
    assert list(discover_checkpoints(tmp_path)) == [100, 1000]


def test_resume_rejects_changed_checkpoint_hash(tmp_path: Path) -> None:
    checkpoint = tmp_path / "step_00000100.pt"
    checkpoint.write_bytes(b"first")
    _checkpoint_sha_with_inventory(tmp_path, checkpoint, step=100)
    _verify_checkpoint_inventory(tmp_path)
    checkpoint.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="identity changed"):
        _verify_checkpoint_inventory(tmp_path)


def test_manifest_preparation_freezes_deterministic_replacements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = []
    for index in range(8):
        path = tmp_path / f"{index}.mp4"
        path.write_bytes(str(index).encode())
        paths.append(path)
    source = tmp_path / "val.parquet"
    pd.DataFrame(
        {
            "sample_id": [f"s{index}" for index in range(8)],
            "path": [str(path) for path in paths],
            "path_exists": True,
            "decode_valid": True,
            "duration": 3.0,
        }
    ).to_parquet(source, index=False)
    ordered = stable_validation_candidates(
        [{"sample_id": f"s{index}"} for index in range(8)]
    )
    failed_id = ordered[0]["sample_id"]

    def fake_decode(path: str, config):
        sample_id = f"s{Path(path).stem}"
        if sample_id == failed_id:
            raise RuntimeError("frozen decode failure")
        return torch.zeros(1), {
            "sampled_timestamps": torch.arange(33, dtype=torch.float64) / 12,
        }

    monkeypatch.setattr(
        "progressive_videorae.evaluation.selection.decode_contiguous_clip", fake_decode
    )
    output = tmp_path / "selection"
    manifest, report = prepare_selection_manifest(
        validation_manifest=source, output_dir=output, count=3
    )
    rows, loaded_report = load_selection_manifest(manifest, expected_count=3)
    assert failed_id not in {row["sample_id"] for row in rows}
    assert [row["sample_id"] for row in rows] == [
        row["sample_id"] for row in ordered[1:4]
    ]
    assert rows[0]["replacement_for"] == failed_id
    assert all(row["replacement_for"] is None for row in rows[1:])
    assert report == loaded_report
    assert report["decode_failures"] == 1


def test_lpips_ranking_and_complete_tie_break() -> None:
    rows = [
        {"optimizer_step": 100, "rgb_lpips": 0.2, "rgb_psnr": 30, "rgb_ssim": 0.9},
        {"optimizer_step": 200, "rgb_lpips": 0.1, "rgb_psnr": 20, "rgb_ssim": 0.8},
        {"optimizer_step": 300, "rgb_lpips": 0.1, "rgb_psnr": 20, "rgb_ssim": 0.8},
        {"optimizer_step": 400, "rgb_lpips": 0.1, "rgb_psnr": 21, "rgb_ssim": 0.7},
    ]
    assert [row["optimizer_step"] for row in rank_checkpoint_rows(rows)] == [
        400,
        300,
        200,
        100,
    ]


def _resolved(stage: str, root: Path, *, maximum: int = 1000) -> dict:
    objective = {
        "stage1a": "full_repa",
        "stage1b": "nested_spectral_hrepa_full_anchor",
    }[stage]
    return {
        "training": {
            "stage": stage,
            "objective_mode": objective,
            "max_steps": maximum,
            "save_every": maximum,
            "save_at_steps": [],
        },
        "model": {},
        "data": {},
        "runtime": {"run_id": "run-1", "checkpoint_dir": str(root)},
    }


def _checkpoint(resolved: dict, *, step: int, complete: bool) -> dict:
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "stage": resolved["training"]["stage"],
        "objective_mode": resolved["training"]["objective_mode"],
        "optimizer_step": step,
        "stage_max_steps": resolved["training"]["max_steps"],
        "stage_complete": complete,
        "state_contract": StateContract().to_dict(),
        "run_id": resolved["runtime"]["run_id"],
        "checkpoint_dir": resolved["runtime"]["checkpoint_dir"],
        "config": resolved,
        "representation_identity": {},
    }


def test_intermediate_candidate_is_evaluable_but_not_handoff_without_certificate(
    tmp_path: Path,
) -> None:
    resolved = _resolved("stage1a", tmp_path)
    candidate = _checkpoint(resolved, step=100, complete=False)
    assert validate_selection_candidate_metadata(
        candidate, resolved_config=resolved, stage="stage1a", expected_step=100
    )["stage_complete"] is False
    with pytest.raises(RuntimeError, match="completed stage1a"):
        validate_checkpoint_transition(
            candidate,
            target_stage="stage1b",
            target_objective_mode="nested_spectral_hrepa_full_anchor",
            mode="init",
            allow_smoke_checkpoint=False,
        )
    validate_checkpoint_transition(
        candidate,
        target_stage="stage1b",
        target_objective_mode="nested_spectral_hrepa_full_anchor",
        mode="init",
        allow_smoke_checkpoint=False,
        selected_intermediate=True,
    )


@pytest.mark.parametrize(
    ("config_name", "expected_geometry"),
    [
        ("checkpoint_selection_17f.yaml", (17, 5)),
        ("checkpoint_selection_33f.yaml", (33, 9)),
    ],
)
def test_selection_configs_are_static_and_full_only(
    config_name: str, expected_geometry: tuple[int, int]
) -> None:
    config = load_yaml(Path("configs/eval") / config_name)
    model = load_yaml(config["model_config"])
    data = load_yaml(config["data_config"])
    normalized = validate_evaluation_config(
        config, model, data, require_runtime_device=False, check_paths=False
    )
    assert (data["num_frames"], model["state"]["num_frames"]) == expected_geometry
    assert normalized["purpose"] == "checkpoint_selection"
    assert normalized["compute_vjepa"] is False
    assert "endpoints" not in normalized


def test_two_sample_smoke_does_not_relax_formal_selection() -> None:
    config = load_yaml("configs/eval/checkpoint_selection_17f.yaml")
    model = load_yaml(config["model_config"])
    data = load_yaml(config["data_config"])
    with pytest.raises(ValueError, match="exactly 5000"):
        validate_evaluation_config(
            {**config, "max_clips": 2},
            model,
            data,
            require_runtime_device=False,
            check_paths=False,
        )
    smoke = validate_evaluation_config(
        {**config, "purpose": "checkpoint_selection_smoke", "max_clips": 2},
        model,
        data,
        require_runtime_device=False,
        check_paths=False,
    )
    assert smoke["max_clips"] == 2


def test_completed_certificate_allows_only_certified_winner(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "training" / "stage1a" / "run-1"
    checkpoint_dir.mkdir(parents=True)
    resolved = _resolved("stage1a", checkpoint_dir, maximum=1000)
    resolved["training"]["save_at_steps"] = [100]
    winner_path = checkpoint_dir / "step_00000100.pt"
    final_path = checkpoint_dir / "step_00001000.pt"
    torch.save(_checkpoint(resolved, step=100, complete=False), winner_path)
    torch.save(_checkpoint(resolved, step=1000, complete=True), final_path)
    output = tmp_path / "selection" / "stage1a" / "run-1"
    output.mkdir(parents=True)
    digest = "a" * 64
    winner = {
        "optimizer_step": 100,
        "checkpoint_path": str(winner_path),
        "checkpoint_sha256": sha256_file(winner_path),
        "geometry": "17f",
        "sample_id_digest": digest,
        "sample_count": 5000,
        "rgb_lpips": 0.1,
        "rgb_psnr": 30.0,
        "rgb_ssim": 0.9,
        "run_dir": str(output / "checkpoints/step_00000100/17f"),
        "rank": 1,
    }
    final_row = {
        **winner,
        "optimizer_step": 1000,
        "checkpoint_path": str(final_path),
        "checkpoint_sha256": sha256_file(final_path),
        "rgb_lpips": 0.2,
        "run_dir": str(output / "checkpoints/step_00001000/17f"),
        "rank": 2,
    }
    leaderboard = {
        "selection_schema_version": 1,
        "protocol": SELECTION_PROTOCOL,
        "ranking": "min(rgb_lpips), max(rgb_psnr), max(rgb_ssim), max(step)",
        "sample_id_digest": digest,
        "rows": [winner, final_row],
    }
    leaderboard_path = output / "leaderboard.json"
    leaderboard_path.write_text(json.dumps(leaderboard), encoding="utf-8")
    for row in (winner, final_row):
        run_dir = Path(row["run_dir"])
        run_dir.mkdir(parents=True)
        (run_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "run_identity": {
                        "checkpoint_sha256": row["checkpoint_sha256"],
                        "checkpoint_selection": {
                            "protocol": SELECTION_PROTOCOL,
                            "stage": "stage1a",
                            "objective": "full_repa",
                            "optimizer_step": row["optimizer_step"],
                            "sample_id_digest": digest,
                            "selection_manifest_sha256": "b" * 64,
                            "sample_count": 5000,
                            "evaluation_geometry": {
                                "rgb_frames": 17,
                                "temporal_latents": 5,
                            },
                            "full_endpoint": 47,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "num_clips": 5000,
                    "sample_id_digest": digest,
                    "metrics": {
                        name: {"mean": row[name]}
                        for name in ("rgb_lpips", "rgb_psnr", "rgb_ssim")
                    },
                }
            ),
            encoding="utf-8",
        )
    certificate = {
        "selection_certificate_schema_version": 1,
        "status": "completed",
        "protocol": SELECTION_PROTOCOL,
        "stage": "stage1a",
        "objective": "full_repa",
        "run_id": "run-1",
        "resolved_config_sha256": canonical_json_sha256(resolved),
        "selection_manifest_sha256": "b" * 64,
        "sample_id_digest": digest,
        "leaderboard_sha256": sha256_file(leaderboard_path),
        "expected_steps": [100, 1000],
        "completed_steps": [100, 1000],
        "ranking": "min(rgb_lpips), max(rgb_psnr), max(rgb_ssim), max(step)",
        "winner": winner,
        "stage_final": {
            "optimizer_step": 1000,
            "checkpoint_path": str(final_path),
            "checkpoint_sha256": sha256_file(final_path),
        },
    }
    certificate_path = output / "evaluation_best.json"
    certificate_path.write_text(json.dumps(certificate), encoding="utf-8")
    validated = validate_selection_certificate(
        certificate_path,
        source_checkpoint=winner_path,
        target_stage="stage1b",
        previous_stage="stage1a",
        previous_objective="full_repa",
        checkpoint_schema_version=CHECKPOINT_SCHEMA_VERSION,
    )
    assert validated["winner"]["optimizer_step"] == 100
    with pytest.raises(RuntimeError, match="not the certified winner"):
        validate_selection_certificate(
            certificate_path,
            source_checkpoint=final_path,
            target_stage="stage1b",
            previous_stage="stage1a",
            previous_objective="full_repa",
            checkpoint_schema_version=CHECKPOINT_SCHEMA_VERSION,
        )
