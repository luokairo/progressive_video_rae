from __future__ import annotations

from pathlib import Path

import pytest

from progressive_videorae.training.paths import (
    resolve_training_run_paths,
    validate_resume_log_file,
)


def training_paths(tmp_path: Path) -> dict[str, str]:
    return {
        "checkpoint_root": str(tmp_path / "checkpoints"),
        "log_dir": str(tmp_path / "logs"),
    }


def test_fresh_and_init_paths_share_one_run_id_for_checkpoint_and_log(tmp_path: Path):
    training = training_paths(tmp_path)
    paths = resolve_training_run_paths(
        training,
        stage="stage1b",
        run_id="20260810T010203123456Z",
        project_root=tmp_path / "repository",
    )
    assert paths.checkpoint_dir == Path(training["checkpoint_root"]) / "stage1b" / paths.run_id
    assert paths.log_file == Path(training["log_dir"]) / (
        f"stage1b_{paths.run_id}.train.jsonl"
    )


def test_resume_reuses_checkpoint_run_directory_and_saved_log(tmp_path: Path):
    training = training_paths(tmp_path)
    checkpoint = (
        Path(training["checkpoint_root"]) / "stage2b" / "existing-run" / "step_1.pt"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    paths = resolve_training_run_paths(
        training,
        stage="stage2b",
        run_id=None,
        resume_checkpoint=checkpoint,
        project_root=tmp_path / "repository",
    )
    assert paths.run_id == "existing-run"
    assert paths.checkpoint_dir == checkpoint.parent.resolve()
    saved_log = Path(training["log_dir"]) / "stage2b_existing-run.train.jsonl"
    assert validate_resume_log_file(
        saved_log,
        log_root=training["log_dir"],
        project_root=tmp_path / "repository",
    ) == saved_log.resolve()


@pytest.mark.parametrize("key", ["checkpoint_root", "log_dir"])
def test_relative_or_repository_internal_roots_are_rejected(tmp_path: Path, key: str):
    training = training_paths(tmp_path)
    training[key] = "relative/path"
    with pytest.raises(ValueError, match="absolute path"):
        resolve_training_run_paths(
            training,
            stage="stage1a",
            run_id="run",
            project_root=tmp_path / "repository",
        )


def test_repository_internal_root_is_rejected(tmp_path: Path):
    repository = tmp_path / "repository"
    training = training_paths(tmp_path)
    training["checkpoint_root"] = str(repository / "outputs")
    with pytest.raises(ValueError, match="outside the code repository"):
        resolve_training_run_paths(
            training,
            stage="stage1a",
            run_id="run",
            project_root=repository,
        )


def test_resume_checkpoint_outside_configured_root_is_rejected(tmp_path: Path):
    training = training_paths(tmp_path)
    checkpoint = tmp_path / "unmanaged" / "stage1a" / "run" / "step.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    with pytest.raises(ValueError, match="under configured stage root"):
        resolve_training_run_paths(
            training,
            stage="stage1a",
            run_id=None,
            resume_checkpoint=checkpoint,
            project_root=tmp_path / "repository",
        )
