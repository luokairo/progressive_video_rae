from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class TrainingRunPaths:
    run_id: str
    checkpoint_root: Path
    checkpoint_dir: Path
    log_root: Path
    log_file: Path | None


def utc_run_id(now: datetime | None = None) -> str:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ValueError("Run timestamp must be timezone-aware")
    return current.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def validate_external_absolute_path(
    path: str | Path,
    *,
    label: str,
    project_root: str | Path = PROJECT_ROOT,
) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        raise ValueError(f"{label} must be an absolute path: {candidate}")
    resolved = candidate.resolve()
    repository = Path(project_root).expanduser().resolve()
    if resolved == repository or resolved.is_relative_to(repository):
        raise ValueError(f"{label} must be outside the code repository: {resolved}")
    return resolved


def resolve_training_run_paths(
    training: Mapping[str, Any],
    *,
    stage: str,
    run_id: str | None,
    resume_checkpoint: str | Path | None = None,
    project_root: str | Path = PROJECT_ROOT,
) -> TrainingRunPaths:
    checkpoint_root = validate_external_absolute_path(
        training["checkpoint_root"], label="checkpoint_root", project_root=project_root
    )
    log_root = validate_external_absolute_path(
        training["log_dir"], label="log_dir", project_root=project_root
    )
    stage_root = checkpoint_root / stage
    if resume_checkpoint is not None:
        checkpoint = Path(resume_checkpoint).expanduser()
        if not checkpoint.is_absolute():
            raise ValueError("Resume checkpoint must be an absolute path")
        checkpoint = checkpoint.resolve()
        if not checkpoint.is_relative_to(stage_root):
            raise ValueError(
                f"Resume checkpoint must be under configured stage root: {stage_root}"
            )
        checkpoint_dir = checkpoint.parent
        resolved_run_id = checkpoint_dir.name
        log_file = None
    else:
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("Fresh/init runs require a non-empty run_id")
        resolved_run_id = run_id.strip()
        if Path(resolved_run_id).name != resolved_run_id:
            raise ValueError("run_id must be one path component")
        checkpoint_dir = stage_root / resolved_run_id
        log_file = log_root / f"{stage}_{resolved_run_id}.train.jsonl"
    return TrainingRunPaths(
        run_id=resolved_run_id,
        checkpoint_root=checkpoint_root,
        checkpoint_dir=checkpoint_dir,
        log_root=log_root,
        log_file=log_file,
    )


def validate_resume_log_file(
    path: str | Path, *, log_root: str | Path, project_root: str | Path = PROJECT_ROOT
) -> Path:
    resolved = validate_external_absolute_path(
        path, label="resume log_file", project_root=project_root
    )
    root = Path(log_root).expanduser().resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Resume log_file must be under configured log_dir: {root}")
    return resolved
