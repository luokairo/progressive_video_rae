from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..model.types import StateContract
from ..training.checkpoint import CHECKPOINT_SCHEMA_VERSION, OBJECTIVE_BY_STAGE
from ..training.stages import STAGE_GEOMETRY, STAGES


def validate_stage_checkpoint_metadata(
    checkpoint: dict[str, Any],
    *,
    expected_stage: str,
    evaluation_frames: int = 17,
    evaluation_latents: int = 5,
) -> dict[str, Any]:
    if expected_stage not in STAGES:
        raise ValueError(f"Unsupported expected stage: {expected_stage}")
    schema = int(checkpoint.get("checkpoint_schema_version", 0))
    if schema != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(f"Evaluation requires checkpoint schema v4, got v{schema}")
    stage = checkpoint.get("stage")
    objective = checkpoint.get("objective_mode")
    if stage != expected_stage:
        raise RuntimeError(f"Checkpoint stage is {stage!r}, expected {expected_stage!r}")
    expected_objective = OBJECTIVE_BY_STAGE[expected_stage]
    if objective != expected_objective:
        raise RuntimeError(
            f"Checkpoint objective is {objective!r}, expected {expected_objective!r}"
        )
    optimizer_step = checkpoint.get("optimizer_step")
    stage_max_steps = checkpoint.get("stage_max_steps")
    if not isinstance(optimizer_step, int) or optimizer_step < 0:
        raise RuntimeError("Checkpoint optimizer_step is invalid")
    if not isinstance(stage_max_steps, int) or stage_max_steps <= 0:
        raise RuntimeError("Checkpoint stage_max_steps is invalid")
    if checkpoint.get("stage_complete") is not True or optimizer_step < stage_max_steps:
        raise RuntimeError("Formal stage evaluation requires stage_complete=true")
    expected_contract = StateContract().to_dict()
    if checkpoint.get("state_contract") != expected_contract:
        raise RuntimeError("Checkpoint StateContract is missing or incompatible")
    representation = checkpoint.get("representation_identity")
    if not isinstance(representation, dict) or not representation:
        raise RuntimeError("Checkpoint representation identity is missing")
    training_frames, training_latents = STAGE_GEOMETRY[expected_stage]
    bundle = checkpoint.get("config")
    if not isinstance(bundle, dict):
        raise RuntimeError("Checkpoint resolved config bundle is missing")
    training = bundle.get("training")
    model = bundle.get("model")
    data = bundle.get("data")
    if not all(isinstance(value, dict) for value in (training, model, data)):
        raise RuntimeError("Checkpoint resolved config bundle is incomplete")
    saved_geometry = (
        int(data.get("num_frames", -1)),
        int(model.get("state", {}).get("num_frames", -1)),
    )
    if saved_geometry != (training_frames, training_latents):
        raise RuntimeError(
            f"Checkpoint training geometry is {saved_geometry}, expected "
            f"{(training_frames, training_latents)} for {expected_stage}"
        )
    if training.get("stage") != expected_stage or training.get("objective_mode") != expected_objective:
        raise RuntimeError("Checkpoint resolved training config stage/objective mismatch")
    if (evaluation_frames, evaluation_latents) != (17, 5):
        raise ValueError("Benchmark evaluation geometry must be 17 RGB frames / 5 latents")
    return {
        "checkpoint_schema_version": schema,
        "stage": expected_stage,
        "objective": expected_objective,
        "optimizer_step": optimizer_step,
        "stage_max_steps": stage_max_steps,
        "stage_complete": True,
        "training_geometry": {
            "rgb_frames": training_frames,
            "temporal_latents": training_latents,
        },
        "evaluation_geometry": {
            "rgb_frames": evaluation_frames,
            "temporal_latents": evaluation_latents,
        },
        "representation_identity": representation,
        "state_contract": expected_contract,
    }


def load_stage_checkpoint_metadata(
    path: str | Path,
    *,
    expected_stage: str,
    evaluation_frames: int = 17,
    evaluation_latents: int = 5,
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    try:
        checkpoint = torch.load(
            str(resolved), map_location="cpu", mmap=True, weights_only=False
        )
    except RuntimeError as exc:
        raise RuntimeError(f"Unable to mmap checkpoint metadata: {resolved}") from exc
    if not isinstance(checkpoint, dict):
        raise TypeError("Training checkpoint must contain a dictionary")
    return validate_stage_checkpoint_metadata(
        checkpoint,
        expected_stage=expected_stage,
        evaluation_frames=evaluation_frames,
        evaluation_latents=evaluation_latents,
    )
