from __future__ import annotations

import pytest
import torch
from torch import nn

from progressive_videorae.config import load_training_bundle
from progressive_videorae.training.stages import (
    adversarial_factor,
    configure_stage,
    sample_microbatch_tasks,
    stage1a_phase,
    validate_stage_objective,
    validate_training_bundle,
)
from progressive_videorae.train import validate_resume_training_contract


def test_stage1a_phase_boundaries_are_exact():
    assert stage1a_phase(0) == "warmup"
    assert stage1a_phase(1999) == "warmup"
    assert stage1a_phase(2000) == "interface"
    assert stage1a_phase(4999) == "interface"
    assert stage1a_phase(5000) == "full"
    with pytest.raises(ValueError, match="boundaries"):
        stage1a_phase(0, wan_interface_step=5, wan_full_step=4)


def test_stage1b_plan_is_exactly_4_full_3_single_1_pair():
    config = load_training_bundle("configs/train/stage1b.yaml")["training"]
    assert validate_stage_objective(config) == "full_repa_spatial_prefix"
    tasks = sample_microbatch_tasks("stage1b", 8, optimizer_step=0)
    kinds = [task.kind for task in tasks]
    assert kinds.count("full") == 4
    assert kinds.count("single_prefix") == 3
    assert kinds.count("paired_prefix") == 1
    pair = next(task for task in tasks if task.kind == "paired_prefix")
    assert pair.previous_endpoint == pair.endpoint - 1


def test_stage1b_resume_rejects_new_four_microbatch_schedule():
    training = load_training_bundle("configs/train/stage1b.yaml")["training"]
    saved = dict(training)
    saved["gradient_accumulation_steps"] = 4
    saved["global_batch_size"] = 32
    saved["prefix_schedule"] = "alternating_2step_2full_2single__2full_1single_1pair"

    with pytest.raises(RuntimeError, match="Resume training contract mismatch"):
        validate_resume_training_contract({"config": {"training": saved}}, training)


@pytest.mark.parametrize("stage", ["stage2a", "stage2b"])
def test_stage2_plans_are_full_only_and_keep_repa_objective(stage):
    config = load_training_bundle(f"configs/train/{stage}.yaml")["training"]
    assert validate_stage_objective(config).startswith("full_repa")
    assert all(task.is_full for task in sample_microbatch_tasks(stage, 8))
    assert not any(key.startswith("prefix") for key in config)


def test_stage2_rejects_any_prefix_configuration():
    config = load_training_bundle("configs/train/stage2a.yaml")["training"]
    config["prefix_schedule"] = "full"
    with pytest.raises(ValueError, match="forbids prefix/DCT/mask-replacement"):
        validate_stage_objective(config)


@pytest.mark.parametrize(
    ("stage", "frames", "latents"),
    (
        ("stage1a", 17, 5),
        ("stage1b", 17, 5),
        ("stage2a", 17, 5),
        ("stage2b", 33, 9),
    ),
)
def test_training_bundle_preflight_enforces_stage_geometry(stage, frames, latents):
    bundle = load_training_bundle(f"configs/train/{stage}.yaml")

    assert validate_training_bundle(
        bundle["training"], bundle["model"], bundle["data"]
    ) == (frames, latents)


def test_training_bundle_preflight_runs_after_frame_override():
    bundle = load_training_bundle("configs/train/stage2b.yaml")
    bundle["data"]["num_frames"] = 17

    with pytest.raises(ValueError, match="exactly 33 RGB frames"):
        validate_training_bundle(bundle["training"], bundle["model"], bundle["data"])


def test_stage2b_preflight_rejects_encoder_context_below_34():
    bundle = load_training_bundle("configs/train/stage2b.yaml")
    bundle["model"]["encoder"]["max_context_frames"] = 32

    with pytest.raises(ValueError, match="exceeds context"):
        validate_training_bundle(bundle["training"], bundle["model"], bundle["data"])


@pytest.mark.parametrize("key", ["dct_weight", "mask_replacement_probability"])
def test_stage2_rejects_dct_and_mask_replacement_configuration(key):
    bundle = load_training_bundle("configs/train/stage2a.yaml")
    bundle["training"][key] = 1.0

    with pytest.raises(ValueError, match="forbids prefix/DCT/mask-replacement"):
        validate_training_bundle(bundle["training"], bundle["model"], bundle["data"])


class _DummyWanCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Linear(1, 1)
        self.time_conv = nn.Linear(1, 1)
        self.spatial = nn.Linear(1, 1)


class _DummyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal_adapter = nn.Linear(1, 1)
        self.pre_decoder = nn.Linear(1, 1)
        self.decoder = _DummyWanCore()


class _DummyProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Linear(1, 1)
        self.shared_mask_set = nn.Parameter(torch.zeros(1))


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(1, 1)
        self.projector = _DummyProjector()
        self.repa_projection = nn.Linear(1, 1)
        self.decoder = _DummyDecoder()


def test_stage1a_trainable_ownership_changes_at_boundaries():
    model = _DummyModel()
    configure_stage(model, "stage1a", optimizer_step=0)
    assert model.decoder.temporal_adapter.weight.requires_grad
    assert not model.decoder.pre_decoder.weight.requires_grad
    assert not model.projector.shared_mask_set.requires_grad

    configure_stage(model, "stage1a", optimizer_step=2000)
    assert model.decoder.pre_decoder.weight.requires_grad
    assert model.decoder.decoder.conv1.weight.requires_grad
    assert model.decoder.decoder.time_conv.weight.requires_grad
    assert not model.decoder.decoder.spatial.weight.requires_grad

    configure_stage(model, "stage1a", optimizer_step=5000)
    assert model.decoder.decoder.spatial.weight.requires_grad
    assert not model.projector.shared_mask_set.requires_grad


def test_adversarial_factor_delays_and_ramps_linearly():
    assert adversarial_factor(4999, 5000, 2000) == 0.0
    assert adversarial_factor(6000, 5000, 2000) == pytest.approx(0.5)
    assert adversarial_factor(7000, 5000, 2000) == 1.0
