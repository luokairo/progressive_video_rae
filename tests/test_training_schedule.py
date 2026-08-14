from __future__ import annotations

import random

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
from progressive_videorae.train import (
    build_optimizers,
    validate_resume_training_contract,
)


def test_stage1a_phase_boundaries_are_exact():
    assert stage1a_phase(0) == "warmup"
    assert stage1a_phase(1999) == "warmup"
    assert stage1a_phase(2000) == "interface"
    assert stage1a_phase(4999) == "interface"
    assert stage1a_phase(5000) == "full"
    with pytest.raises(ValueError, match="boundaries"):
        stage1a_phase(0, wan_interface_step=5, wan_full_step=4)


def test_stage1b_plan_is_exactly_7_random_prefixes_and_1_p47_full():
    config = load_training_bundle("configs/train/stage1b.yaml")["training"]
    assert validate_stage_objective(config) == "nested_spectral_hrepa_full_anchor"
    tasks = sample_microbatch_tasks("stage1b", 8, optimizer_step=0)
    kinds = [task.kind for task in tasks]
    prefix_tasks = [task for task in tasks if not task.is_full]

    assert kinds.count("full") == 1
    assert kinds.count("single_prefix") == 7
    assert kinds.count("paired_prefix") == 0
    assert len(prefix_tasks) == 7
    assert all(0 <= task.endpoint <= 46 for task in prefix_tasks)
    assert all(task.previous_endpoint is None for task in tasks)


def test_stage1b_random_prefixes_are_approximately_uniform():
    rng_state = random.getstate()
    random.seed(20260811)
    counts = [0] * 47
    try:
        for step in range(4700):
            for task in sample_microbatch_tasks("stage1b", 8, optimizer_step=step):
                if not task.is_full:
                    counts[task.endpoint] += 1
    finally:
        random.setstate(rng_state)

    expected = sum(counts) / len(counts)
    assert expected == pytest.approx(700.0)
    assert max(abs(count - expected) for count in counts) < expected * 0.15


def test_stage1b_resume_rejects_schedule_contract_change():
    training = load_training_bundle("configs/train/stage1b.yaml")["training"]
    saved = dict(training)
    saved["p47_full_microbatches_per_step"] = 0

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


def test_stage2_rejects_stage1b_decoder_policy():
    config = load_training_bundle("configs/train/stage2a.yaml")["training"]
    config["decoder_trainable_policy"] = "temporal_interface"
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
    assert model.decoder.training

    configure_stage(model, "stage1a", optimizer_step=2000)
    assert model.decoder.pre_decoder.weight.requires_grad
    assert model.decoder.decoder.conv1.weight.requires_grad
    assert model.decoder.decoder.time_conv.weight.requires_grad
    assert not model.decoder.decoder.spatial.weight.requires_grad

    configure_stage(model, "stage1a", optimizer_step=5000)
    assert model.decoder.decoder.spatial.weight.requires_grad
    assert not model.projector.shared_mask_set.requires_grad


def test_stage1b_trains_projector_and_temporal_interface_only():
    model = _DummyModel()

    configure_stage(model, "stage1b")

    assert model.projector.main.weight.requires_grad
    assert model.projector.shared_mask_set.requires_grad
    assert not model.repa_projection.weight.requires_grad
    assert model.decoder.training
    assert model.decoder.temporal_adapter.weight.requires_grad
    assert model.decoder.pre_decoder.weight.requires_grad
    assert model.decoder.decoder.conv1.weight.requires_grad
    assert model.decoder.decoder.time_conv.weight.requires_grad
    assert not model.decoder.decoder.spatial.weight.requires_grad


def test_stage1b_optimizer_uses_projector_and_temporal_interface_learning_rates():
    model = _DummyModel()
    configure_stage(model, "stage1b")
    training = load_training_bundle("configs/train/stage1b.yaml")["training"]
    training["fused_optimizer"] = False

    generator, _ = build_optimizers(model, nn.Linear(1, 1), training)
    groups = {group["name"]: group for group in generator.param_groups}
    fast_ids = {id(parameter) for parameter in groups["rae_fast"]["params"]}
    temporal_ids = {id(parameter) for parameter in groups["wan_temporal"]["params"]}

    assert groups["rae_fast"]["lr"] == pytest.approx(1e-4)
    assert groups["wan_temporal"]["lr"] == pytest.approx(2e-5)
    assert id(model.projector.main.weight) in fast_ids
    assert id(model.projector.shared_mask_set) in fast_ids
    assert id(model.decoder.temporal_adapter.weight) in temporal_ids
    assert id(model.decoder.pre_decoder.weight) in temporal_ids
    assert id(model.decoder.decoder.conv1.weight) in temporal_ids
    assert id(model.decoder.decoder.time_conv.weight) in temporal_ids
    assert id(model.decoder.temporal_adapter.weight) not in fast_ids
    assert all(not p.requires_grad for p in groups["wan_spatial"]["params"])


def test_adversarial_factor_delays_and_ramps_linearly():
    assert adversarial_factor(4999, 5000, 2000) == 0.0
    assert adversarial_factor(6000, 5000, 2000) == pytest.approx(0.5)
    assert adversarial_factor(7000, 5000, 2000) == 1.0
    assert adversarial_factor(0, 0, 1000) == 0.0
    assert adversarial_factor(500, 0, 1000) == pytest.approx(0.5)
    assert adversarial_factor(1000, 0, 1000) == 1.0
