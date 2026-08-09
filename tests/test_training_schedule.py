from __future__ import annotations

import random

import pytest

from progressive_videorae.config import load_training_bundle
from progressive_videorae.train import (
    completed_discriminator_updates,
    discriminator_accumulation_boundary,
    discriminator_cycle_flags,
    validate_training_schedule,
)
from progressive_videorae.training.stages import (
    adversarial_factor,
    normalize_objective_weights,
    objective_loss_scales,
    sample_microbatch_prefixes,
)


def test_objective_weights_are_normalized_and_applied_per_task():
    assert normalize_objective_weights(2.0, 1.0) == pytest.approx((2 / 3, 1 / 3))

    prefixes = (7, 64, 23, 64, 41, 64, 63, 64)
    scales, effective = objective_loss_scales(prefixes, 2.0, 1.0)

    assert effective == pytest.approx((2 / 3, 1 / 3))
    assert sum(scale for scale, prefix in zip(scales, prefixes) if prefix == 64) == pytest.approx(
        2 / 3
    )
    assert sum(scale for scale, prefix in zip(scales, prefixes) if prefix != 64) == pytest.approx(
        1 / 3
    )


@pytest.mark.parametrize("weights", [(-1.0, 1.0), (1.0, -1.0), (0.0, 0.0)])
def test_objective_weights_reject_invalid_values(weights):
    with pytest.raises(ValueError):
        normalize_objective_weights(*weights)


def test_full_only_schedule_keeps_unit_loss_scale():
    scales, effective = objective_loss_scales((64,) * 8, 1.0, 1.0)

    assert scales == pytest.approx((0.125,) * 8)
    assert effective == pytest.approx((1.0, 0.0))


def test_mixed_accumulation_is_balanced_and_ends_with_full():
    random.seed(123)
    prefixes = sample_microbatch_prefixes(
        "stage1b",
        optimizer_step=0,
        accumulation_steps=8,
        minimum=1,
        maximum=63,
        schedule="mixed_accumulation",
        full_microbatches_per_step=4,
    )

    assert len(prefixes) == 8
    assert prefixes[-1] == 64
    assert prefixes[1::2] == (64, 64, 64, 64)
    assert all(1 <= prefix <= 63 for prefix in prefixes[0::2])


def test_mixed_accumulation_rejects_non_balanced_layout():
    with pytest.raises(ValueError, match="exactly half"):
        sample_microbatch_prefixes(
            "stage1b",
            optimizer_step=0,
            accumulation_steps=8,
            schedule="mixed_accumulation",
            full_microbatches_per_step=3,
        )


def test_adversarial_factor_delays_and_ramps_linearly():
    assert adversarial_factor(4999, 5000, 2000) == 0.0
    assert adversarial_factor(5000, 5000, 2000) == 0.0
    assert adversarial_factor(6000, 5000, 2000) == pytest.approx(0.5)
    assert adversarial_factor(7000, 5000, 2000) == 1.0
    assert adversarial_factor(9000, 5000, 2000) == 1.0


def test_stage_configs_have_expected_discriminator_update_counts():
    stage1a = load_training_bundle("configs/train/stage1a.yaml")["training"]
    stage1b = load_training_bundle("configs/train/stage1b.yaml")["training"]
    stage2a = load_training_bundle("configs/train/stage2a.yaml")["training"]

    assert validate_training_schedule(stage1a) == 5000
    assert validate_training_schedule(stage1b) == 45000
    assert validate_training_schedule(stage2a) == 50000
    assert completed_discriminator_updates(1000, stage1b) == 500


def test_discriminator_steps_only_after_two_mixed_generator_updates():
    flags = [
        discriminator_cycle_flags(step, 0, 2, has_full_microbatch=True)
        for step in range(4)
    ]

    assert flags == [
        (True, True, False),
        (True, False, True),
        (True, True, False),
        (True, False, True),
    ]
    assert not discriminator_accumulation_boundary(1, 0, 2)
    assert discriminator_accumulation_boundary(2, 0, 2)

