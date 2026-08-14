from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from progressive_videorae.model.model import ProgressiveVideoRAEOutput
from progressive_videorae.model.types import EncoderOutput, RepaReference
from progressive_videorae.training import losses as losses_module


class ConstantLPIPS(nn.Module):
    def forward(self, prediction, target, chunk_size=None):
        del chunk_size
        return prediction.new_tensor(2.0)


class ConstantDiscriminator(nn.Module):
    def forward(self, video):
        return video.new_ones(video.shape[0], 1, 1, 1)


def build_output(*, reconstruction, target, repa_features=None, reference=None):
    if reference is None:
        reference = RepaReference(
            anchor=torch.ones(1, 1, 1, 1, 2),
            video_phases=torch.empty(1, 0, 2, 1, 1, 2),
        )
    return ProgressiveVideoRAEOutput(
        reconstruction=reconstruction,
        target=target,
        state=None,
        state_view=None,
        encoder_output=EncoderOutput(tokens=reference.anchor, grid_size=(1, 1, 1)),
        decoder_output=None,
        repa_features=repa_features,
        repa_reference=reference,
    )


def test_prefix_loss_uses_configured_weights(monkeypatch):
    monkeypatch.setattr(losses_module, "FrozenLPIPS", ConstantLPIPS)
    losses = losses_module.ProgressiveLosses(
        l1_weight=2.0,
        prefix_lpips_weight=3.0,
        lpips_weight=99.0,
        band_weight=0.0,
        leakage_weight=0.0,
        paired_delta_weight=0.0,
    )
    prediction = torch.zeros(1, 3, 1, 2, 2)
    target = torch.ones_like(prediction)

    result = losses.prefix(
        build_output(reconstruction=prediction, target=target),
        target,
        endpoint=47,
        previous_prediction=torch.zeros_like(prediction),
    )

    assert result.total.item() == pytest.approx(8.0)
    assert result.terms["l1"].item() == pytest.approx(1.0)
    assert result.terms["lpips"].item() == pytest.approx(2.0)
    assert result.weighted_terms["l1"].item() == pytest.approx(2.0)
    assert result.weighted_terms["lpips"].item() == pytest.approx(6.0)
    assert sum(result.weighted_terms.values()).item() == pytest.approx(
        result.total.item()
    )


def test_full_loss_computes_repa_and_uses_all_weights(monkeypatch):
    monkeypatch.setattr(losses_module, "FrozenLPIPS", ConstantLPIPS)
    losses = losses_module.ProgressiveLosses(
        l1_weight=2.0,
        lpips_weight=3.0,
        repa_local_weight=4.0,
        repa_global_weight=5.0,
        adversarial_weight=6.0,
    )
    prediction = torch.zeros(1, 3, 1, 2, 2)
    target = torch.ones_like(prediction)
    reference = RepaReference(
        anchor=torch.ones(1, 1, 1, 1, 2),
        video_phases=torch.ones(1, 1, 2, 1, 1, 2),
    )
    repa = RepaReference(
        anchor=torch.zeros_like(reference.anchor),
        video_phases=torch.zeros_like(reference.video_phases),
    )
    output = build_output(
        reconstruction=prediction,
        target=target,
        repa_features=repa,
        reference=reference,
    )

    result = losses.full_generator(output, ConstantDiscriminator())

    assert result.terms["repa_local"].item() == pytest.approx(1.0)
    assert result.terms["repa_global"].item() == pytest.approx(1.0)
    assert result.terms["adversarial"].item() == pytest.approx(-1.0)
    assert result.weighted_terms["adversarial"].item() == pytest.approx(-6.0)
    assert sum(result.weighted_terms.values()).item() == pytest.approx(
        result.total.item()
    )
    assert result.total.item() == pytest.approx(11.0)


class FailingDiscriminator(nn.Module):
    def forward(self, _video):
        raise AssertionError("discriminator should not run while GAN weight is zero")


def test_temporal_l1_matches_first_order_differences_and_backpropagates():
    prediction = torch.tensor([0.0, 1.0, 3.0]).reshape(1, 1, 3, 1, 1)
    prediction.requires_grad_(True)
    target = torch.tensor([0.0, 2.0, 2.0]).reshape_as(prediction)

    value = losses_module.temporal_l1(prediction, target)

    assert value.item() == pytest.approx(1.5)
    value.backward()
    assert prediction.grad is not None


def test_temporal_l1_is_differentiable_zero_for_single_frame():
    prediction = torch.ones(1, 3, 1, 2, 2, requires_grad=True)
    value = losses_module.temporal_l1(prediction, torch.zeros_like(prediction))

    assert value.item() == 0.0
    value.backward()
    assert prediction.grad is not None


def test_prefix_loss_uses_temporal_l1_weight(monkeypatch):
    monkeypatch.setattr(losses_module, "FrozenLPIPS", ConstantLPIPS)
    losses = losses_module.ProgressiveLosses(
        l1_weight=0.0,
        prefix_lpips_weight=0.0,
        temporal_l1_weight=2.0,
        band_weight=0.0,
        leakage_weight=0.0,
        paired_delta_weight=0.0,
    )
    prediction = torch.tensor([0.0, 1.0, 3.0]).reshape(1, 1, 3, 1, 1)
    target = torch.tensor([0.0, 2.0, 2.0]).reshape_as(prediction)

    result = losses.prefix(
        build_output(reconstruction=prediction, target=target),
        target,
        endpoint=47,
    )

    assert result.terms["temporal_l1"].item() == pytest.approx(1.5)
    assert result.total.item() == pytest.approx(3.0)


def test_zero_adversarial_factor_skips_discriminator(monkeypatch):
    monkeypatch.setattr(losses_module, "FrozenLPIPS", ConstantLPIPS)
    losses = losses_module.ProgressiveLosses(
        l1_weight=0.0,
        lpips_weight=0.0,
        repa_local_weight=0.0,
        repa_global_weight=0.0,
        adversarial_weight=1.0,
    )
    prediction = torch.zeros(1, 3, 1, 2, 2)
    output = build_output(
        reconstruction=prediction,
        target=torch.ones_like(prediction),
        repa_features=RepaReference(
            anchor=torch.ones(1, 1, 1, 1, 2),
            video_phases=torch.empty(1, 0, 2, 1, 1, 2),
        ),
    )

    result = losses.full_generator(
        output, FailingDiscriminator(), adversarial_factor=0.0
    )

    assert result.terms["adversarial"].item() == 0.0
    assert result.total.item() == 0.0


def test_frozen_repa_head_still_backpropagates_to_decoder_features(monkeypatch):
    from progressive_videorae.model.model import PhaseSpecificRepaProjection

    monkeypatch.setattr(losses_module, "FrozenLPIPS", ConstantLPIPS)
    losses = losses_module.ProgressiveLosses(
        l1_weight=0.0,
        lpips_weight=0.0,
        temporal_l1_weight=0.0,
        repa_local_weight=1.0,
        repa_global_weight=1.0,
        adversarial_weight=0.0,
    )
    upstream = nn.Conv3d(4, 8, 1)
    repa = PhaseSpecificRepaProjection(8, 6).requires_grad_(False)
    features = upstream(torch.randn(1, 4, 2, 3, 4))
    predicted = repa(features)
    reference = RepaReference(
        anchor=torch.randn_like(predicted.anchor),
        video_phases=torch.randn_like(predicted.video_phases),
    )
    output = build_output(
        reconstruction=torch.zeros(1, 3, 5, 4, 4, requires_grad=True),
        target=torch.zeros(1, 3, 5, 4, 4),
        repa_features=predicted,
        reference=reference,
    )

    result = losses.full_generator(
        output, FailingDiscriminator(), adversarial_factor=0.0
    )
    result.total.backward()

    assert all(parameter.grad is None for parameter in repa.parameters())
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in upstream.parameters()
    )


class PairwiseScore(nn.Module):
    def forward(self, prediction, target):
        return (prediction - target).square().mean(dim=(1, 2, 3), keepdim=True)


def test_frozen_lpips_checkpoint_preserves_value_and_prediction_gradient():
    checkpointed = losses_module.FrozenLPIPS.__new__(losses_module.FrozenLPIPS)
    nn.Module.__init__(checkpointed)
    checkpointed.model = PairwiseScore().requires_grad_(False)
    checkpointed.train()

    prediction = torch.randn(1, 3, 5, 8, 8, requires_grad=True)
    target = torch.randn_like(prediction)
    actual = checkpointed(prediction, target, chunk_size=2)
    actual_gradient = torch.autograd.grad(actual, prediction)[0]

    reference_prediction = prediction.detach().clone().requires_grad_(True)
    pred_frames = reference_prediction.permute(0, 2, 1, 3, 4).reshape(5, 3, 8, 8)
    target_frames = target.permute(0, 2, 1, 3, 4).reshape(5, 3, 8, 8)
    reference = torch.cat(
        [
            checkpointed.model(pred_frames[start : start + 2], target_frames[start : start + 2])
            for start in range(0, 5, 2)
        ]
    ).mean()
    reference_gradient = torch.autograd.grad(reference, reference_prediction)[0]

    torch.testing.assert_close(actual, reference, atol=0.0, rtol=0.0)
    torch.testing.assert_close(actual_gradient, reference_gradient, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    ("endpoint", "expected_level", "expected_grid", "expected_scale"),
    (
        (0, 0, (1, 1), 0.0),
        (7, 0, (1, 1), 0.0),
        (8, 1, (2, 3), 0.2),
        (15, 1, (2, 3), 0.2),
        (16, 2, (4, 6), 0.4),
        (23, 2, (4, 6), 0.4),
        (24, 3, (8, 12), 0.6),
        (31, 3, (8, 12), 0.6),
        (32, 4, (15, 24), 0.8),
        (39, 4, (15, 24), 0.8),
        (40, 5, (30, 48), 1.0),
        (46, 5, (30, 48), 1.0),
    ),
)
def test_hierarchical_repa_level_boundaries(
    endpoint, expected_level, expected_grid, expected_scale
):
    level, grid, scale = losses_module.hierarchical_repa_level(endpoint)

    assert level == expected_level
    assert grid == expected_grid
    assert scale == pytest.approx(expected_scale)


def test_hierarchical_repa_rejects_p47_full_anchor():
    with pytest.raises(ValueError, match=r"\[0,46\]"):
        losses_module.hierarchical_repa_level(47)


def test_hierarchical_repa_keeps_groups_and_phases_separate():
    anchor = torch.tensor([1.0, 0.0]).reshape(1, 1, 1, 1, 2).expand(1, 1, 4, 6, 2)
    phase_vectors = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0]], [[-1.0, 0.0], [0.0, -1.0]]]]
    ).reshape(1, 2, 2, 1, 1, 2)
    predicted_phases = phase_vectors.expand(1, 2, 2, 4, 6, 2)
    target_phases = (
        torch.tensor([1.0, 0.0])
        .reshape(1, 1, 1, 1, 1, 2)
        .expand_as(predicted_phases)
    )

    terms = losses_module.hierarchical_repa_terms(
        RepaReference(anchor=anchor, video_phases=predicted_phases),
        RepaReference(anchor=anchor, video_phases=target_phases),
        endpoint=16,
    )

    expected = torch.tensor([[0.0, 1.0], [2.0, 1.0]])
    torch.testing.assert_close(terms.group_phase_global_errors, expected)
    torch.testing.assert_close(terms.group_phase_local_errors, expected)


def test_prefix_loss_adds_hierarchical_repa_and_reports_pyramid_statistics(
    monkeypatch,
):
    monkeypatch.setattr(losses_module, "FrozenLPIPS", ConstantLPIPS)

    def frequency_terms_without_dct(prediction, full_target, endpoint):
        assert endpoint == 16
        coefficients = torch.zeros_like(prediction)
        return SimpleNamespace(
            target=full_target,
            target_coefficients=coefficients,
            prediction_coefficients=coefficients,
            band_mask=torch.zeros_like(coefficients, dtype=torch.bool),
            leakage=prediction.sum() * 0.0,
        )

    monkeypatch.setattr(
        losses_module, "progressive_frequency_terms", frequency_terms_without_dct
    )
    losses = losses_module.ProgressiveLosses(
        l1_weight=0.0,
        prefix_lpips_weight=0.0,
        temporal_l1_weight=0.0,
        band_weight=0.0,
        leakage_weight=0.0,
        paired_delta_weight=0.0,
        prefix_repa_global_weight=1.0,
        prefix_repa_local_weight=1.0,
    )
    predicted_anchor = torch.randn(1, 1, 4, 6, 3, requires_grad=True)
    predicted_phases = torch.randn(1, 2, 2, 4, 6, 3, requires_grad=True)
    teacher_anchor = torch.randn_like(predicted_anchor, requires_grad=True)
    teacher_phases = torch.randn_like(predicted_phases, requires_grad=True)
    reconstruction = torch.zeros(1, 3, 1, 2, 2, requires_grad=True)
    target = torch.zeros_like(reconstruction)
    result = losses.prefix(
        build_output(
            reconstruction=reconstruction,
            target=target,
            repa_features=RepaReference(predicted_anchor, predicted_phases),
            reference=RepaReference(teacher_anchor, teacher_phases),
        ),
        target,
        endpoint=16,
    )

    expected = result.terms["prefix_repa_global"] + 0.4 * result.terms[
        "prefix_repa_local"
    ]
    torch.testing.assert_close(result.total, expected)
    assert result.statistics["prefix_repa/level"].item() == 2
    assert result.statistics["prefix_repa/grid_height"].item() == 4
    assert result.statistics["prefix_repa/grid_width"].item() == 6
    result.total.backward()
    assert predicted_anchor.grad is not None
    assert predicted_phases.grad is not None
    assert teacher_anchor.grad is None
    assert teacher_phases.grad is None


def test_prefix_hrepa_freezes_teacher_and_head_but_backpropagates_upstream():
    from progressive_videorae.model.model import PhaseSpecificRepaProjection

    upstream = nn.Conv3d(4, 8, 1)
    repa_head = PhaseSpecificRepaProjection(8, 6).requires_grad_(False)
    decoder_features = upstream(torch.randn(1, 4, 3, 8, 12))
    prediction = repa_head(decoder_features)
    teacher_anchor = torch.randn_like(prediction.anchor, requires_grad=True)
    teacher_phases = torch.randn_like(prediction.video_phases, requires_grad=True)

    terms = losses_module.hierarchical_repa_terms(
        prediction,
        RepaReference(anchor=teacher_anchor, video_phases=teacher_phases),
        endpoint=24,
    )
    (terms.global_loss + terms.local_scale * terms.local_loss).backward()

    assert all(parameter.grad is None for parameter in repa_head.parameters())
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in upstream.parameters()
    )
    assert teacher_anchor.grad is None
    assert teacher_phases.grad is None


class CountingDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, video):
        self.calls += 1
        return video.mean(dim=(1, 2, 3, 4), keepdim=True) + 0.25


def test_p47_gan_ramp_scales_generator_and_discriminator_with_three_forwards(monkeypatch):
    monkeypatch.setattr(losses_module, "FrozenLPIPS", ConstantLPIPS)
    losses = losses_module.ProgressiveLosses(
        l1_weight=0.0,
        lpips_weight=0.0,
        repa_local_weight=0.0,
        repa_global_weight=0.0,
        adversarial_weight=0.05,
    )
    prediction = torch.zeros(1, 3, 1, 2, 2)
    target = torch.ones_like(prediction)
    reference = RepaReference(
        anchor=torch.ones(1, 1, 1, 1, 2),
        video_phases=torch.empty(1, 0, 2, 1, 1, 2),
    )
    output = build_output(
        reconstruction=prediction,
        target=target,
        repa_features=reference,
        reference=reference,
    )
    discriminator = CountingDiscriminator()

    disc_loss = losses.discriminator(
        discriminator,
        prediction,
        target,
        adversarial_factor=0.5,
    )
    generator_loss = losses.full_generator(
        output,
        discriminator,
        adversarial_factor=0.5,
    )

    assert discriminator.calls == 3
    assert disc_loss.terms["disc_total"].item() == pytest.approx(0.625)
    assert disc_loss.total.item() == pytest.approx(0.3125)
    assert generator_loss.terms["adversarial"].item() == pytest.approx(-0.25)
    assert generator_loss.weighted_terms["adversarial"].item() == pytest.approx(-0.00625)
