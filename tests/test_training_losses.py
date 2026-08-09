from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from progressive_videorae.model.model import ProgressiveVideoRAEOutput
from progressive_videorae.model.types import EncoderOutput
from progressive_videorae.training import losses as losses_module


class ConstantLPIPS(nn.Module):
    def forward(self, prediction, target):
        return prediction.new_tensor(2.0)


class ConstantDiscriminator(nn.Module):
    def forward(self, video):
        return video.new_ones(video.shape[0], 1, 1, 1)


def build_output(*, reconstruction, target, repa_features=None, reference=None):
    if reference is None:
        reference = torch.ones(1, 1, 1, 1, 2)
    return ProgressiveVideoRAEOutput(
        reconstruction=reconstruction,
        target=target,
        state=None,
        encoder_output=EncoderOutput(tokens=reference, grid_size=(1, 1, 1)),
        decoder_output=None,
        repa_features=repa_features,
    )


def test_prefix_loss_uses_configured_weights(monkeypatch):
    monkeypatch.setattr(losses_module, "FrozenLPIPS", ConstantLPIPS)
    losses = losses_module.ProgressiveLosses(
        l1_weight=2.0,
        prefix_lpips_weight=3.0,
        lpips_weight=99.0,
    )
    prediction = torch.zeros(1, 3, 1, 2, 2)
    target = torch.ones_like(prediction)

    result = losses.prefix(build_output(reconstruction=prediction, target=target), target)

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
    reference = torch.ones(1, 1, 1, 1, 2)
    repa = torch.zeros_like(reference)
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
    )
    prediction = torch.tensor([0.0, 1.0, 3.0]).reshape(1, 1, 3, 1, 1)
    target = torch.tensor([0.0, 2.0, 2.0]).reshape_as(prediction)

    result = losses.prefix(build_output(reconstruction=prediction, target=target), target)

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
        repa_features=torch.ones(1, 1, 1, 1, 2),
    )

    result = losses.full_generator(
        output, FailingDiscriminator(), adversarial_factor=0.0
    )

    assert result.terms["adversarial"].item() == 0.0
    assert result.total.item() == 0.0
