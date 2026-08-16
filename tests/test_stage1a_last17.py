from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from progressive_videorae.config import load_training_bundle
from progressive_videorae.model.model import ProgressiveVideoRAEOutput
from progressive_videorae.model.projector import (
    CausalFrequencyProjector,
    IdentityInitializedTemporalAttentionPool,
)
from progressive_videorae.model.types import (
    EncoderOutput,
    PrefixGroupFeatures,
    RepaReference,
    StateContract,
)
from progressive_videorae.train import (
    build_optimizers,
    clip_optimizer_gradients,
    cosine_scheduler,
)
from progressive_videorae.training import losses as losses_module
from progressive_videorae.training.checkpoint import representation_identity
from progressive_videorae.training.stages import (
    adversarial_factor,
    configure_stage,
    repa_factor,
    stage1a_phase,
)


def make_k23_projector() -> CausalFrequencyProjector:
    return CausalFrequencyProjector(
        input_dim=8,
        hidden_dim=16,
        output_dim=48,
        num_input_layers=23,
        depth=1,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        layer_fusion="fixed_sum",
        layer_fusion_norm="non_affine_layer_norm",
        temporal_pooling="input_dim_attention",
        temporal_pooling_heads=4,
    )


def test_fixed_sum_mls_uses_all_layers_without_trainable_mix_logits():
    projector = make_k23_projector()
    layers = tuple(
        torch.randn(1, 2, 2, 3, 8) + layer_index
        for layer_index in range(23)
    )
    group = PrefixGroupFeatures(
        tokens=layers[-1],
        layer_tokens=layers,
        latent_type="video_group",
        source_start=0,
        source_end=4,
        input_frames=6,
    )

    actual = projector._mix_layers(group)
    expected = torch.nn.functional.layer_norm(
        torch.stack(layers).sum(dim=0),
        (8,),
    )

    assert not hasattr(projector, "layer_mix_logits")
    torch.testing.assert_close(actual, expected)
    reduced, _ = projector._reduce_group(group, actual)
    assert reduced.shape == (1, 2, 3, 16)


def test_input_dim_temporal_pool_starts_as_normalized_mean_and_backpropagates():
    pool = IdentityInitializedTemporalAttentionPool(
        8,
        group_size=2,
        num_heads=4,
    )
    values = torch.randn(2, 2, 3, 4, 8, requires_grad=True)

    actual, attention = pool(values, return_attention=True)
    expected = torch.nn.functional.layer_norm(values.mean(dim=1), (8,))

    torch.testing.assert_close(attention, torch.full_like(attention, 0.5))
    torch.testing.assert_close(actual, expected, atol=1.0e-6, rtol=1.0e-5)
    actual.square().mean().backward()
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()


def test_stage1a_formal_config_uses_default_k23_h512_model():
    bundle = load_training_bundle("configs/train/stage1a_recon_last17_12k.yaml")
    training = bundle["training"]
    model = bundle["model"]

    assert training["max_steps"] == 12000
    assert training["wan_interface_step"] == 0
    assert training["wan_full_step"] == 200
    assert training["adversarial_weight"] == 0.0
    assert model["encoder"]["output_layers"] == list(range(2, 25))
    assert model["projector"]["hidden_dim"] == 512
    assert model["projector"]["layer_fusion"] == "fixed_sum"
    assert model["projector"]["temporal_pooling"] == "input_dim_attention"
    assert model["state"]["channels"] == 48
    assert model["state"]["num_sets"] == 48
    assert model["state"]["tokens_per_set"] == 30


def test_new_projector_modes_are_bound_into_representation_identity():
    contract = StateContract()
    model = SimpleNamespace(
        encoder=SimpleNamespace(
            variant="vitl",
            output_layers=tuple(range(2, 25)),
            load_report=SimpleNamespace(checkpoint_path="/encoder.pt"),
        ),
        projector=SimpleNamespace(
            layout_checksum=contract.layout_checksum,
            layout_version=contract.layout_version,
            layer_fusion="fixed_sum",
            layer_fusion_norm_mode="non_affine_layer_norm",
            temporal_pooling="input_dim_attention",
            temporal_pooling_heads=16,
        ),
        decoder=SimpleNamespace(codec_id="codec-v4", decoder_id="decoder-v4"),
    )

    identity = representation_identity(model)

    assert identity["selected_vjepa_layers"] == list(range(2, 25))
    assert identity["layer_fusion"] == "fixed_sum"
    assert identity["layer_fusion_norm"] == "non_affine_layer_norm"
    assert identity["temporal_pooling"] == "input_dim_attention"
    assert identity["temporal_pooling_heads"] == 16


def test_stage1a_last17_phase_and_repa_boundaries_are_exact():
    training = load_training_bundle(
        "configs/train/stage1a_recon_k23_h512_24fps_12k.yaml"
    )["training"]
    assert stage1a_phase(0, wan_interface_step=0, wan_full_step=200) == "interface"
    assert stage1a_phase(199, wan_interface_step=0, wan_full_step=200) == "interface"
    assert stage1a_phase(200, wan_interface_step=0, wan_full_step=200) == "full"

    repa_args = (
        training["repa_start_step"],
        training["repa_ramp_steps"],
        training["repa_max_factor"],
    )
    assert repa_factor(999, *repa_args) == 0.0
    assert repa_factor(1000, *repa_args) == 0.0
    assert repa_factor(1750, *repa_args) == pytest.approx(0.25)
    assert repa_factor(2500, *repa_args) == pytest.approx(0.5)


def test_stage1a_formal_schedule_never_activates_gan():
    training = load_training_bundle(
        "configs/train/stage1a_recon_k23_h512_24fps_12k.yaml"
    )["training"]

    assert training["adversarial_weight"] == 0.0
    assert training["disc_start"] == training["max_steps"] == 12000
    assert training["adversarial_ramp_steps"] == 0
    assert all(
        adversarial_factor(
            step,
            training["disc_start"],
            training["adversarial_ramp_steps"],
        )
        == 0.0
        for step in range(training["max_steps"])
    )


def test_cosine_scheduler_reaches_base_lr_and_ten_percent_floor():
    parameter = nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = cosine_scheduler(
        optimizer,
        warmup_steps=500,
        total_steps=12000,
        min_lr_ratio=0.1,
    )
    scale = scheduler.lr_lambdas[0]

    assert scale(499) == pytest.approx(1.0)
    assert scale(500) == pytest.approx(1.0)
    assert scale(12000) == pytest.approx(0.1)


class DummyWanCore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Linear(1, 1)
        self.time_conv = nn.Linear(1, 1)
        self.spatial = nn.Linear(1, 1)


class DummyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.temporal_adapter = nn.Linear(1, 1)
        self.pre_decoder = nn.Linear(1, 1)
        self.decoder = DummyWanCore()


class DummyProjector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.main = nn.Linear(1, 1)
        self.shared_mask_set = nn.Parameter(torch.zeros(1))


class DummyTrainingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(1, 1)
        self.projector = DummyProjector()
        self.repa_projection = nn.Linear(1, 1)
        self.decoder = DummyDecoder()


def test_optimizer_groups_apply_update_hierarchy_and_independent_clipping():
    model = DummyTrainingModel()
    configure_stage(
        model,
        "stage1a",
        optimizer_step=0,
        wan_interface_step=0,
        wan_full_step=200,
    )
    training = load_training_bundle(
        "configs/train/stage1a_recon_last17_12k.yaml"
    )["training"]
    training["fused_optimizer"] = False
    generator, _ = build_optimizers(model, nn.Linear(1, 1), training)
    groups = {group["name"]: group for group in generator.param_groups}

    assert groups["rae_fast"]["weight_decay"] == pytest.approx(0.01)
    assert groups["wan_temporal"]["weight_decay"] == 0.0
    assert groups["wan_spatial"]["weight_decay"] == 0.0
    assert groups["rae_fast"]["max_grad_norm"] == 5.0
    assert groups["wan_temporal"]["max_grad_norm"] == 5.0
    assert groups["wan_spatial"]["max_grad_norm"] == 10.0

    first = nn.Parameter(torch.zeros(2))
    second = nn.Parameter(torch.zeros(2))
    first.grad = torch.tensor([3.0, 4.0])
    second.grad = torch.tensor([6.0, 8.0])
    optimizer = torch.optim.SGD(
        [
            {"params": [first], "name": "first", "max_grad_norm": 1.0},
            {"params": [second], "name": "second", "max_grad_norm": 20.0},
        ],
        lr=1.0,
    )
    metrics = clip_optimizer_gradients(optimizer, default_max_norm=1.0)

    assert first.grad.norm().item() == pytest.approx(1.0)
    assert second.grad.norm().item() == pytest.approx(10.0)
    assert metrics["grad_clip_scale/first"] == pytest.approx(0.2)
    assert metrics["grad_clip_scale/second"] == 1.0


class ZeroLPIPS(nn.Module):
    def forward(self, prediction, target, chunk_size=None):
        del target, chunk_size
        return prediction.sum() * 0.0


class FailingDiscriminator(nn.Module):
    def forward(self, _video):
        raise AssertionError("GAN-disabled loss must not call the discriminator")


def test_full_loss_applies_dynamic_repa_factor(monkeypatch):
    monkeypatch.setattr(losses_module, "FrozenLPIPS", ZeroLPIPS)
    losses = losses_module.ProgressiveLosses(
        l1_weight=0.0,
        lpips_weight=0.0,
        temporal_l1_weight=0.0,
        repa_local_weight=1.0,
        repa_global_weight=1.0,
        adversarial_weight=0.0,
    )
    reference = RepaReference(
        anchor=torch.ones(1, 1, 1, 1, 2),
        video_phases=torch.empty(1, 0, 2, 1, 1, 2),
    )
    prediction = RepaReference(
        anchor=torch.zeros_like(reference.anchor),
        video_phases=torch.empty_like(reference.video_phases),
    )
    reconstruction = torch.zeros(1, 3, 1, 2, 2)
    output = ProgressiveVideoRAEOutput(
        reconstruction=reconstruction,
        target=reconstruction.clone(),
        state=None,
        state_view=None,
        encoder_output=EncoderOutput(tokens=reference.anchor, grid_size=(1, 1, 1)),
        decoder_output=None,
        repa_features=prediction,
        repa_reference=reference,
    )

    result = losses.full_generator(
        output,
        FailingDiscriminator(),
        adversarial_factor=0.0,
        repa_factor=0.25,
    )

    assert result.terms["repa_local"].item() == pytest.approx(1.0)
    assert result.terms["repa_global"].item() == pytest.approx(1.0)
    assert result.total.item() == pytest.approx(0.5)
