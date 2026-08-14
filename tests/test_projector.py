import pytest

torch = pytest.importorskip("torch")

from progressive_videorae.model.projector import CausalFrequencyProjector
from progressive_videorae.model.progressive_sets import LAYOUT_CHECKSUM, LAYOUT_VERSION
from progressive_videorae.model.types import (
    PrefixEncoderOutput,
    PrefixGroupFeatures,
    ProgressiveState,
    SpatialPrefixView,
    StateContract,
)


def make_features(tensor):
    group = PrefixGroupFeatures(
        tokens=tensor,
        layer_tokens=tuple(tensor.clone() for _ in range(5)),
        latent_type="image_first",
        source_start=0,
        source_end=0,
        input_frames=2,
    )
    return PrefixEncoderOutput(
        groups=(group,), spatial_grid=(30, 48), embed_dim=tensor.shape[-1], max_context_frames=64
    )


def make_projector():
    return CausalFrequencyProjector(
        input_dim=8,
        hidden_dim=16,
        output_dim=48,
        depth=1,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
    ).eval()


def test_future_set_perturbation_cannot_change_earlier_output():
    torch.manual_seed(7)
    projector = make_projector()
    tensor = torch.randn(1, 1, 30, 48, 8)
    perturbed = tensor.clone()
    perturbed[:, :, projector.set_ids == 47] += 100.0
    first = projector(make_features(tensor)).state.tokens
    second = projector(make_features(perturbed)).state.tokens
    torch.testing.assert_close(first[:, :, :47], second[:, :, :47], atol=1e-5, rtol=1e-5)


def test_prefix_uses_one_strictly_zero_initialized_shared_mask_token():
    projector = make_projector()
    state = projector(make_features(torch.randn(1, 1, 30, 48, 8))).state
    view = projector.make_prefix_view(state, 37)
    hidden = view.tokens[:, :, 38:]
    expected = projector.shared_mask_set.expand_as(hidden)
    torch.testing.assert_close(hidden, expected)
    assert torch.count_nonzero(projector.shared_mask_set) == 0


def test_only_masked_prefix_views_backpropagate_to_shared_mask():
    projector = make_projector().train()
    features = make_features(torch.randn(1, 1, 30, 48, 8))

    canonical = projector(features).state
    canonical.tokens.sum().backward()
    assert projector.shared_mask_set.grad is None

    projector.zero_grad(set_to_none=True)
    canonical = projector(features).state
    prefix = projector.make_prefix_view(canonical, 12)
    prefix.tokens[:, :, 13:].sum().backward()

    assert projector.shared_mask_set.grad is not None
    assert torch.count_nonzero(projector.shared_mask_set.grad) > 0


def test_full_state_and_phase_specific_repa_shapes():
    projector = make_projector()
    anchor = torch.randn(1, 1, 30, 48, 8)
    video = torch.randn(1, 2, 30, 48, 8)
    groups = (
        make_features(anchor).groups[0],
        PrefixGroupFeatures(
            tokens=video,
            layer_tokens=tuple(video.clone() for _ in range(5)),
            latent_type="video_group",
            source_start=0,
            source_end=4,
            input_frames=6,
        ),
    )
    output = projector(
        PrefixEncoderOutput(groups=groups, spatial_grid=(30, 48), embed_dim=8, max_context_frames=64)
    )
    assert output.state.tokens.shape == (1, 2, 48, 30, 48)
    assert output.repa_reference.anchor.shape == (1, 1, 30, 48, 8)
    assert output.repa_reference.video_phases.shape == (1, 1, 2, 30, 48, 8)


def test_candidate_v3_contract_binds_fixed_prefix_indexing_and_layout_checksum():
    projector = make_projector()
    state = projector(make_features(torch.randn(1, 1, 30, 48, 8))).state
    view = projector.make_prefix_view(state, 12)
    contract = projector.contract
    assert contract.prefix_indexing == "zero_based_inclusive_endpoint_v1"
    assert contract.layout_version == LAYOUT_VERSION
    assert contract.layout_checksum == LAYOUT_CHECKSUM
    for value in (state, view):
        assert value.layout_version == LAYOUT_VERSION
        assert value.layout_checksum == LAYOUT_CHECKSUM


def test_state_and_view_construction_reject_layout_identity_mismatch():
    tokens = torch.randn(1, 1, 48, 30, 48)
    contract = StateContract()
    with pytest.raises(ValueError, match="layout identity"):
        ProgressiveState(tokens, LAYOUT_VERSION, "tampered", contract=contract)
    source = ProgressiveState(tokens, LAYOUT_VERSION, LAYOUT_CHECKSUM, contract=contract)
    with pytest.raises(ValueError, match="layout identity"):
        SpatialPrefixView(tokens, 3, source=source, layout_checksum="tampered")
