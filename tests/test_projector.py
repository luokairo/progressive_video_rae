import pytest

torch = pytest.importorskip("torch")

from progressive_video_rae.model.projector import CausalFrequencyProjector


def test_future_set_perturbation_cannot_change_earlier_output():
    torch.manual_seed(7)
    projector = CausalFrequencyProjector(
        input_dim=8,
        hidden_dim=16,
        output_dim=4,
        num_frames=4,
        input_frames=2,
        height=2,
        width=4,
        depth=1,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        layout_version="test",
        set_sizes=(3, 2, 2, 1),
    ).eval()
    features = torch.randn(1, 2, 2, 4, 8)
    high_mask = projector.set_ids == 3
    perturbed = features.clone()
    perturbed[:, :, high_mask, :] += 100.0
    first = projector(features, prefix_len=4).metadata["unmasked_tokens"]
    second = projector(perturbed, prefix_len=4).metadata["unmasked_tokens"]
    low_mask = projector.set_ids < 3
    torch.testing.assert_close(first[:, :, low_mask], second[:, :, low_mask], atol=1e-5, rtol=1e-5)


def test_prefix_uses_one_shared_mask_token():
    projector = CausalFrequencyProjector(
        input_dim=8,
        hidden_dim=16,
        output_dim=4,
        num_frames=4,
        input_frames=2,
        height=2,
        width=4,
        depth=1,
        num_heads=4,
        set_sizes=(3, 2, 2, 1),
        layout_version="test",
    ).eval()
    state = projector(torch.randn(1, 2, 2, 4, 8), prefix_len=2)
    hidden = state.tokens[:, :, projector.set_ids >= 2]
    expected = projector.mask_token.reshape(1, 1, 1, 4).expand_as(hidden)
    torch.testing.assert_close(hidden, expected)

