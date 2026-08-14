import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_dct")

from progressive_videorae.model.dct import (
    dct_band_coefficients,
    dct_lowpass_target,
    frequency_leakage,
    frequency_cutoff,
    progressive_frequency_terms,
)


def test_endpoint_47_is_full_frequency_roundtrip():
    video = torch.randn(1, 3, 2, 32, 48)
    reconstructed = dct_lowpass_target(video, 47)
    torch.testing.assert_close(reconstructed, video, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("endpoint", [0, 17, 47])
def test_fused_progressive_frequency_terms_match_individual_helpers(endpoint):
    prediction = torch.randn(1, 3, 2, 16, 24, requires_grad=True)
    target = torch.randn_like(prediction)

    fused = progressive_frequency_terms(prediction, target, endpoint)
    expected_target = dct_lowpass_target(target, endpoint)
    expected_target_coefficients, expected_mask = dct_band_coefficients(target, endpoint)
    expected_prediction_coefficients, _ = dct_band_coefficients(prediction, endpoint)
    expected_leakage = frequency_leakage(prediction, endpoint)

    torch.testing.assert_close(fused.target, expected_target, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(fused.target_coefficients, expected_target_coefficients)
    torch.testing.assert_close(fused.prediction_coefficients, expected_prediction_coefficients)
    assert torch.equal(fused.band_mask, expected_mask)
    torch.testing.assert_close(fused.leakage, expected_leakage)
    if endpoint == 47:
        assert fused.leakage.item() == 0.0

    loss = fused.prediction_coefficients.abs().mean() + fused.leakage
    loss.backward()
    assert prediction.grad is not None


def test_half_cosine_frequency_support_is_monotonic():
    cutoffs = [frequency_cutoff(endpoint) for endpoint in range(48)]
    assert all(a.height <= b.height and a.width <= b.width for a, b in zip(cutoffs, cutoffs[1:]))
    assert (cutoffs[-1].height, cutoffs[-1].width) == (480, 768)

