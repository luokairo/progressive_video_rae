import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_dct")

from progressive_videorae.model.dct import (
    dct_lowpass_target,
    frequency_cutoff,
)


def test_endpoint_47_is_full_frequency_roundtrip():
    video = torch.randn(1, 3, 2, 32, 48)
    reconstructed = dct_lowpass_target(video, 47)
    torch.testing.assert_close(reconstructed, video, atol=1e-4, rtol=1e-4)


def test_half_cosine_frequency_support_is_monotonic():
    cutoffs = [frequency_cutoff(endpoint) for endpoint in range(48)]
    assert all(a.height <= b.height and a.width <= b.width for a, b in zip(cutoffs, cutoffs[1:]))
    assert (cutoffs[-1].height, cutoffs[-1].width) == (480, 768)

