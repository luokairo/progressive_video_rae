import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_dct")

from progressive_videorae.model.dct import dct_lowpass_target, frequency_cutoff


def test_prefix_64_is_full_frequency_roundtrip():
    video = torch.randn(1, 3, 2, 32, 48)
    reconstructed = dct_lowpass_target(video, 64)
    torch.testing.assert_close(reconstructed, video, atol=1e-4, rtol=1e-4)


def test_frequency_support_is_monotonic():
    cutoffs = [frequency_cutoff(prefix) for prefix in range(1, 65)]
    assert all(a.height <= b.height and a.width <= b.width for a, b in zip(cutoffs, cutoffs[1:]))
    assert cutoffs[-1].height == 480
    assert cutoffs[-1].width == 768

