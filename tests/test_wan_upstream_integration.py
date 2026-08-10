from pathlib import Path
import os

import pytest

torch = pytest.importorskip("torch")

from progressive_videorae.model.wan_decoder import WanVideoDecoder


@pytest.mark.integration
def test_real_wan_decoder_cache_matches_full_sequence_without_weights():
    if os.environ.get("PVR_RUN_LARGE_WEIGHT_LOAD_TESTS") != "1":
        pytest.skip("Set PVR_RUN_LARGE_WEIGHT_LOAD_TESTS=1 for the 480p Wan integration test")
    source_root = Path("/share/project/lgy/Wan2.2")
    if not (source_root / "wan/modules/vae2_2.py").is_file():
        pytest.skip("Pinned local Wan2.2 source is unavailable")
    model = WanVideoDecoder(
        checkpoint_path=None,
        source_root=str(source_root),
        latent_channels=48,
        base_dim=8,
        output_size=(480, 768),
        load_pretrained=False,
    ).eval()
    state = torch.randn(1, 3, 48, 30, 48)
    with torch.no_grad():
        full = model.decode(state, cache_mode="disabled").video
        cached = model.decode(state, cache_mode="reset").video
    assert tuple(full.shape) == (1, 3, 9, 480, 768)
    torch.testing.assert_close(full, cached, atol=1e-5, rtol=1e-5)
