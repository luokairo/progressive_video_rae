import pytest

torch = pytest.importorskip("torch")
from torch import nn

from progressive_videorae.model.wan_decoder import WanVideoDecoder


class FakeCausalDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsamples = nn.ModuleList([nn.Identity()])

    def forward(self, x, feat_cache=None, feat_idx=None, first_chunk=False):
        if feat_cache is None:
            return x.cumsum(dim=2)
        previous = 0.0 if feat_cache[0] is None else feat_cache[0]
        outputs = []
        for index in range(x.shape[2]):
            previous = previous + x[:, :, index : index + 1]
            outputs.append(previous)
        feat_cache[0] = previous.detach()
        feat_idx[0] += 1
        return torch.cat(outputs, dim=2)


def make_lightweight_adapter():
    adapter = WanVideoDecoder.__new__(WanVideoDecoder)
    nn.Module.__init__(adapter)
    adapter.pre_decoder = nn.Identity()
    adapter.decoder = FakeCausalDecoder()
    adapter.output_size = (2, 3)
    adapter.gradient_checkpointing = False
    adapter.register_buffer("latent_mean", torch.zeros(1, 48, 1, 1, 1))
    adapter.register_buffer("latent_std", torch.ones(1, 48, 1, 1, 1))
    adapter._count_conv3d = lambda _module: 1
    adapter._unpatchify = lambda value, patch_size: value
    return adapter


def test_cached_sequence_matches_full_sequence():
    adapter = make_lightweight_adapter()
    state = torch.randn(1, 4, 2, 3, 48)
    full = adapter.decode(state, cache_mode="disabled").video
    cached = adapter.decode(state, cache_mode="reset").video
    torch.testing.assert_close(full, cached)


def test_reuse_requires_explicit_state_and_reset_discards_history():
    adapter = make_lightweight_adapter()
    frame = torch.randn(1, 1, 2, 3, 48)
    with pytest.raises(ValueError):
        adapter.decode(frame, cache_mode="reuse")
    first = adapter.decode(frame, cache_mode="reset")
    reused = adapter.decode(frame, cache_mode="reuse", cache_state=first.cache_state)
    reset = adapter.decode(frame, cache_mode="reset")
    assert not torch.allclose(reused.video, reset.video)

