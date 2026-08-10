import pytest

torch = pytest.importorskip("torch")
from torch import nn

from progressive_videorae.model.progressive_sets import build_progressive_layout
from progressive_videorae.model.types import StateContract
from progressive_videorae.model.wan_decoder import RAETemporalAdapter, WanVideoDecoder


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
    adapter.temporal_adapter = RAETemporalAdapter(48)
    adapter.output_size = (30, 48)
    adapter.gradient_checkpointing = False
    adapter.contract = StateContract()
    layout = build_progressive_layout()
    traversal = torch.tensor(layout.traversal)
    inverse = torch.empty_like(traversal)
    inverse[traversal] = torch.arange(traversal.numel())
    adapter.register_buffer("inverse_fps_permutation", inverse)
    adapter.codec_id = "candidate_v3_unfrozen"
    adapter.decoder_id = "wan22_native_r4_bridge_v1"
    adapter.register_buffer("latent_mean", torch.zeros(1, 48, 1, 1, 1))
    adapter.register_buffer("latent_std", torch.ones(1, 48, 1, 1, 1))
    adapter._count_conv3d = lambda _module: 1
    adapter._unpatchify = lambda value, patch_size: value
    return adapter


def test_cached_sequence_matches_full_sequence():
    adapter = make_lightweight_adapter()
    state = torch.randn(1, 4, 48, 30, 48)
    full = adapter.decode(state, cache_mode="disabled").video
    cached = adapter.decode(state, cache_mode="reset").video
    torch.testing.assert_close(full, cached)


def test_reuse_requires_explicit_state_and_reset_discards_history():
    adapter = make_lightweight_adapter()
    frame = torch.randn(1, 1, 48, 30, 48)
    with pytest.raises(ValueError):
        adapter.decode(frame, cache_mode="reuse")
    first = adapter.decode(frame, cache_mode="reset")
    reused = adapter.decode(frame, cache_mode="reuse", cache_state=first.cache_state)
    reset = adapter.decode(frame, cache_mode="reset")
    assert not torch.allclose(reused.video, reset.video)



def test_cache_aware_gradient_checkpointing_backpropagates():
    adapter = make_lightweight_adapter().train()
    adapter.enable_gradient_checkpointing(True)
    state = torch.randn(1, 3, 48, 30, 48, requires_grad=True)

    output = adapter.decode(state, cache_mode="reset", sequence_id="gradient-smoke")
    output.video.square().mean().backward()

    assert state.grad is not None
    assert torch.isfinite(state.grad).all()
    assert any(
        parameter.grad is not None
        for parameter in adapter.temporal_adapter.parameters()
    )
