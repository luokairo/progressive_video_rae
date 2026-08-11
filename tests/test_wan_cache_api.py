import copy

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from progressive_videorae.model.progressive_sets import build_progressive_layout
from progressive_videorae.model.types import (
    IMAGE_FIRST_ID,
    VIDEO_GROUP_ID,
    ProgressiveState,
    StateContract,
)
from progressive_videorae.model.wan_decoder import RAETemporalAdapter, WanVideoDecoder


class FakeCausalDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsamples = nn.ModuleList([nn.Identity()])
        self.gain = nn.Parameter(torch.tensor(1.0))
        self.calls = 0
        self.first_chunks = []

    def forward(self, x, feat_cache=None, feat_idx=None, first_chunk=False):
        self.calls += 1
        self.first_chunks.append(bool(first_chunk))
        x = self.gain * x
        if feat_cache is None:
            return x.cumsum(dim=2)
        previous = 0.0 if feat_cache[0] is None else feat_cache[0]
        outputs = []
        for index in range(x.shape[2]):
            previous = previous + x[:, :, index : index + 1]
            outputs.append(previous)
        feat_cache[0] = previous
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


def make_state(
    tokens,
    *,
    sequence_start: bool,
    latent_types=None,
):
    layout = build_progressive_layout()
    if latent_types is None:
        latent_types = torch.full(
            (tokens.shape[1],), VIDEO_GROUP_ID, dtype=torch.long, device=tokens.device
        )
        if sequence_start:
            latent_types[0] = IMAGE_FIRST_ID
    return ProgressiveState(
        tokens=tokens,
        layout_version=layout.version,
        layout_checksum=layout.checksum,
        latent_types=latent_types,
        contract=StateContract(),
    )


def test_cached_sequence_matches_full_sequence():
    adapter = make_lightweight_adapter()
    state = torch.randn(1, 4, 48, 30, 48)
    full = adapter.decode(state, cache_mode="disabled").video
    cached = adapter.decode(
        make_state(state, sequence_start=True),
        cache_mode="reset",
        sequence_id="parity",
    ).video
    torch.testing.assert_close(full, cached)


def test_reuse_requires_explicit_state_and_reset_discards_history():
    adapter = make_lightweight_adapter()
    frame = torch.randn(1, 1, 48, 30, 48)
    with pytest.raises(TypeError, match="requires ProgressiveState"):
        adapter.decode(frame, cache_mode="reuse")
    typed_reset = make_state(frame, sequence_start=True)
    with pytest.raises(ValueError, match="non-empty sequence_id"):
        adapter.decode(typed_reset, cache_mode="reset")
    first = adapter.decode(
        typed_reset,
        cache_mode="reset",
        sequence_id="sequence-a",
    )
    reused = adapter.decode(
        make_state(frame, sequence_start=False),
        cache_mode="reuse",
        cache_state=first.cache_state,
        sequence_id="sequence-a",
    )
    reset = adapter.decode(
        make_state(frame, sequence_start=True),
        cache_mode="reset",
        sequence_id="sequence-b",
    )
    assert not torch.allclose(reused.video, reset.video)

def test_cache_aware_gradient_checkpointing_backpropagates():
    adapter = make_lightweight_adapter().train()
    adapter.enable_gradient_checkpointing(True)
    state = torch.randn(1, 3, 48, 30, 48, requires_grad=True)

    output = adapter.decode(
        make_state(state, sequence_start=True), cache_mode="reset", sequence_id="gradient-smoke"
    )
    output.video.square().mean().backward()

    assert state.grad is not None
    assert torch.isfinite(state.grad).all()
    assert any(
        parameter.grad is not None
        for parameter in adapter.temporal_adapter.parameters()
    )


@pytest.mark.parametrize("latent_count", [5, 9])
def test_every_latent_calls_bridge_and_wan_once(latent_count):
    adapter = make_lightweight_adapter()
    latent_type_calls = []
    handle = adapter.temporal_adapter.register_forward_pre_hook(
        lambda _module, inputs: latent_type_calls.append(int(inputs[1].item()))
    )
    try:
        adapter.decode(
            torch.randn(1, latent_count, 48, 30, 48),
            cache_mode="disabled",
        )
    finally:
        handle.remove()

    assert adapter.decoder.calls == latent_count
    assert adapter.decoder.first_chunks == [True] + [False] * (latent_count - 1)
    assert latent_type_calls == [0] + [1] * (latent_count - 1)


def _decode_with_splits(adapter, source, splits):
    state = source.detach().clone().requires_grad_(True)
    if splits is None:
        video = adapter.decode(state, cache_mode="disabled").video
    else:
        videos = []
        cache = None
        start = 0
        for index, size in enumerate(splits):
            chunk = state[:, start : start + size]
            output = adapter.decode(
                make_state(chunk, sequence_start=index == 0),
                cache_mode="reset" if index == 0 else "reuse",
                cache_state=cache,
                sequence_id="parity",
            )
            videos.append(output.video)
            cache = output.cache_state
            start += size
        video = torch.cat(videos, dim=2)
    gradient = torch.autograd.grad(video.square().mean(), state)[0]
    return video.detach(), gradient.detach()


def test_disabled_reset_and_cross_call_chunks_match_outputs_and_input_gradients():
    torch.manual_seed(7)
    base = make_lightweight_adapter().train()
    source = torch.randn(1, 9, 48, 30, 48)
    paths = {
        "disabled": None,
        "reset_full": [9],
        "reset_1_reuse_8": [1, 8],
        "reset_5_reuse_4": [5, 4],
    }
    results = {
        name: _decode_with_splits(copy.deepcopy(base), source, splits)
        for name, splits in paths.items()
    }
    reference_video, reference_gradient = results["disabled"]
    for name, (video, gradient) in results.items():
        torch.testing.assert_close(
            video,
            reference_video,
            atol=1e-5,
            rtol=1e-5,
            msg=lambda message: f"{name}: {message}",
        )
        torch.testing.assert_close(
            gradient,
            reference_gradient,
            atol=1e-5,
            rtol=1e-5,
            msg=lambda message: f"{name}: {message}",
        )


def test_cross_call_training_keeps_bridge_and_wan_cache_graphs():
    adapter = make_lightweight_adapter().train()
    state = torch.randn(1, 3, 48, 30, 48, requires_grad=True)

    first = adapter.decode(
        make_state(state[:, :1], sequence_start=True),
        cache_mode="reset",
        sequence_id="gradient-sequence",
    )
    assert first.cache_state.rae.raw_history.grad_fn is not None
    assert first.cache_state.features[0].grad_fn is not None
    second = adapter.decode(
        make_state(state[:, 1:], sequence_start=False),
        cache_mode="reuse",
        cache_state=first.cache_state,
        sequence_id="gradient-sequence",
    )
    torch.cat((first.video, second.video), dim=2).square().mean().backward()

    assert state.grad is not None
    assert adapter.temporal_adapter.state_to_wan.weight.grad is not None
    assert adapter.decoder.gain.grad is not None
    assert second.cache_state.rae.raw_history.grad_fn is not None


def test_latent_type_contract_is_strict_for_disabled_reset_and_reuse():
    adapter = make_lightweight_adapter()
    state = torch.randn(1, 2, 48, 30, 48)
    image_first = torch.tensor([0, 0])
    video_first = torch.tensor([1, 1])

    with pytest.raises(ValueError, match="after image_first"):
        adapter.decode(state, cache_mode="disabled", latent_types=image_first)
    with pytest.raises(ValueError, match="begin with an image_first"):
        adapter.decode(
            make_state(state, sequence_start=False, latent_types=video_first),
            cache_mode="reset",
            sequence_id="types",
        )
    first = adapter.decode(
        make_state(state[:, :1], sequence_start=True),
        cache_mode="reset",
        sequence_id="types",
    )
    with pytest.raises(ValueError, match="video_group"):
        adapter.decode(
            make_state(
                state[:, :1], sequence_start=False, latent_types=torch.tensor([0])
            ),
            cache_mode="reuse",
            cache_state=first.cache_state,
            sequence_id="types",
        )


@pytest.mark.parametrize("sequence_id", [None, "", "   "])
def test_stateful_modes_reject_missing_or_blank_sequence_id(sequence_id):
    adapter = make_lightweight_adapter()
    state = make_state(torch.randn(1, 1, 48, 30, 48), sequence_start=True)
    with pytest.raises(ValueError, match="non-empty sequence_id"):
        adapter.decode(
            state, cache_mode="reset", sequence_id=sequence_id
        )


def test_reuse_rejects_wrong_sequence_id():
    adapter = make_lightweight_adapter()
    frame = torch.randn(1, 1, 48, 30, 48)
    first = adapter.decode(
        make_state(frame, sequence_start=True),
        cache_mode="reset",
        sequence_id="sequence-a",
    )
    with pytest.raises(ValueError, match="different sequence_id"):
        adapter.decode(
            make_state(frame, sequence_start=False),
            cache_mode="reuse",
            cache_state=first.cache_state,
            sequence_id="sequence-b",
        )


def test_decoder_rechecks_mutable_state_layout_identity_every_call():
    adapter = make_lightweight_adapter()
    state = make_state(torch.randn(1, 1, 48, 30, 48), sequence_start=True)
    state.layout_checksum = "tampered"
    with pytest.raises(ValueError, match="layout identity"):
        adapter.decode(state, cache_mode="disabled")


def test_cache_binds_complete_state_contract_identity():
    adapter = make_lightweight_adapter()
    frame = torch.randn(1, 1, 48, 30, 48)
    first = adapter.decode(
        make_state(frame, sequence_start=True),
        cache_mode="reset",
        sequence_id="contract",
    )
    first.cache_state.state_contract = StateContract(normalization="tampered")
    with pytest.raises(ValueError, match="StateContract identity"):
        adapter.decode(
            make_state(frame, sequence_start=False),
            cache_mode="reuse",
            cache_state=first.cache_state,
            sequence_id="contract",
        )
