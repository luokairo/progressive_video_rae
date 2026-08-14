from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from progressive_videorae.model.model import ProgressiveVideoRAE
from progressive_videorae.train import _sequence_id, forward_training_batch
from progressive_videorae.training.stages import MicrobatchTask


class RecordingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.kwargs = None

    def forward(self, pixel_values, **kwargs):
        self.kwargs = kwargs
        return pixel_values


@pytest.mark.parametrize(
    ("stage", "cache_mode"), (("stage2a", "disabled"), ("stage2b", "reset"))
)
def test_stage2_forward_is_full_repa_without_prefix(stage, cache_mode):
    model = RecordingModel()
    pixels = torch.zeros(1, 3, 17, 2, 2)
    output = forward_training_batch(
        model, pixels, task=MicrobatchTask("full"), stage=stage, sequence_id="sequence"
    )
    assert output is pixels
    assert model.kwargs == {
        "endpoint": None,
        "paired_previous_endpoint": None,
        "cache_mode": cache_mode,
        "return_decoder_features": True,
        "sequence_id": "sequence",
    }


def test_stage2_forward_rejects_prefix_task():
    with pytest.raises(RuntimeError, match="cannot execute"):
        forward_training_batch(
            RecordingModel(),
            torch.zeros(1, 3, 17, 2, 2),
            task=MicrobatchTask("single_prefix", 12),
            stage="stage2a",
        )


def test_stage1b_prefix_is_single_view_with_repa_features():
    model = RecordingModel()
    pixels = torch.zeros(1, 3, 17, 2, 2)
    forward_training_batch(
        model,
        pixels,
        task=MicrobatchTask("single_prefix", 12),
        stage="stage1b",
    )
    assert model.kwargs["endpoint"] == 12
    assert model.kwargs["paired_previous_endpoint"] is None
    assert model.kwargs["return_decoder_features"] is True
    assert model.kwargs["cache_mode"] == "disabled"


def test_stage1b_p47_full_anchor_is_canonical_with_repa_features():
    model = RecordingModel()
    pixels = torch.zeros(1, 3, 17, 2, 2)
    forward_training_batch(
        model,
        pixels,
        task=MicrobatchTask("full"),
        stage="stage1b",
    )
    assert model.kwargs["endpoint"] is None
    assert model.kwargs["paired_previous_endpoint"] is None
    assert model.kwargs["return_decoder_features"] is True
    assert model.kwargs["cache_mode"] == "disabled"


def test_endpoint_47_decode_uses_canonical_state_without_mask_replacement():
    state = SimpleNamespace(full_endpoint=47)
    projected = SimpleNamespace(state=state, repa_reference=None)

    class Harness:
        def __init__(self):
            self.canonical_state = None
            self.projector = SimpleNamespace(
                make_prefix_view=lambda *_args: pytest.fail(
                    "P47 must not construct a SpatialPrefixView"
                )
            )

        def _decode_state_view(self, _pixels, _encoder, state_view, **kwargs):
            self.canonical_state = kwargs["canonical_state"]
            return state_view

    harness = Harness()
    result = ProgressiveVideoRAE._decode_projected(
        harness,
        torch.zeros(1),
        object(),
        projected,
        endpoint=47,
        cache_mode="disabled",
        cache_state=None,
        return_decoder_features=False,
        sequence_id=None,
    )

    assert result is state
    assert harness.canonical_state is state


def test_sequence_identity_contains_all_indices_and_timestamps():
    batch = {
        "sample_id": ["sample-a"],
        "sampled_frame_indices": [torch.tensor([3, 5, 7])],
        "sampled_timestamps": [torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)],
    }

    identity = _sequence_id(batch)

    assert identity == (
        "sample-a|indices=3,5,7|"
        "timestamps=0.250000000,0.500000000,0.750000000"
    )
