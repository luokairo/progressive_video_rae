from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn

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


def test_paired_p47_keeps_inclusive_endpoint():
    model = RecordingModel()
    pixels = torch.zeros(1, 3, 17, 2, 2)
    forward_training_batch(
        model,
        pixels,
        task=MicrobatchTask("paired_prefix", 47, 46),
        stage="stage1b",
    )
    assert model.kwargs["endpoint"] == 47
    assert model.kwargs["paired_previous_endpoint"] == 46


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
