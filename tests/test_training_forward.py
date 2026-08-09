from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from progressive_videorae.train import forward_training_batch


class RecordingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.kwargs = None

    def forward(self, pixel_values, **kwargs):
        self.kwargs = kwargs
        return pixel_values


def test_training_forward_passes_sampled_prefix_to_model():
    model = RecordingModel()
    pixels = torch.zeros(1, 3, 1, 2, 2)

    output = forward_training_batch(model, pixels, prefix_len=17, full_state=False)

    assert output is pixels
    assert model.kwargs == {
        "prefix_len": 17,
        "cache_mode": "disabled",
        "return_decoder_features": False,
    }
