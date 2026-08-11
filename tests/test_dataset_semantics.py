from __future__ import annotations

import pandas as pd
import pytest
import torch

from progressive_videorae.data import dataset as dataset_module
from progressive_videorae.data.dataset import (
    VideoManifestDataset,
    VideoSamplingConfig,
    collate_video_samples,
)
from progressive_videorae.training.stages import validate_training_batch


def test_random_clip_is_explicit_codec_segment_start(monkeypatch):
    sampled_timestamps = torch.tensor([1.25, 1.5, 1.75], dtype=torch.float64)
    sampled_indices = torch.tensor([30, 36, 42], dtype=torch.long)

    def fake_decode(_path, _config):
        return torch.zeros(3, 3, 2, 2), {
            "native_fps": 24.0,
            "sampled_timestamps": sampled_timestamps,
            "sampled_frame_indices": sampled_indices,
        }

    monkeypatch.setattr(dataset_module, "decode_contiguous_clip", fake_decode)
    dataset = VideoManifestDataset.__new__(VideoManifestDataset)
    dataset.frame = pd.DataFrame(
        [
            {
                "path": "/video/source.mp4",
                "caption": "caption",
                "sample_id": "sample-a",
                "category": "human",
                "source_tags": ["human"],
            }
        ]
    )
    dataset.config = VideoSamplingConfig(num_frames=3, height=2, width=2)
    dataset.max_decode_retries = 0

    sample = dataset[0]

    assert sample["is_sequence_start"].dtype == torch.bool
    assert bool(sample["is_sequence_start"])
    assert sample["sequence_origin"] == "sampled_segment"
    assert sample["segment_start_timestamp"].dtype == torch.float64
    assert sample["segment_start_timestamp"].item() == pytest.approx(1.25)
    assert sample["sample_id"] == "sample-a"
    assert "indices=30,36,42" in sample["codec_sequence_id"]
    assert "timestamps=1.250000000,1.500000000,1.750000000" in sample[
        "codec_sequence_id"
    ]


def _sample(sequence_id: str, *, sequence_start: bool = True) -> dict:
    frames = 33
    return {
        "pixel_values": torch.zeros(3, frames, 2, 4),
        "sampled_timestamps": torch.arange(frames, dtype=torch.float64) / 12.0,
        "sampled_frame_indices": torch.arange(frames, dtype=torch.long),
        "is_sequence_start": torch.tensor(sequence_start, dtype=torch.bool),
        "segment_start_timestamp": torch.tensor(0.0, dtype=torch.float64),
        "caption": "",
        "path": "/video/source.mp4",
        "sample_id": sequence_id,
        "category": "human",
        "source_tags": ["human"],
        "native_fps": 24.0,
        "codec_sequence_id": sequence_id,
        "sequence_origin": "sampled_segment",
    }


def test_collate_preserves_segment_metadata_and_stage2b_preflight():
    batch = collate_video_samples([_sample("segment-a"), _sample("segment-b")])

    assert batch["is_sequence_start"].shape == (2,)
    assert batch["is_sequence_start"].dtype == torch.bool
    assert batch["segment_start_timestamp"].shape == (2,)
    assert batch["segment_start_timestamp"].dtype == torch.float64
    assert batch["codec_sequence_id"] == ["segment-a", "segment-b"]
    assert batch["sequence_origin"] == ["sampled_segment", "sampled_segment"]
    validate_training_batch(
        batch,
        stage="stage2b",
        expected_frames=33,
        expected_height=2,
        expected_width=4,
    )


def test_stage2b_preflight_rejects_implicit_continuation():
    batch = collate_video_samples([_sample("segment-a", sequence_start=False)])

    with pytest.raises(ValueError, match="declare sequence start"):
        validate_training_batch(
            batch,
            stage="stage2b",
            expected_frames=33,
            expected_height=2,
            expected_width=4,
        )
