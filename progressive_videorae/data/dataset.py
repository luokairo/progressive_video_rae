from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import random
from typing import Any, Literal

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset


class VideoDecodeError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoSamplingConfig:
    num_frames: int = 16
    target_fps: float = 12.0
    height: int = 480
    width: int = 768
    split: Literal["train", "val", "test"] = "train"
    horizontal_flip: bool = True


def _duration_seconds(container, stream) -> float | None:
    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    if container.duration is not None:
        import av

        return float(container.duration / av.time_base)
    return None


def decode_contiguous_clip(path: str, config: VideoSamplingConfig) -> tuple[Tensor, dict[str, Any]]:
    try:
        import av
    except ImportError as exc:
        raise ImportError("Install av to decode videos") from exc

    with av.open(path, metadata_errors="ignore") as container:
        if not container.streams.video:
            raise VideoDecodeError(f"No video stream: {path}")
        stream = container.streams.video[0]
        native_fps = float(stream.average_rate) if stream.average_rate else config.target_fps
        duration = _duration_seconds(container, stream)
        span = (config.num_frames - 1) / config.target_fps
        if duration is not None and duration + 1e-6 < span:
            raise VideoDecodeError(f"Video is too short ({duration:.3f}s): {path}")
        max_start = max(0.0, (duration - span) if duration is not None else 0.0)
        start = random.uniform(0.0, max_start) if config.split == "train" else max_start / 2.0
        timestamps = [start + index / config.target_fps for index in range(config.num_frames)]

        if stream.time_base is not None:
            seek_pts = int(max(0.0, start - 1.0) / float(stream.time_base))
            container.seek(seek_pts, stream=stream, backward=True)
        frames = []
        frame_indices = []
        target_index = 0
        decoded_index = 0
        for frame in container.decode(stream):
            frame_time = (
                float(frame.pts * stream.time_base)
                if frame.pts is not None
                else decoded_index / native_fps
            )
            while (
                target_index < len(timestamps)
                and frame_time + 0.5 / native_fps >= timestamps[target_index]
            ):
                array = frame.to_ndarray(format="rgb24")
                frames.append(torch.from_numpy(array))
                frame_indices.append(decoded_index)
                target_index += 1
            decoded_index += 1
            if target_index >= len(timestamps):
                break
        if len(frames) != config.num_frames:
            raise VideoDecodeError(
                f"Decoded {len(frames)}/{config.num_frames} target frames: {path}"
            )

    video = torch.stack(frames).permute(0, 3, 1, 2).float().div_(255.0)
    _, _, source_h, source_w = video.shape
    scale = max(config.height / source_h, config.width / source_w)
    resized_h = max(config.height, math.ceil(source_h * scale))
    resized_w = max(config.width, math.ceil(source_w * scale))
    video = F.interpolate(video, size=(resized_h, resized_w), mode="bilinear", align_corners=False)
    if config.split == "train":
        top = random.randint(0, resized_h - config.height)
        left = random.randint(0, resized_w - config.width)
    else:
        top = (resized_h - config.height) // 2
        left = (resized_w - config.width) // 2
    video = video[:, :, top : top + config.height, left : left + config.width]
    if config.split == "train" and config.horizontal_flip and random.random() < 0.5:
        video = video.flip(-1)
    return video.permute(1, 0, 2, 3).contiguous(), {
        "native_fps": native_fps,
        "sampled_timestamps": torch.tensor(timestamps, dtype=torch.float64),
        "sampled_frame_indices": torch.tensor(frame_indices, dtype=torch.long),
    }


class VideoManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: Literal["train", "val", "test"] = "train",
        num_frames: int = 16,
        target_fps: float = 12.0,
        height: int = 480,
        width: int = 768,
        horizontal_flip: bool = True,
        max_decode_retries: int = 3,
    ) -> None:
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("Install pandas and pyarrow to load manifests") from exc
        self.frame = pd.read_parquet(manifest_path).reset_index(drop=True)
        minimum_duration = (num_frames - 1) / target_fps
        if "path_exists" in self.frame:
            self.frame = self.frame[self.frame["path_exists"].fillna(False)]
        if "decode_valid" in self.frame:
            self.frame = self.frame[self.frame["decode_valid"].fillna(False)]
        if "duration" in self.frame:
            duration = self.frame["duration"]
            self.frame = self.frame[duration.isna() | (duration >= minimum_duration - 1e-6)]
        self.frame = self.frame.reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(
                f"No videos can provide {num_frames} frames at {target_fps} FPS"
            )
        self.config = VideoSamplingConfig(
            num_frames=num_frames,
            target_fps=target_fps,
            height=height,
            width=width,
            split=split,
            horizontal_flip=horizontal_flip,
        )
        self.max_decode_retries = max_decode_retries

    def __len__(self) -> int:
        return len(self.frame)

    def _candidate_index(self, index: int, attempt: int) -> int:
        if attempt == 0:
            return index
        return (index + 104729 * attempt) % len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.max_decode_retries + 1):
            candidate = self._candidate_index(index, attempt)
            row = self.frame.iloc[candidate]
            try:
                pixel_values, metadata = decode_contiguous_clip(str(row.path), self.config)
            except Exception as exc:
                last_error = exc
                continue
            source_tags = row.source_tags
            if hasattr(source_tags, "tolist"):
                source_tags = source_tags.tolist()
            if isinstance(source_tags, str):
                source_tags = [source_tags]
            sampled_indices = metadata["sampled_frame_indices"]
            sampled_timestamps = metadata["sampled_timestamps"]
            identity_payload = (
                f"{row.sample_id}|indices="
                + ",".join(str(int(value)) for value in sampled_indices.tolist())
                + "|timestamps="
                + ",".join(f"{float(value):.9f}" for value in sampled_timestamps.tolist())
            )
            codec_sequence_id = identity_payload
            return {
                "pixel_values": pixel_values,
                "caption": "" if row.caption is None else str(row.caption),
                "path": str(row.path),
                "sample_id": str(row.sample_id),
                "category": str(row.category),
                "source_tags": list(source_tags),
                "codec_sequence_id": codec_sequence_id,
                "is_sequence_start": torch.tensor(True, dtype=torch.bool),
                "sequence_origin": "sampled_segment",
                "segment_start_timestamp": sampled_timestamps[0].clone(),
                **metadata,
            }
        raise VideoDecodeError(
            f"Failed after {self.max_decode_retries + 1} attempts"
        ) from last_error


def collate_video_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    tensor_keys = (
        "pixel_values",
        "sampled_timestamps",
        "sampled_frame_indices",
        "is_sequence_start",
        "segment_start_timestamp",
    )
    batch = {
        key: torch.stack([sample[key] for sample in samples])
        for key in tensor_keys
    }
    for key in (
        "caption",
        "path",
        "sample_id",
        "category",
        "source_tags",
        "native_fps",
        "codec_sequence_id",
        "sequence_origin",
    ):
        batch[key] = [sample[key] for sample in samples]
    return batch
