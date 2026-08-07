from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass
class EncoderOutput:
    """Common output of all video foundation encoder adapters."""

    tokens: Tensor  # [B, T, H, W, C]
    grid_size: tuple[int, int, int]
    layer_tokens: tuple[Tensor, ...] = ()


@dataclass
class ProgressiveState:
    """Fixed-layout continuous state consumed by the Wan decoder."""

    tokens: Tensor  # [B, T, H, W, C]
    flat_tokens: Tensor  # [B, T, H*W, C]
    set_ids: Tensor  # [H, W]
    set_sizes: Tensor  # [num_sets]
    prefix_len: int
    layout_version: str
    layout_checksum: str
    metadata: dict[str, Any] | None = None


def assert_video_tensor(pixel_values: Tensor, *, frames: int, height: int, width: int) -> None:
    expected = (3, frames, height, width)
    if pixel_values.ndim != 5 or tuple(pixel_values.shape[1:]) != expected:
        raise ValueError(
            f"Expected pixel_values [B,{expected[0]},{expected[1]},{expected[2]},{expected[3]}], "
            f"got {tuple(pixel_values.shape)}"
        )
    if not torch.is_floating_point(pixel_values):
        raise TypeError("pixel_values must be floating point in [0, 1]")

