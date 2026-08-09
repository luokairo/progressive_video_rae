from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
from torch import Tensor


@dataclass
class EncoderOutput:
    """Common output of all video foundation encoder adapters."""

    tokens: Tensor  # [B, T, H, W, C]
    grid_size: tuple[int, int, int]
    layer_tokens: tuple[Tensor, ...] = ()


LatentType = Literal["image_first", "video_group"]
IMAGE_FIRST_ID = 0
VIDEO_GROUP_ID = 1


@dataclass(frozen=True)
class PrefixGroupFeatures:
    """Unmodified V-JEPA features selected from one causal RGB prefix/window."""

    tokens: Tensor  # [B, 1|2, H, W, C]
    layer_tokens: tuple[Tensor, ...]
    latent_type: LatentType
    source_start: int
    source_end: int
    input_frames: int


@dataclass(frozen=True)
class PrefixEncoderOutput:
    """One immutable V-JEPA result per production latent frame."""

    groups: tuple[PrefixGroupFeatures, ...]
    spatial_grid: tuple[int, int]
    embed_dim: int
    max_context_frames: int

    @property
    def num_latents(self) -> int:
        return len(self.groups)


@dataclass(frozen=True)
class StateContract:
    """Versioned production-state contract shared by RAE, NOVA, and decoders."""

    version: str = "native_vjepa_prefix_r4_v1"
    temporal_mode: str = "native_vjepa_prefix_r4"
    decoder_mode: str = "rae_group_expand_wan_spatial"
    height: int = 30
    width: int = 48
    channels: int = 48
    num_sets: int = 64
    max_encoder_context_frames: int = 64
    first_latent_rgb_frames: int = 1
    continuation_latent_rgb_frames: int = 4
    normalization: str = "token_layernorm_affine_v1"

    def rgb_frames_for_latents(self, latent_count: int, *, sequence_start: bool) -> int:
        if latent_count <= 0:
            raise ValueError("latent_count must be positive")
        if sequence_start:
            return self.first_latent_rgb_frames + self.continuation_latent_rgb_frames * (
                latent_count - 1
            )
        return self.continuation_latent_rgb_frames * latent_count

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def assert_compatible(self, other: "StateContract") -> None:
        if self != other:
            raise ValueError(
                f"Incompatible StateContract: expected={self.to_dict()}, got={other.to_dict()}"
            )


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
    latent_types: Tensor | None = None  # [T], IMAGE_FIRST_ID or VIDEO_GROUP_ID
    contract: StateContract | None = None

    def __post_init__(self) -> None:
        if self.tokens.ndim != 5:
            raise ValueError(f"ProgressiveState.tokens must be [B,T,H,W,C], got {self.tokens.shape}")
        if self.latent_types is not None:
            if self.latent_types.ndim != 1 or self.latent_types.numel() != self.tokens.shape[1]:
                raise ValueError("latent_types must have shape [T]")
            valid = (self.latent_types == IMAGE_FIRST_ID) | (
                self.latent_types == VIDEO_GROUP_ID
            )
            if not bool(valid.all()):
                raise ValueError("latent_types contains an unsupported value")
        if self.contract is not None:
            expected = (
                self.contract.height,
                self.contract.width,
                self.contract.channels,
            )
            if tuple(self.tokens.shape[2:]) != expected:
                raise ValueError(
                    f"State shape {tuple(self.tokens.shape[2:])} violates contract {expected}"
                )


def assert_video_tensor(pixel_values: Tensor, *, frames: int, height: int, width: int) -> None:
    expected = (3, frames, height, width)
    if pixel_values.ndim != 5 or tuple(pixel_values.shape[1:]) != expected:
        raise ValueError(
            f"Expected pixel_values [B,{expected[0]},{expected[1]},{expected[2]},{expected[3]}], "
            f"got {tuple(pixel_values.shape)}"
        )
    if not torch.is_floating_point(pixel_values):
        raise TypeError("pixel_values must be floating point in [0, 1]")
