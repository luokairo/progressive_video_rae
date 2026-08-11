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

    version: str = "native_vjepa_prefix_r4_v3"
    temporal_mode: str = "native_vjepa_prefix_r4"
    decoder_mode: str = "rae_latent_causal_bridge_wan_native_r4"
    spatial_prefix_fill: str = "shared_zero_init_learnable_mask_v1"
    prefix_indexing: str = "zero_based_inclusive_endpoint_v1"
    layout_version: str = "fps_v2_h30_w48_s48_k30"
    layout_checksum: str = "00c67dda3753fe5c7f800b2f20d84a1116a9acfa07fcaf5b7281910d2048c535"
    height: int = 30
    width: int = 48
    channels: int = 48
    num_sets: int = 48
    tokens_per_set: int = 30
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

    def assert_layout_identity(self, *, version: str | None, checksum: str | None) -> None:
        if version != self.layout_version or checksum != self.layout_checksum:
            raise ValueError(
                "State layout identity violates StateContract: "
                f"expected=({self.layout_version}, {self.layout_checksum}), "
                f"got=({version}, {checksum})"
            )


@dataclass
class ProgressiveState:
    """Complete production state. It never contains spatial-prefix mask tokens."""

    tokens: Tensor  # [B, T, S, K, C]
    layout_version: str
    layout_checksum: str
    latent_types: Tensor | None = None  # [T], IMAGE_FIRST_ID or VIDEO_GROUP_ID
    contract: StateContract | None = None

    def __post_init__(self) -> None:
        if self.tokens.ndim != 5:
            raise ValueError(f"ProgressiveState.tokens must be [B,T,S,K,C], got {self.tokens.shape}")
        if self.latent_types is not None:
            if self.latent_types.ndim != 1 or self.latent_types.numel() != self.tokens.shape[1]:
                raise ValueError("latent_types must have shape [T]")
            valid = (self.latent_types == IMAGE_FIRST_ID) | (
                self.latent_types == VIDEO_GROUP_ID
            )
            if not bool(valid.all()):
                raise ValueError("latent_types contains an unsupported value")
        if self.contract is None:
            raise ValueError("ProgressiveState must carry a StateContract")
        expected = (
            self.contract.num_sets,
            self.contract.tokens_per_set,
            self.contract.channels,
        )
        if tuple(self.tokens.shape[2:]) != expected:
            raise ValueError(
                f"State shape {tuple(self.tokens.shape[2:])} violates contract {expected}"
            )
        self.contract.assert_layout_identity(
            version=self.layout_version,
            checksum=self.layout_checksum,
        )

    @property
    def full_endpoint(self) -> int:
        if self.contract is None:
            raise ValueError("ProgressiveState must carry a StateContract")
        return self.contract.num_sets - 1


@dataclass
class SpatialPrefixView:
    """Dense masked training view; it is never a canonical production state."""

    tokens: Tensor  # [B, T, S, K, C]
    endpoint: int
    source: ProgressiveState | None = None
    latent_types: Tensor | None = None
    contract: StateContract | None = None
    layout_version: str | None = None
    layout_checksum: str | None = None

    def __post_init__(self) -> None:
        if self.source is not None:
            if self.tokens.shape != self.source.tokens.shape:
                raise ValueError("SpatialPrefixView must preserve the canonical state shape")
            if self.contract is not None and self.contract != self.source.contract:
                raise ValueError("SpatialPrefixView contract must match its source state")
            if (
                self.layout_version is not None
                and self.layout_version != self.source.layout_version
            ):
                raise ValueError(
                    "SpatialPrefixView layout identity version must match its source state"
                )
            if (
                self.layout_checksum is not None
                and self.layout_checksum != self.source.layout_checksum
            ):
                raise ValueError(
                    "SpatialPrefixView layout identity checksum must match its source state"
                )
            if self.latent_types is None:
                self.latent_types = self.source.latent_types
            if self.contract is None:
                self.contract = self.source.contract
            if self.layout_version is None:
                self.layout_version = self.source.layout_version
            if self.layout_checksum is None:
                self.layout_checksum = self.source.layout_checksum
        if self.contract is None:
            raise ValueError("SpatialPrefixView must carry a StateContract")
        self.contract.assert_layout_identity(
            version=self.layout_version,
            checksum=self.layout_checksum,
        )
        expected = (
            self.contract.num_sets,
            self.contract.tokens_per_set,
            self.contract.channels,
        )
        if self.tokens.ndim != 5 or tuple(self.tokens.shape[2:]) != expected:
            raise ValueError(f"SpatialPrefixView shape must end in {expected}")
        if not 0 <= self.endpoint < self.contract.num_sets - 1:
            raise ValueError("SpatialPrefixView endpoint must be in [0, num_sets-2]")


@dataclass(frozen=True)
class RepaReference:
    anchor: Tensor  # [B, 1, H, W, C]
    video_phases: Tensor  # [B, G, 2, H, W, C]


@dataclass(frozen=True)
class ProjectorOutput:
    state: ProgressiveState
    repa_reference: RepaReference
    prefix_windows: tuple[tuple[int, int, int], ...]



def assert_video_tensor(pixel_values: Tensor, *, frames: int, height: int, width: int) -> None:
    expected = (3, frames, height, width)
    if pixel_values.ndim != 5 or tuple(pixel_values.shape[1:]) != expected:
        raise ValueError(
            f"Expected pixel_values [B,{expected[0]},{expected[1]},{expected[2]},{expected[3]}], "
            f"got {tuple(pixel_values.shape)}"
        )
    if not torch.is_floating_point(pixel_values):
        raise TypeError("pixel_values must be floating point in [0, 1]")
