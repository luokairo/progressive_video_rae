from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .encoders.base import VideoFoundationEncoder
from .projector import CausalFrequencyProjector
from .types import (
    PrefixEncoderOutput,
    ProgressiveState,
    ProgressiveStateChunk,
    RepaReference,
    SpatialPrefixView,
    assert_video_tensor,
    ProjectorOutput,
)
from .wan_decoder import CacheMode, WanCacheState, WanDecoderOutput, WanVideoDecoder


@dataclass
class ProgressiveVideoRAEOutput:
    reconstruction: Tensor
    target: Tensor
    state: ProgressiveState | None
    state_view: ProgressiveState | SpatialPrefixView
    encoder_output: PrefixEncoderOutput
    decoder_output: WanDecoderOutput
    repa_features: RepaReference | None = None
    repa_reference: RepaReference | None = None

@dataclass
class ProgressiveVideoRAEChunkOutput:
    reconstruction: Tensor
    target: Tensor
    state_chunk: ProgressiveStateChunk
    encoder_output: PrefixEncoderOutput
    decoder_output: WanDecoderOutput



class PhaseSpecificRepaProjection(nn.Module):
    def __init__(self, decoder_dim: int, encoder_dim: int) -> None:
        super().__init__()
        self.anchor = nn.Conv2d(decoder_dim, encoder_dim, 1)
        self.video_phases = nn.Conv2d(decoder_dim, encoder_dim * 2, 1)

    def forward(self, features: Tensor) -> RepaReference:
        if features.ndim != 5 or features.shape[2] < 1:
            raise ValueError("REPA features must be [B,C,T>=1,H,W]")
        anchor = self.anchor(features[:, :, 0]).permute(0, 2, 3, 1).unsqueeze(1)
        tail = features[:, :, 1:]
        b, c, groups, h, w = tail.shape
        if groups:
            flat = tail.permute(0, 2, 1, 3, 4).reshape(b * groups, c, h, w)
            phases = self.video_phases(flat)
            phases = phases.reshape(b, groups, 2, -1, h, w).permute(0, 1, 2, 4, 5, 3)
        else:
            phases = anchor.new_empty(b, 0, 2, h, w, anchor.shape[-1])
        return RepaReference(anchor=anchor, video_phases=phases)


class ProgressiveVideoRAE(nn.Module):
    def __init__(
        self,
        encoder: VideoFoundationEncoder,
        projector: CausalFrequencyProjector,
        decoder: WanVideoDecoder,
        encoder_dim: int,
        decoder_feature_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.decoder = decoder
        self.repa_projection = PhaseSpecificRepaProjection(decoder_feature_dim, encoder_dim)
        self.projector.contract.assert_compatible(self.decoder.contract)
        self._runtime_timing_enabled = False
        self._runtime_timing_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}

    def enable_runtime_timing(self, enabled: bool) -> None:
        self._runtime_timing_enabled = bool(enabled)
        self._runtime_timing_events.clear()

    def _timed(self, name: str, function):
        if not self._runtime_timing_enabled or not torch.cuda.is_available():
            return function()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = function()
        end.record()
        self._runtime_timing_events.setdefault(name, []).append((start, end))
        return result

    def runtime_timings(self) -> dict[str, float]:
        return {
            name: sum(start.elapsed_time(end) for start, end in pairs) / 1000.0
            for name, pairs in self._runtime_timing_events.items()
        }

    @property
    def state_contract(self):
        return self.projector.contract

    def pretrained_load_report(self) -> dict:
        encoder_report = getattr(self.encoder, "load_report", None)
        decoder_report = getattr(self.decoder, "load_report", None)
        return {
            "ready": bool(
                encoder_report is not None
                and encoder_report.ready
                and decoder_report is not None
                and decoder_report.ready
            ),
            "components": {
                "encoder": encoder_report.to_dict() if encoder_report is not None else None,
                "decoder": decoder_report.to_dict() if decoder_report is not None else None,
            },
            "random_initialized_components": [
                "causal_frequency_projector",
                "repa_projection",
                "patch_discriminator",
            ],
        }

    def assert_pretrained_ready(self) -> None:
        encoder_report = getattr(self.encoder, "load_report", None)
        decoder_report = getattr(self.decoder, "load_report", None)
        if encoder_report is None:
            raise RuntimeError("Encoder has no pretrained load report; training cannot start")
        if decoder_report is None:
            raise RuntimeError("Wan2.2 decoder has no pretrained load report; training cannot start")
        encoder_report.assert_ready()
        decoder_report.assert_ready()

    def encode_features(self, pixel_values: Tensor) -> PrefixEncoderOutput:
        frames = int(pixel_values.shape[2]) if pixel_values.ndim == 5 else -1
        if frames < 1 or (frames - 1) % 4:
            raise ValueError("ProgressiveVideoRAE requires F=1+4*n RGB frames")
        assert_video_tensor(pixel_values, frames=frames, height=480, width=768)
        encode_prefixes = getattr(self.encoder, "encode_prefixes", None)
        if encode_prefixes is None:
            raise TypeError("Encoder does not implement native causal prefix extraction")
        return encode_prefixes(pixel_values)

    def encode(self, pixel_values: Tensor):
        encoder_output = self.encode_features(pixel_values)
        projected = self.projector(encoder_output)
        return encoder_output, projected

    def encode_chunk(
        self,
        pixel_values: Tensor,
        *,
        sequence_id: str,
        latent_start: int,
        is_sequence_start: bool,
        target_fps: float,
        start_timestamp: float,
    ) -> tuple[PrefixEncoderOutput, ProgressiveStateChunk]:
        encoder_output, projected = self.encode(pixel_values)
        state = projected.state
        if not is_sequence_start:
            if state.tokens.shape[1] <= 1 or state.latent_types is None:
                raise ValueError("Continuation window must provide anchor plus video latents")
            state = ProgressiveState(
                tokens=state.tokens[:, 1:],
                layout_version=state.layout_version,
                layout_checksum=state.layout_checksum,
                latent_types=state.latent_types[1:],
                contract=state.contract,
            )
        return encoder_output, ProgressiveStateChunk(
            state=state,
            sequence_id=sequence_id,
            latent_start=latent_start,
            is_sequence_start=is_sequence_start,
            target_fps=target_fps,
            start_timestamp=start_timestamp,
        )

    def _forward_chunk(
        self,
        pixel_values: Tensor,
        *,
        sequence_id: str,
        latent_start: int,
        is_sequence_start: bool,
        target_fps: float,
        start_timestamp: float,
        cache_state: WanCacheState | None,
    ) -> ProgressiveVideoRAEChunkOutput:
        encoder_output, chunk = self.encode_chunk(
            pixel_values,
            sequence_id=sequence_id,
            latent_start=latent_start,
            is_sequence_start=is_sequence_start,
            target_fps=target_fps,
            start_timestamp=start_timestamp,
        )
        decoder_output = self.decoder.decode_chunk(chunk, cache_state=cache_state)
        target_pixels = pixel_values if is_sequence_start else pixel_values[:, :, 1:]
        target = target_pixels.mul(2.0).sub(1.0)
        if decoder_output.video.shape != target.shape:
            raise RuntimeError(
                "Chunk reconstruction shape mismatch: "
                f"{tuple(decoder_output.video.shape)} vs {tuple(target.shape)}"
            )
        return ProgressiveVideoRAEChunkOutput(
            reconstruction=decoder_output.video,
            target=target,
            state_chunk=chunk,
            encoder_output=encoder_output,
            decoder_output=decoder_output,
        )


    def forward(
        self,
        pixel_values: Tensor,
        *,
        endpoint: int | None = None,
        paired_previous_endpoint: int | None = None,
        cache_mode: CacheMode = "disabled",
        cache_state: WanCacheState | None = None,
        return_decoder_features: bool = False,
        sequence_id: str | None = None,
        chunk_latent_start: int | None = None,
        chunk_is_sequence_start: bool | None = None,
        chunk_target_fps: float | None = None,
        chunk_start_timestamp: float | None = None,
    ) -> (
        ProgressiveVideoRAEOutput
        | ProgressiveVideoRAEChunkOutput
        | tuple[ProgressiveVideoRAEOutput, ProgressiveVideoRAEOutput]
    ):
        if chunk_latent_start is not None:
            if (
                sequence_id is None
                or chunk_is_sequence_start is None
                or chunk_target_fps is None
                or chunk_start_timestamp is None
            ):
                raise ValueError("Chunk forward requires complete sequence metadata")
            if endpoint is not None or paired_previous_endpoint is not None:
                raise ValueError("Chunk forward cannot execute a spatial-prefix task")
            return self._forward_chunk(
                pixel_values,
                sequence_id=sequence_id,
                latent_start=chunk_latent_start,
                is_sequence_start=chunk_is_sequence_start,
                target_fps=chunk_target_fps,
                start_timestamp=chunk_start_timestamp,
                cache_state=cache_state,
            )
        encoder_output = self.encode_features(pixel_values)
        if paired_previous_endpoint is not None:
            if endpoint is None or paired_previous_endpoint != endpoint - 1:
                raise ValueError("paired prefix endpoints must be adjacent")
            if cache_mode != "disabled":
                raise ValueError("paired prefix decode requires ephemeral caches")
            projected = self._timed("projector", lambda: self.projector(encoder_output))
            current_view = (
                projected.state
                if endpoint == projected.state.full_endpoint
                else self.projector.make_prefix_view(projected.state, endpoint)
            )
            previous_view = self.projector.make_prefix_view(
                projected.state, paired_previous_endpoint
            )
            return self._decode_pair(
                pixel_values,
                encoder_output,
                previous_view,
                current_view,
                sequence_id=sequence_id,
            )
        projected = self._timed("projector", lambda: self.projector(encoder_output))
        return self._decode_projected(
            pixel_values,
            encoder_output,
            projected,
            endpoint=endpoint,
            cache_mode=cache_mode,
            cache_state=cache_state,
            return_decoder_features=return_decoder_features,
            sequence_id=sequence_id,
        )

    def _decode_pair(
        self,
        pixel_values: Tensor,
        encoder_output: PrefixEncoderOutput,
        previous_view: SpatialPrefixView,
        current_view: ProgressiveState | SpatialPrefixView,
        *,
        sequence_id: str | None,
    ) -> tuple[ProgressiveVideoRAEOutput, ProgressiveVideoRAEOutput]:
        previous = self._decode_state_view(
            pixel_values,
            encoder_output,
            previous_view,
            canonical_state=None,
            cache_mode="disabled",
            cache_state=None,
            return_decoder_features=False,
            sequence_id=sequence_id,
        )
        current = self._decode_state_view(
                pixel_values,
                encoder_output,
                current_view,
                canonical_state=(
                    current_view if isinstance(current_view, ProgressiveState) else None
                ),
                cache_mode="disabled",
                cache_state=None,
                return_decoder_features=False,
                sequence_id=sequence_id,
            )
        return previous, current

    def _decode_projected(
        self,
        pixel_values: Tensor,
        encoder_output: PrefixEncoderOutput,
        projected: ProjectorOutput,
        *,
        endpoint: int | None,
        cache_mode: CacheMode,
        cache_state: WanCacheState | None,
        return_decoder_features: bool,
        sequence_id: str | None,
    ) -> ProgressiveVideoRAEOutput:
        state = projected.state
        state_view: ProgressiveState | SpatialPrefixView = (
            state
            if endpoint is None or endpoint == state.full_endpoint
            else self.projector.make_prefix_view(state, endpoint)
        )
        return self._decode_state_view(
            pixel_values,
            encoder_output,
            state_view,
            canonical_state=state,
            cache_mode=cache_mode,
            cache_state=cache_state,
            return_decoder_features=return_decoder_features,
            sequence_id=sequence_id,
            repa_reference=projected.repa_reference,
        )

    def _decode_state_view(
        self,
        pixel_values: Tensor,
        encoder_output: PrefixEncoderOutput,
        state_view: ProgressiveState | SpatialPrefixView,
        *,
        canonical_state: ProgressiveState | None,
        cache_mode: CacheMode,
        cache_state: WanCacheState | None,
        return_decoder_features: bool,
        sequence_id: str | None,
        repa_reference: RepaReference | None = None,
    ) -> ProgressiveVideoRAEOutput:
        decoder_output = self._timed(
            "wan_forward",
            lambda: self.decoder.decode(
                state_view,
                cache_mode=cache_mode,
                cache_state=cache_state,
                sequence_id=sequence_id,
                return_features=return_decoder_features,
            ),
        )
        repa_features: RepaReference | None = None
        if return_decoder_features:
            if decoder_output.intermediate_features is None:
                raise RuntimeError("Decoder did not return the requested REPA feature map")
            repa_features = self.repa_projection(decoder_output.intermediate_features)
            if repa_reference is None:
                raise RuntimeError("Full-state decode requires a REPA reference")
        return self._make_output(
            pixel_values,
            encoder_output,
            state_view,
            canonical_state,
            decoder_output,
            repa_features=repa_features,
            repa_reference=repa_reference,
        )

    @staticmethod
    def _make_output(
        pixel_values: Tensor,
        encoder_output: PrefixEncoderOutput,
        state_view: ProgressiveState | SpatialPrefixView,
        canonical_state: ProgressiveState | None,
        decoder_output: WanDecoderOutput,
        *,
        repa_features: RepaReference | None = None,
        repa_reference: RepaReference | None = None,
    ) -> ProgressiveVideoRAEOutput:
        target = pixel_values.mul(2.0).sub(1.0)
        if decoder_output.video.shape != target.shape:
            raise RuntimeError(
                "Decoder reconstruction shape mismatch: "
                f"got {tuple(decoder_output.video.shape)}, expected {tuple(target.shape)}"
            )
        return ProgressiveVideoRAEOutput(
            reconstruction=decoder_output.video,
            target=target,
            state=canonical_state,
            state_view=state_view,
            encoder_output=encoder_output,
            decoder_output=decoder_output,
            repa_features=repa_features,
            repa_reference=repa_reference,
        )
