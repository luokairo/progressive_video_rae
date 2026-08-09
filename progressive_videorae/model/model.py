from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn
from torch.nn import functional as F

from .encoders.base import VideoFoundationEncoder
from .projector import CausalFrequencyProjector
from .types import PrefixEncoderOutput, ProgressiveState, assert_video_tensor
from .wan_decoder import CacheMode, WanCacheState, WanDecoderOutput, WanVideoDecoder


@dataclass
class ProgressiveVideoRAEOutput:
    reconstruction: Tensor
    target: Tensor
    state: ProgressiveState
    encoder_output: PrefixEncoderOutput
    decoder_output: WanDecoderOutput
    repa_features: Tensor | None = None
    repa_reference: Tensor | None = None


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
        self.repa_projection = nn.Conv3d(decoder_feature_dim, encoder_dim, kernel_size=1)
        self.projector.contract.assert_compatible(self.decoder.contract)

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

    def encode(
        self, pixel_values: Tensor, prefix_len: int = 64
    ) -> tuple[PrefixEncoderOutput, ProgressiveState]:
        frames = int(pixel_values.shape[2]) if pixel_values.ndim == 5 else -1
        if frames < 1 or (frames - 1) % 4:
            raise ValueError("ProgressiveVideoRAE requires F=1+4*n RGB frames")
        assert_video_tensor(pixel_values, frames=frames, height=480, width=768)
        encode_prefixes = getattr(self.encoder, "encode_prefixes", None)
        if encode_prefixes is None:
            raise TypeError("Encoder does not implement native causal prefix extraction")
        encoder_output = encode_prefixes(pixel_values)
        state = self.projector(encoder_output, prefix_len=prefix_len)
        return encoder_output, state

    def forward(
        self,
        pixel_values: Tensor,
        *,
        prefix_len: int = 64,
        cache_mode: CacheMode = "disabled",
        cache_state: WanCacheState | None = None,
        return_decoder_features: bool = False,
    ) -> ProgressiveVideoRAEOutput:
        encoder_output, state = self.encode(pixel_values, prefix_len=prefix_len)
        decoder_output = self.decoder.decode(
            state,
            prefix_len=prefix_len,
            cache_mode=cache_mode,
            cache_state=cache_state,
            return_features=return_decoder_features,
        )
        repa_features = None
        repa_reference = None
        if return_decoder_features:
            if decoder_output.intermediate_features is None:
                raise RuntimeError("Decoder did not return the requested REPA feature map")
            features = decoder_output.intermediate_features
            if (features.shape[2] - 1) % 4:
                raise RuntimeError("Decoder REPA time length must satisfy 1+4*n")
            grouped = [features[:, :, :1]]
            if features.shape[2] > 1:
                tail = features[:, :, 1:]
                b, c, tail_t, h, w = tail.shape
                tail = tail.reshape(b, c, tail_t // 4, 4, h, w).mean(dim=3)
                grouped.append(tail)
            temporal = torch.cat(grouped, dim=2)
            pooled = F.avg_pool3d(temporal, kernel_size=(1, 2, 2))
            repa_features = self.repa_projection(pooled).permute(0, 2, 3, 4, 1)
            assert state.metadata is not None
            repa_reference = state.metadata["repa_targets"]
        return ProgressiveVideoRAEOutput(
            reconstruction=decoder_output.video,
            target=pixel_values.mul(2.0).sub(1.0),
            state=state,
            encoder_output=encoder_output,
            decoder_output=decoder_output,
            repa_features=repa_features,
            repa_reference=repa_reference,
        )
