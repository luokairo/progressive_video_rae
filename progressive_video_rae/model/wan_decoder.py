from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import sys
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .pretrained import PretrainedLoadReport, load_validated_pretrained
from .types import ProgressiveState


# Copied from Wan2.2 vae2_2.py
WAN22_LATENT_MEAN = (
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
    -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
    -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
    -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
    -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
    0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
)
WAN22_LATENT_STD = (
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
    0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
    0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
    0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
    0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
)

CacheMode = Literal["disabled", "reset", "reuse"]


@dataclass
class WanCacheState:
    features: list[Any]
    frames_seen: int = 0


@dataclass
class WanDecoderOutput:
    video: Tensor
    cache_state: WanCacheState | None
    intermediate_features: Tensor | None = None


@dataclass
class WanLoadReport:
    decoder: PretrainedLoadReport
    pre_decoder: PretrainedLoadReport

    @property
    def ready(self) -> bool:
        return self.decoder.ready and self.pre_decoder.ready

    def assert_ready(self) -> None:
        self.pre_decoder.assert_ready()
        self.decoder.assert_ready()

    def to_dict(self) -> dict:
        return {
            "ready": self.ready,
            "pre_decoder": self.pre_decoder.to_dict(),
            "decoder": self.decoder.to_dict(),
        }


class WanVideoDecoder(nn.Module):
    """Trainable Wan2.2 decoder with temporal upsampling disabled and explicit cache state."""

    def __init__(
        self,
        checkpoint_path: str | None = None,
        *,
        source_root: str | None = None,
        latent_channels: int = 48,
        base_dim: int = 256,
        output_size: tuple[int, int] = (480, 768),
        load_pretrained: bool = True,
    ) -> None:
        super().__init__()
        if latent_channels != 48:
            raise ValueError("Wan2.2 state compatibility requires 48 latent channels")
        source_root = source_root or os.environ.get("PVR_WAN_SOURCE_ROOT", "/share/project/lgy/Wan2.2")
        module_path = Path(source_root).expanduser().resolve() / "wan/modules/vae2_2.py"
        if not module_path.is_file():
            raise FileNotFoundError(f"Pinned Wan2.2 vae2_2.py not found: {module_path}")
        module_name = "_progressive_video_rae_wan2_2_vae"
        module = sys.modules.get(module_name)
        if module is None:
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load Wan2.2 module spec from {module_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        self.pre_decoder = module.CausalConv3d(latent_channels, latent_channels, 1)
        self.decoder = module.Decoder3d(
            dim=base_dim,
            z_dim=latent_channels,
            dim_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            attn_scales=[],
            temperal_upsample=[False, False, False],
            dropout=0.0,
        )
        self._count_conv3d = module.count_conv3d
        self._unpatchify = module.unpatchify

        self.output_size = tuple(output_size)
        self.gradient_checkpointing = False
        self.register_buffer(
            "latent_mean", torch.tensor(WAN22_LATENT_MEAN).view(1, 48, 1, 1, 1), persistent=True
        )
        self.register_buffer(
            "latent_std", torch.tensor(WAN22_LATENT_STD).view(1, 48, 1, 1, 1), persistent=True
        )
        self.load_report: WanLoadReport | None = None
        if load_pretrained:
            if checkpoint_path is None:
                raise ValueError("checkpoint_path is required when load_pretrained=True")
            self.load_report = self.load_pretrained(checkpoint_path)

    def load_pretrained(self, checkpoint_path: str) -> WanLoadReport:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Wan2.2 VAE checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint
        if isinstance(state, dict):
            for wrapper in ("state_dict", "model"):
                if isinstance(state.get(wrapper), dict):
                    state = state[wrapper]
                    break
        if not isinstance(state, dict):
            raise TypeError("Wan2.2 checkpoint root must contain a state dictionary")
        state = {key.removeprefix("module.").removeprefix("model."): value for key, value in state.items()}
        decoder_state = {key[len("decoder.") :]: value for key, value in state.items() if key.startswith("decoder.")}
        conv_state = {key[len("conv2.") :]: value for key, value in state.items() if key.startswith("conv2.")}
        pre_decoder_report = load_validated_pretrained(
            self.pre_decoder,
            conv_state,
            component="wan2.2_pre_decoder",
            checkpoint_path=checkpoint_path,
            minimum_coverage=1.0,
            required_groups={"latent_input_adapter": ("weight",)},
        )
        decoder_report = load_validated_pretrained(
            self.decoder,
            decoder_state,
            component="wan2.2_decoder",
            checkpoint_path=checkpoint_path,
            minimum_coverage=0.98,
            required_groups={
                "decoder_input": ("conv1.*",),
                "decoder_middle": ("middle.*",),
                "spatial_upsample_0": ("upsamples.0.*.resample.*",),
                "spatial_upsample_1": ("upsamples.1.*.resample.*",),
                "spatial_upsample_2": ("upsamples.2.*.resample.*",),
                "decoder_output": ("head.2.*",),
            },
            ignored_checkpoint_patterns=("*time_conv*",),
        )
        report = WanLoadReport(decoder=decoder_report, pre_decoder=pre_decoder_report)
        report.assert_ready()
        return report

    def _new_cache(self) -> WanCacheState:
        return WanCacheState(features=[None] * int(self._count_conv3d(self.decoder)), frames_seen=0)

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = bool(enabled)

    def _prepare_latent(self, state: ProgressiveState | Tensor) -> Tensor:
        tokens = state.tokens if isinstance(state, ProgressiveState) else state
        if tokens.ndim != 5 or tokens.shape[-1] != 48:
            raise ValueError(f"Expected state [B,T,H,W,48], got {tuple(tokens.shape)}")
        latent = tokens.permute(0, 4, 1, 2, 3)
        latent = latent * self.latent_std.to(latent) + self.latent_mean.to(latent)
        return self.pre_decoder(latent)

    def decode(
        self,
        state: ProgressiveState | Tensor,
        *,
        prefix_len: int = 64,
        cache_mode: CacheMode = "disabled",
        cache_state: WanCacheState | None = None,
        return_features: bool = False,
    ) -> WanDecoderOutput:
        if cache_mode not in ("disabled", "reset", "reuse"):
            raise ValueError(f"Unsupported cache_mode: {cache_mode}")
        if isinstance(state, ProgressiveState) and prefix_len != state.prefix_len:
            raise ValueError("prefix_len does not match ProgressiveState.prefix_len")
        latent = self._prepare_latent(state)
        captured: list[Tensor] = []
        hook = None
        if return_features:
            hook = self.decoder.upsamples[0].register_forward_hook(
                lambda _module, _inputs, output: captured.append(output)
            )
        try:
            if cache_mode == "disabled":
                if self.gradient_checkpointing and self.training:
                    decoded = checkpoint(
                        lambda value: self.decoder(
                            value, feat_cache=None, feat_idx=[0], first_chunk=True
                        ),
                        latent,
                        use_reentrant=False,
                    )
                else:
                    decoded = self.decoder(latent, feat_cache=None, feat_idx=[0], first_chunk=True)
                result_cache = None
            else:
                if cache_mode == "reset":
                    cache_state = self._new_cache()
                elif cache_state is None:
                    raise ValueError("cache_mode='reuse' requires cache_state from a prior decode call")
                outputs = []
                for frame_index in range(latent.shape[2]):
                    feat_index = [0]
                    output = self.decoder(
                        latent[:, :, frame_index : frame_index + 1],
                        feat_cache=cache_state.features,
                        feat_idx=feat_index,
                        first_chunk=cache_state.frames_seen == 0,
                    )
                    outputs.append(output)
                    cache_state.frames_seen += 1
                decoded = torch.cat(outputs, dim=2)
                result_cache = cache_state
        finally:
            if hook is not None:
                hook.remove()
        video = self._unpatchify(decoded, patch_size=2)
        if tuple(video.shape[-2:]) != self.output_size:
            raise RuntimeError(f"Wan decoder produced {tuple(video.shape[-2:])}, expected {self.output_size}")
        features = torch.cat(captured, dim=2) if captured else None
        return WanDecoderOutput(video=video, cache_state=result_cache, intermediate_features=features)

    forward = decode
