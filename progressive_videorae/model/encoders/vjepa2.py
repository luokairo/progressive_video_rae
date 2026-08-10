from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from .base import VideoFoundationEncoder, clean_state_dict_keys, upstream_import_path
from ..pretrained import load_validated_pretrained
from ..types import EncoderOutput, PrefixEncoderOutput, PrefixGroupFeatures


VJEPA2_MEAN = (0.485, 0.456, 0.406)
VJEPA2_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class VJEPA2Variant:
    builder: str
    depth: int
    embed_dim: int
    default_output_layers: tuple[int, ...]


VJEPA2_VARIANTS = {
    "vitl": VJEPA2Variant(
        builder="vit_large",
        depth=24,
        embed_dim=1024,
        default_output_layers=(8, 12, 16, 20, 24),
    ),
    "vitg": VJEPA2Variant(
        builder="vit_giant_xformers",
        depth=40,
        embed_dim=1408,
        default_output_layers=(8, 16, 24, 32, 40),
    ),
}


class VJEPA2Encoder(VideoFoundationEncoder):
    """Thin adapter around Meta's official V-JEPA2 VisionTransformer."""

    def __init__(
        self,
        checkpoint_path: str,
        *,
        source_root: str | None = None,
        input_size: tuple[int, int] = (480, 768),
        num_frames: int = 64,
        patch_size: int = 16,
        tubelet_size: int = 2,
        output_layers: Sequence[int] | None = None,
        variant: str = "vitl",
        freeze: bool = True,
        handle_nonsquare_inputs: bool = True,
    ) -> None:
        super().__init__()
        if variant not in VJEPA2_VARIANTS:
            raise ValueError(
                f"Unsupported V-JEPA2 variant: {variant}. "
                f"Expected one of {tuple(VJEPA2_VARIANTS)}"
            )
        variant_spec = VJEPA2_VARIANTS[variant]
        self.variant = variant
        self.input_size = tuple(input_size)
        self.num_frames = int(num_frames)
        if self.num_frames < 2 or self.num_frames % 2:
            raise ValueError("V-JEPA2 max num_frames must be an even integer >= 2")
        self.patch_size = int(patch_size)
        self.tubelet_size = int(tubelet_size)
        requested_layers = tuple(output_layers or variant_spec.default_output_layers)
        self.output_layers = self._validate_layers(requested_layers, variant_spec.depth)
        self.grid_size = (
            self.num_frames // self.tubelet_size,
            self.input_size[0] // self.patch_size,
            self.input_size[1] // self.patch_size,
        )
        if self.grid_size[1:] != (30, 48):
            raise ValueError(
                f"Full-resolution baseline requires V-JEPA2 spatial grid (30,48), got {self.grid_size[1:]}"
            )
        if not freeze:
            raise ValueError("native_vjepa_prefix_r4 requires the V-JEPA2 backbone to stay frozen")

        with upstream_import_path(source_root):
            try:
                from src.models import vision_transformer
            except ImportError as exc:
                raise ImportError(
                    "V-JEPA2 source is unavailable. Clone the pinned repository and set encoder.source_root."
                ) from exc
            try:
                builder = getattr(vision_transformer, variant_spec.builder)
            except AttributeError as exc:
                raise ImportError(
                    f"Pinned V-JEPA2 source does not provide {variant_spec.builder}"
                ) from exc
            self.backbone = builder(
                img_size=self.input_size,
                patch_size=self.patch_size,
                num_frames=self.num_frames,
                tubelet_size=self.tubelet_size,
                out_layers=[x - 1 for x in self.output_layers],
                use_rope=True,
                use_sdpa=True,
                handle_nonsquare_inputs=handle_nonsquare_inputs,
                use_activation_checkpointing=False,
            )

        if not hasattr(self.backbone, "blocks") or not hasattr(self.backbone, "embed_dim"):
            raise TypeError("V-JEPA2 backbone must expose blocks and embed_dim")
        self.depth = len(self.backbone.blocks)
        self.embed_dim = int(self.backbone.embed_dim)
        if self.depth != variant_spec.depth or self.embed_dim != variant_spec.embed_dim:
            raise RuntimeError(
                f"V-JEPA2 {variant} builder produced depth={self.depth}, embed_dim={self.embed_dim}; "
                f"expected depth={variant_spec.depth}, embed_dim={variant_spec.embed_dim}"
            )
        self.output_layers = self._validate_layers(self.output_layers, self.depth)

        checkpoint = self._load_checkpoint(checkpoint_path)
        if not isinstance(checkpoint, dict):
            raise TypeError("V-JEPA2 checkpoint root must contain a state dictionary")
        root = checkpoint.get("state_dict", checkpoint)
        state = root.get("target_encoder", root.get("encoder", root)) if isinstance(root, dict) else root
        if not isinstance(state, dict):
            raise TypeError("V-JEPA2 checkpoint encoder entry must be a state dictionary")
        self.load_report = load_validated_pretrained(
            self.backbone,
            clean_state_dict_keys(state),
            component="vjepa2_encoder",
            checkpoint_path=checkpoint_path,
            minimum_coverage=0.90,
            allowed_missing_patterns=("pos_embed", "*.pos_embed"),
            ignored_checkpoint_patterns=(
                "predictor.*",
                "*.predictor.*",
                "pos_embed",
                "*.pos_embed",
            ),
        )

        self.register_buffer("mean", torch.tensor(VJEPA2_MEAN).view(1, 3, 1, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(VJEPA2_STD).view(1, 3, 1, 1, 1), persistent=False)
        self.frozen = bool(freeze)
        if self.frozen:
            self.freeze_backbone()

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "frozen", False):
            self.backbone.eval()
        return self

    def forward(
        self,
        pixel_values: Tensor,
        output_layers: Sequence[int] | None = None,
    ) -> EncoderOutput:
        layers = (
            self.output_layers
            if output_layers is None
            else self._validate_layers(output_layers, self.depth)
        )
        if tuple(layers) != self.output_layers:
            raise ValueError("V-JEPA2 output layers are fixed at construction so the upstream out_layers match")
        if pixel_values.ndim != 5 or tuple(pixel_values.shape[1:2] + pixel_values.shape[3:]) != (
            3,
            *self.input_size,
        ):
            raise ValueError(
                f"Expected [B,3,T,{self.input_size[0]},{self.input_size[1]}], "
                f"got {tuple(pixel_values.shape)}"
            )
        frames = int(pixel_values.shape[2])
        if frames < 2 or frames > self.num_frames or frames % self.tubelet_size:
            raise ValueError(
                f"V-JEPA2 prefix length must be divisible by {self.tubelet_size} "
                f"and within [2,{self.num_frames}], got {frames}"
            )
        x = (pixel_values - self.mean.to(pixel_values)) / self.std.to(pixel_values)
        context = torch.no_grad() if self.frozen else torch.enable_grad()
        with context:
            outputs = self.backbone(x)
        if not isinstance(outputs, (tuple, list)) or len(outputs) != len(self.output_layers):
            raise RuntimeError("V-JEPA2 did not return the configured intermediate layers")
        official_layers = tuple(outputs)
        tokens = official_layers[-1]
        b, n, c = tokens.shape
        t = frames // self.tubelet_size
        _, h, w = self.grid_size
        if n != t * h * w:
            raise RuntimeError(f"V-JEPA2 token count {n} does not match grid {self.grid_size}")
        grid = tokens.reshape(b, t, h, w, c)
        layer_grids = tuple(tensor.reshape(b, t, h, w, c) for tensor in official_layers)
        return EncoderOutput(tokens=grid, grid_size=(t, h, w), layer_tokens=layer_grids)

    def encode_prefixes(self, pixel_values: Tensor) -> PrefixEncoderOutput:
        """Encode one unmodified full-attention V-JEPA prefix per RAE latent frame."""

        if pixel_values.ndim != 5 or tuple(pixel_values.shape[1:2] + pixel_values.shape[3:]) != (
            3,
            *self.input_size,
        ):
            raise ValueError(
                f"Expected [B,3,F,{self.input_size[0]},{self.input_size[1]}], "
                f"got {tuple(pixel_values.shape)}"
            )
        rgb_frames = int(pixel_values.shape[2])
        if rgb_frames < 1 or (rgb_frames - 1) % 4:
            raise ValueError("Production RGB length must satisfy F=1+4*n")

        groups: list[PrefixGroupFeatures] = []
        num_latents = 1 + (rgb_frames - 1) // 4
        for latent_index in range(num_latents):
            end = 4 * latent_index
            if latent_index == 0:
                prefix = pixel_values[:, :, :1].repeat(1, 1, 2, 1, 1)
                source_start = source_end = 0
                selected_tubelets = 1
                latent_type = "image_first"
            elif end + 2 <= self.num_frames:
                prefix = torch.cat(
                    (pixel_values[:, :, :1], pixel_values[:, :, : end + 1]), dim=2
                )
                source_start, source_end = 0, end
                selected_tubelets = 2
                latent_type = "video_group"
            else:
                source_start = end - self.num_frames + 1
                source_end = end
                prefix = pixel_values[:, :, source_start : end + 1]
                selected_tubelets = 2
                latent_type = "video_group"

            encoded = self(prefix)
            groups.append(
                PrefixGroupFeatures(
                    tokens=encoded.tokens[:, -selected_tubelets:],
                    layer_tokens=tuple(
                        layer[:, -selected_tubelets:] for layer in encoded.layer_tokens
                    ),
                    latent_type=latent_type,
                    source_start=source_start,
                    source_end=source_end,
                    input_frames=int(prefix.shape[2]),
                )
            )

        return PrefixEncoderOutput(
            groups=tuple(groups),
            spatial_grid=self.grid_size[1:],
            embed_dim=self.embed_dim,
            max_context_frames=self.num_frames,
        )
