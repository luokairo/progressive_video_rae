from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn

from .base import VideoFoundationEncoder, clean_state_dict_keys, upstream_import_path
from ..pretrained import load_validated_pretrained
from ..types import EncoderOutput


VJEPA2_MEAN = (0.485, 0.456, 0.406)
VJEPA2_STD = (0.229, 0.224, 0.225)


class VJEPA2Encoder(VideoFoundationEncoder):
    """Thin adapter around Meta's official V-JEPA2 VisionTransformer."""

    def __init__(
        self,
        checkpoint_path: str,
        *,
        source_root: str | None = None,
        input_size: tuple[int, int] = (480, 768),
        num_frames: int = 16,
        patch_size: int = 16,
        tubelet_size: int = 2,
        output_layers: Sequence[int] = (8, 12, 16, 20, 24),
        freeze: bool = True,
        handle_nonsquare_inputs: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = tuple(input_size)
        self.num_frames = int(num_frames)
        self.patch_size = int(patch_size)
        self.tubelet_size = int(tubelet_size)
        self.output_layers = self._validate_layers(output_layers, 24)
        self.grid_size = (
            self.num_frames // self.tubelet_size,
            self.input_size[0] // self.patch_size,
            self.input_size[1] // self.patch_size,
        )
        if self.grid_size != (8, 30, 48):
            raise ValueError(f"Full-resolution v1 requires V-JEPA2 grid (8,30,48), got {self.grid_size}")

        with upstream_import_path(source_root):
            try:
                from src.models.vision_transformer import vit_large
            except ImportError as exc:
                raise ImportError(
                    "V-JEPA2 source is unavailable. Clone the pinned repository and set encoder.source_root."
                ) from exc
            self.backbone = vit_large(
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

        checkpoint = self._load_checkpoint(checkpoint_path)
        if not isinstance(checkpoint, dict):
            raise TypeError("V-JEPA2 checkpoint root must contain a state dictionary")
        root = checkpoint.get("state_dict", checkpoint)
        state = root.get("encoder", root.get("target_encoder", root)) if isinstance(root, dict) else root
        if not isinstance(state, dict):
            raise TypeError("V-JEPA2 checkpoint encoder entry must be a state dictionary")
        depth = len(self.backbone.blocks)
        self.load_report = load_validated_pretrained(
            self.backbone,
            clean_state_dict_keys(state),
            component="vjepa2_encoder",
            checkpoint_path=checkpoint_path,
            minimum_coverage=0.98,
            required_groups={
                "patch_embedding": ("patch_embed.*",),
                "first_transformer_block": ("blocks.0.*",),
                "last_transformer_block": (f"blocks.{depth - 1}.*",),
            },
            allowed_missing_patterns=("pos_embed", "*.pos_embed"),
            ignored_checkpoint_patterns=(
                "predictor.*",
                "*.predictor.*",
                "pos_embed",
                "*.pos_embed",
            ),
        )

        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(1024, eps=1e-6, elementwise_affine=False) for _ in self.output_layers]
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
        layers = self.output_layers if output_layers is None else self._validate_layers(output_layers, 24)
        if tuple(layers) != self.output_layers:
            raise ValueError("V-JEPA2 output layers are fixed at construction so the upstream out_layers match")
        if tuple(pixel_values.shape[1:]) != (3, self.num_frames, *self.input_size):
            raise ValueError(f"Expected [B,3,{self.num_frames},{self.input_size[0]},{self.input_size[1]}]")
        x = (pixel_values - self.mean.to(pixel_values)) / self.std.to(pixel_values)
        context = torch.no_grad() if self.frozen else torch.enable_grad()
        with context:
            outputs = self.backbone(x)
        if not isinstance(outputs, (tuple, list)) or len(outputs) != len(self.output_layers):
            raise RuntimeError("V-JEPA2 did not return the configured intermediate layers")
        normalized = tuple(norm(tensor) for norm, tensor in zip(self.layer_norms, outputs))
        fused = torch.stack(normalized, dim=0).sum(dim=0)
        b, n, c = fused.shape
        t, h, w = self.grid_size
        if n != t * h * w:
            raise RuntimeError(f"V-JEPA2 token count {n} does not match grid {self.grid_size}")
        grid = fused.reshape(b, t, h, w, c)
        layer_grids = tuple(tensor.reshape(b, t, h, w, c) for tensor in normalized)
        return EncoderOutput(tokens=grid, grid_size=self.grid_size, layer_tokens=layer_grids)
