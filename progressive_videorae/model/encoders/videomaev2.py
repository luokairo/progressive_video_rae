from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn

from .base import VideoFoundationEncoder, clean_state_dict_keys, upstream_import_path
from ..pretrained import load_validated_pretrained
from ..types import EncoderOutput


class VideoMAEv2Encoder(VideoFoundationEncoder):
    """VideoMAEv2 adapter with explicit intermediate-layer extraction."""

    def __init__(
        self,
        checkpoint_path: str,
        *,
        source_root: str,
        variant: str = "vit_base_patch16_224",
        input_size: tuple[int, int] = (480, 768),
        num_frames: int = 16,
        patch_size: int = 16,
        tubelet_size: int = 2,
        output_layers: Sequence[int] = (4, 6, 8, 10, 12),
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = tuple(input_size)
        self.num_frames = int(num_frames)
        self.patch_size = int(patch_size)
        self.tubelet_size = int(tubelet_size)
        self.variant = variant

        with upstream_import_path(source_root):
            try:
                from models import modeling_finetune
            except ImportError as exc:
                raise ImportError(
                    "VideoMAEv2 source is unavailable. Clone the pinned repository and set encoder.source_root."
                ) from exc
            if not hasattr(modeling_finetune, variant):
                raise ValueError(f"Unknown VideoMAEv2 variant: {variant}")
            builder = getattr(modeling_finetune, variant)
            self.backbone = builder(
                pretrained=False,
                img_size=self.input_size,
                all_frames=self.num_frames,
                tubelet_size=self.tubelet_size,
                num_classes=0,
                use_mean_pooling=False,
            )

        depth = len(self.backbone.blocks)
        self.output_layers = self._validate_layers(output_layers, depth)
        self.embed_dim = int(self.backbone.embed_dim)
        self.grid_size = (
            self.num_frames // self.tubelet_size,
            self.input_size[0] // self.patch_size,
            self.input_size[1] // self.patch_size,
        )
        checkpoint = self._load_checkpoint(checkpoint_path)
        if not isinstance(checkpoint, dict):
            raise TypeError("VideoMAEv2 checkpoint root must contain a state dictionary")
        state = checkpoint
        if isinstance(state, dict):
            for wrapper in ("model", "state_dict", "module"):
                if isinstance(state.get(wrapper), dict):
                    state = state[wrapper]
                    break
        if not isinstance(state, dict):
            raise TypeError("VideoMAEv2 checkpoint wrapper must contain a state dictionary")
        cleaned = {}
        for key, value in clean_state_dict_keys(state).items():
            if key.startswith("head."):
                continue
            cleaned[key] = value
        self.load_report = load_validated_pretrained(
            self.backbone,
            cleaned,
            component="videomaev2_encoder",
            checkpoint_path=checkpoint_path,
            minimum_coverage=0.90,
            allowed_missing_patterns=("pos_embed", "*.pos_embed"),
            ignored_checkpoint_patterns=("head.*",),
        )
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(self.embed_dim, eps=1e-6, elementwise_affine=False) for _ in self.output_layers]
        )
        self.register_buffer("mean", torch.tensor((0.5, 0.5, 0.5)).view(1, 3, 1, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor((0.5, 0.5, 0.5)).view(1, 3, 1, 1, 1), persistent=False)
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
        layers = self.output_layers if output_layers is None else self._validate_layers(
            output_layers, len(self.backbone.blocks)
        )
        if tuple(pixel_values.shape[1:]) != (3, self.num_frames, *self.input_size):
            raise ValueError(f"Expected [B,3,{self.num_frames},{self.input_size[0]},{self.input_size[1]}]")
        x = (pixel_values - self.mean.to(pixel_values)) / self.std.to(pixel_values)
        context = torch.no_grad() if self.frozen else torch.enable_grad()
        with context:
            x = self.backbone.patch_embed(x)
            if self.backbone.pos_embed is not None:
                x = x + self.backbone.pos_embed.to(device=x.device, dtype=x.dtype)
            x = self.backbone.pos_drop(x)
            outputs = []
            selected = set(layers)
            for index, block in enumerate(self.backbone.blocks, start=1):
                x = block(x)
                if index in selected:
                    outputs.append(x)
        normalized = tuple(norm(tensor) for norm, tensor in zip(self.layer_norms, outputs))
        fused = torch.stack(normalized, dim=0).sum(dim=0)
        b, n, c = fused.shape
        t, h, w = self.grid_size
        if n != t * h * w:
            raise RuntimeError(f"VideoMAEv2 token count {n} does not match grid {self.grid_size}")
        return EncoderOutput(
            tokens=fused.reshape(b, t, h, w, c),
            grid_size=self.grid_size,
            layer_tokens=tuple(tensor.reshape(b, t, h, w, c) for tensor in normalized),
        )
