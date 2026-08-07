from __future__ import annotations

import torch
from torch import Tensor, nn

from .progressive_sets import (
    LAYOUT_VERSION,
    SET_SIZES,
    build_causal_attention_mask,
    build_prefix_mask,
    build_progressive_layout,
)
from .types import EncoderOutput, ProgressiveState


class CausalFrequencyProjector(nn.Module):
    """Per-frame transformer with deterministic low-to-high set causality."""

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        output_dim: int = 48,
        *,
        num_frames: int = 16,
        input_frames: int = 8,
        height: int = 30,
        width: int = 48,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        layout_version: str = LAYOUT_VERSION,
        set_sizes: tuple[int, ...] = SET_SIZES,
    ) -> None:
        super().__init__()
        if num_frames != input_frames * 2:
            raise ValueError("v1 projector requires temporal upsampling 8 -> 16")
        self.num_frames = num_frames
        self.input_frames = input_frames
        self.height = height
        self.width = width
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        layout = build_progressive_layout(height, width, set_sizes, layout_version)
        set_ids = layout.set_ids_tensor()
        self.layout_version = layout.version
        self.layout_checksum = layout.checksum
        self.register_buffer("set_ids", set_ids, persistent=True)
        self.register_buffer("set_sizes", torch.tensor(layout.set_sizes, dtype=torch.long), persistent=True)
        self.register_buffer(
            "causal_attention_mask", build_causal_attention_mask(set_ids), persistent=False
        )

        self.temporal_upsample = nn.ConvTranspose3d(
            input_dim,
            hidden_dim,
            kernel_size=(2, 1, 1),
            stride=(2, 1, 1),
        )
        token_count = height * width
        self.spatial_embedding = nn.Parameter(torch.zeros(1, token_count, hidden_dim))
        self.temporal_embedding = nn.Parameter(torch.zeros(1, num_frames, 1, hidden_dim))
        self.set_embedding = nn.Embedding(len(layout.set_sizes), hidden_dim)
        self.frame_parity_embedding = nn.Embedding(2, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=int(hidden_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, 1, output_dim))
        nn.init.trunc_normal_(self.spatial_embedding, std=0.02)
        nn.init.trunc_normal_(self.temporal_embedding, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, features: EncoderOutput | Tensor, prefix_len: int = 64) -> ProgressiveState:
        x = features.tokens if isinstance(features, EncoderOutput) else features
        expected = (self.input_frames, self.height, self.width)
        if x.ndim != 5 or tuple(x.shape[1:4]) != expected:
            raise ValueError(f"Expected VFM grid [B,{expected[0]},{expected[1]},{expected[2]},C]")
        x = x.permute(0, 4, 1, 2, 3)
        x = self.temporal_upsample(x).permute(0, 2, 3, 4, 1)
        b, t, h, w, c = x.shape
        flat = x.reshape(b, t, h * w, c)
        set_embedding = self.set_embedding(self.set_ids.reshape(-1)).view(1, 1, h * w, c)
        parity = self.frame_parity_embedding(
            torch.arange(t, device=x.device, dtype=torch.long) % 2
        ).view(1, t, 1, c)
        flat = flat + self.spatial_embedding + self.temporal_embedding + set_embedding + parity
        flat = flat.reshape(b * t, h * w, c)
        mask = self.causal_attention_mask.to(device=flat.device)
        flat = self.transformer(flat, mask=mask)
        flat = self.output_projection(self.output_norm(flat)).reshape(b, t, h, w, self.output_dim)

        active = build_prefix_mask(self.set_ids, prefix_len).view(1, 1, h, w, 1)
        tokens = torch.where(active, flat, self.mask_token.to(flat))
        return ProgressiveState(
            tokens=tokens,
            flat_tokens=tokens.reshape(b, t, h * w, self.output_dim),
            set_ids=self.set_ids,
            set_sizes=self.set_sizes,
            prefix_len=int(prefix_len),
            layout_version=self.layout_version,
            layout_checksum=self.layout_checksum,
            metadata={"unmasked_tokens": flat},
        )
