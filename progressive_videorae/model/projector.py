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
from .types import (
    IMAGE_FIRST_ID,
    VIDEO_GROUP_ID,
    PrefixEncoderOutput,
    PrefixGroupFeatures,
    ProgressiveState,
    StateContract,
)


class CausalFrequencyProjector(nn.Module):
    """Map native V-JEPA prefix features to typed, set-causal production states."""

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        output_dim: int = 48,
        *,
        num_frames: int = 5,
        input_frames: int | None = None,
        num_input_layers: int = 5,
        max_context_frames: int = 64,
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
        del input_frames  # retained in the signature to make legacy config errors explicit at forward
        if output_dim != 48:
            raise ValueError("native_vjepa_prefix_r4 fixes the production state to 48 channels")
        if num_frames < 1:
            raise ValueError("num_frames must be positive")
        if num_input_layers < 1:
            raise ValueError("num_input_layers must be positive")
        self.num_frames = int(num_frames)
        self.num_input_layers = int(num_input_layers)
        self.max_context_frames = int(max_context_frames)
        self.height = int(height)
        self.width = int(width)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)

        layout = build_progressive_layout(height, width, set_sizes, layout_version)
        self.layout_version = layout.version
        self.layout_checksum = layout.checksum
        self.register_buffer("set_ids", layout.set_ids_tensor(), persistent=True)
        self.register_buffer(
            "set_sizes", torch.tensor(layout.set_sizes, dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "causal_attention_mask",
            build_causal_attention_mask(self.set_ids),
            persistent=False,
        )
        self.contract = StateContract(
            height=height,
            width=width,
            channels=output_dim,
            num_sets=len(layout.set_sizes),
            max_encoder_context_frames=max_context_frames,
        )

        self.layer_mix_logits = nn.Parameter(torch.zeros(num_input_layers))
        self.first_projection = nn.Linear(input_dim, hidden_dim)
        self.video_projection = nn.Linear(input_dim, hidden_dim)
        self.video_pair_norm = nn.LayerNorm(hidden_dim)
        self.video_pair_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.spatial_embedding = nn.Parameter(
            torch.zeros(1, height * width, hidden_dim)
        )
        self.set_embedding = nn.Embedding(len(layout.set_sizes), hidden_dim)
        self.type_embedding = nn.Embedding(2, hidden_dim)

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
        self.state_norm = nn.LayerNorm(output_dim, elementwise_affine=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, 1, output_dim))

        nn.init.trunc_normal_(self.spatial_embedding, std=0.02)
        nn.init.trunc_normal_(self.type_embedding.weight, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def _mix_layers(self, group: PrefixGroupFeatures) -> Tensor:
        layers = group.layer_tokens or (group.tokens,)
        if len(layers) != self.num_input_layers:
            raise ValueError(
                f"Expected {self.num_input_layers} V-JEPA layers, got {len(layers)}"
            )
        weights = self.layer_mix_logits.softmax(dim=0).to(layers[0])
        stacked = torch.stack(layers, dim=0)
        return (stacked * weights.view(-1, 1, 1, 1, 1, 1)).sum(dim=0)

    def _reduce_group(self, group: PrefixGroupFeatures, mixed: Tensor) -> tuple[Tensor, int]:
        if group.latent_type == "image_first":
            if mixed.shape[1] != 1:
                raise ValueError("image_first must contain exactly one selected tubelet")
            return self.first_projection(mixed[:, 0]), IMAGE_FIRST_ID
        if group.latent_type != "video_group":
            raise ValueError(f"Unsupported latent type: {group.latent_type}")
        if mixed.shape[1] != 2:
            raise ValueError("video_group must contain exactly two selected tubelets")

        x = self.video_projection(mixed)
        b, pair, h, w, c = x.shape
        x = x.permute(0, 2, 3, 1, 4).reshape(b * h * w, pair, c)
        normalized = self.video_pair_norm(x)
        attended, _ = self.video_pair_attention(
            normalized, normalized, normalized, need_weights=False
        )
        x = (x + attended).mean(dim=1).reshape(b, h, w, c)
        return x, VIDEO_GROUP_ID

    def forward(
        self, features: PrefixEncoderOutput, prefix_len: int = 64
    ) -> ProgressiveState:
        if not isinstance(features, PrefixEncoderOutput):
            raise TypeError(
                "native_vjepa_prefix_r4 projector requires PrefixEncoderOutput; "
                "legacy 8→16 features are a different StateContract"
            )
        if features.spatial_grid != (self.height, self.width):
            raise ValueError(
                f"Expected spatial grid {(self.height, self.width)}, got {features.spatial_grid}"
            )
        if features.max_context_frames != self.max_context_frames:
            raise ValueError("Encoder/projector max context contract mismatch")

        reduced: list[Tensor] = []
        repa_targets: list[Tensor] = []
        latent_type_ids: list[int] = []
        windows: list[tuple[int, int, int]] = []
        for group in features.groups:
            mixed = self._mix_layers(group)
            group_state, latent_type_id = self._reduce_group(group, mixed)
            reduced.append(group_state)
            repa_targets.append(mixed.mean(dim=1))
            latent_type_ids.append(latent_type_id)
            windows.append((group.source_start, group.source_end, group.input_frames))

        x = torch.stack(reduced, dim=1)
        b, t, h, w, c = x.shape
        if t != features.num_latents:
            raise RuntimeError("Projector lost a prefix group")
        flat = x.reshape(b, t, h * w, c)
        set_embedding = self.set_embedding(self.set_ids.reshape(-1)).view(
            1, 1, h * w, c
        )
        type_ids = torch.tensor(latent_type_ids, device=x.device, dtype=torch.long)
        type_embedding = self.type_embedding(type_ids).view(1, t, 1, c)
        flat = flat + self.spatial_embedding + set_embedding + type_embedding
        flat = flat.reshape(b * t, h * w, c)
        flat = self.transformer(
            flat, mask=self.causal_attention_mask.to(device=flat.device)
        )
        canonical = self.state_norm(
            self.output_projection(self.output_norm(flat))
        ).reshape(b, t, h, w, self.output_dim)

        active = build_prefix_mask(self.set_ids, prefix_len).view(1, 1, h, w, 1)
        tokens = torch.where(active, canonical, self.mask_token.to(canonical))
        return ProgressiveState(
            tokens=tokens,
            flat_tokens=tokens.reshape(b, t, h * w, self.output_dim),
            set_ids=self.set_ids,
            set_sizes=self.set_sizes,
            prefix_len=int(prefix_len),
            layout_version=self.layout_version,
            layout_checksum=self.layout_checksum,
            metadata={
                "unmasked_tokens": canonical,
                "repa_targets": torch.stack(repa_targets, dim=1),
                "prefix_windows": tuple(windows),
            },
            latent_types=type_ids,
            contract=self.contract,
        )
