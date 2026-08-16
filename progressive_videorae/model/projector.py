from __future__ import annotations

import torch
from torch import Tensor, nn

from .progressive_sets import LAYOUT_CHECKSUM, LAYOUT_VERSION, SET_SIZES, build_progressive_layout
from .types import (
    IMAGE_FIRST_ID,
    VIDEO_GROUP_ID,
    PrefixEncoderOutput,
    PrefixGroupFeatures,
    ProgressiveState,
    ProjectorOutput,
    RepaReference,
    SpatialPrefixView,
    StateContract,
)


class IdentityInitializedTemporalAttentionPool(nn.Module):
    """Pool one same-spatial temporal group with a normalized mean at init."""

    def __init__(
        self,
        dim: int,
        *,
        group_size: int = 2,
        num_heads: int = 16,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.group_size = int(group_size)
        self.num_heads = int(num_heads)
        if self.group_size <= 0:
            raise ValueError("Temporal pooling group_size must be positive")
        if self.num_heads <= 0 or self.dim % self.num_heads:
            raise ValueError(
                f"Temporal pooling dim={self.dim} must be divisible by "
                f"num_heads={self.num_heads}"
            )
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim**-0.5

        self.norm_k = nn.LayerNorm(self.dim)
        self.query = nn.Parameter(torch.zeros(1, self.num_heads, 1, self.head_dim))
        self.key = nn.Linear(self.dim, self.dim)
        self.value = nn.Linear(self.dim, self.dim)
        self.proj = nn.Linear(self.dim, self.dim)
        self.time_bias = nn.Parameter(torch.zeros(self.num_heads, self.group_size))
        self.norm_out = nn.LayerNorm(self.dim, elementwise_affine=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.ones_(self.norm_k.weight)
        nn.init.zeros_(self.norm_k.bias)
        nn.init.zeros_(self.query)
        for layer in (self.key, self.value, self.proj):
            nn.init.eye_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.time_bias)

    def forward(
        self,
        x: Tensor,
        *,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if x.ndim != 5:
            raise ValueError(f"Expected temporal features [B,P,H,W,C], got {tuple(x.shape)}")
        batch, group_size, height, width, channels = x.shape
        if group_size != self.group_size or channels != self.dim:
            raise ValueError(
                "Temporal pooling shape mismatch: "
                f"expected P={self.group_size}, C={self.dim}; "
                f"got P={group_size}, C={channels}"
            )

        grouped = x.permute(0, 2, 3, 1, 4).reshape(
            batch * height * width, group_size, channels
        )
        locations = grouped.shape[0]
        key = self.key(self.norm_k(grouped)).reshape(
            locations, group_size, self.num_heads, self.head_dim
        )
        value = self.value(grouped).reshape(
            locations, group_size, self.num_heads, self.head_dim
        )
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        query = self.query.expand(locations, -1, -1, -1)
        logits = (query * key).sum(dim=-1) * self.scale
        logits = logits + self.time_bias.to(device=logits.device, dtype=logits.dtype)[None]
        attention = logits.softmax(dim=-1)
        pooled = (attention.unsqueeze(-1) * value).sum(dim=2).reshape(locations, channels)
        pooled = self.norm_out(self.proj(pooled))
        pooled = pooled.reshape(batch, height, width, channels)
        if not return_attention:
            return pooled
        return pooled, attention.reshape(
            batch, height, width, self.num_heads, group_size
        )


class CausalFrequencyProjector(nn.Module):
    """Map native V-JEPA prefix features to the v3 progressive state."""

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
        spatial_attention_mode: str = "set_causal",
        layer_fusion: str = "learned_softmax",
        layer_fusion_norm: str = "none",
        temporal_pooling: str = "hidden_dim_attention",
        temporal_pooling_heads: int = 16,
        layout_version: str = LAYOUT_VERSION,
        set_sizes: tuple[int, ...] = SET_SIZES,
    ) -> None:
        super().__init__()
        del num_frames, input_frames
        if output_dim != 48:
            raise ValueError("candidate v3 fixes the state channel width at 48")
        if tuple(set_sizes) != SET_SIZES:
            raise ValueError("candidate v3 requires 48 equal sets of 30 tokens")
        if spatial_attention_mode not in ("set_causal", "full"):
            raise ValueError(
                "spatial_attention_mode must be 'set_causal' or 'full', "
                f"got {spatial_attention_mode!r}"
            )
        if layer_fusion not in ("learned_softmax", "fixed_sum"):
            raise ValueError(
                "layer_fusion must be 'learned_softmax' or 'fixed_sum', "
                f"got {layer_fusion!r}"
            )
        if layer_fusion_norm not in ("none", "non_affine_layer_norm"):
            raise ValueError(
                "layer_fusion_norm must be 'none' or 'non_affine_layer_norm', "
                f"got {layer_fusion_norm!r}"
            )
        if temporal_pooling not in ("hidden_dim_attention", "input_dim_attention"):
            raise ValueError(
                "temporal_pooling must be 'hidden_dim_attention' or "
                f"'input_dim_attention', got {temporal_pooling!r}"
            )
        self.num_input_layers = int(num_input_layers)
        self.max_context_frames = int(max_context_frames)
        self.height = int(height)
        self.width = int(width)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.spatial_attention_mode = spatial_attention_mode
        self.layer_fusion = layer_fusion
        self.layer_fusion_norm_mode = layer_fusion_norm
        self.temporal_pooling = temporal_pooling
        self.temporal_pooling_heads = int(temporal_pooling_heads)

        layout = build_progressive_layout(height, width, set_sizes, layout_version)
        self.layout_version = layout.version
        self.layout_checksum = layout.checksum
        traversal = torch.tensor(layout.traversal, dtype=torch.long)
        if layout.checksum != LAYOUT_CHECKSUM:
            raise ValueError(
                f"candidate v3 layout checksum mismatch: {layout.checksum} != {LAYOUT_CHECKSUM}"
            )
        inverse = torch.empty_like(traversal)
        inverse[traversal] = torch.arange(traversal.numel())
        self.register_buffer("fps_permutation", traversal, persistent=True)
        self.register_buffer("inverse_fps_permutation", inverse, persistent=True)
        self.register_buffer("set_ids", layout.set_ids_tensor(), persistent=True)
        canonical_set_ids = torch.arange(48).repeat_interleave(30)
        self.register_buffer("canonical_set_ids", canonical_set_ids, persistent=False)
        self.register_buffer(
            "causal_attention_mask",
            canonical_set_ids[None, :] > canonical_set_ids[:, None],
            persistent=False,
        )
        self.contract = StateContract(
            height=height,
            width=width,
            channels=output_dim,
            num_sets=48,
            tokens_per_set=30,
            layout_version=layout.version,
            max_encoder_context_frames=max_context_frames,
            layout_checksum=layout.checksum,
        )

        if self.layer_fusion == "learned_softmax":
            self.layer_mix_logits = nn.Parameter(torch.zeros(num_input_layers))
        if self.layer_fusion_norm_mode == "non_affine_layer_norm":
            self.layer_fusion_norm = nn.LayerNorm(input_dim, elementwise_affine=False)
        else:
            self.layer_fusion_norm = nn.Identity()
        self.first_projection = nn.Linear(input_dim, hidden_dim)
        self.video_projection = nn.Linear(input_dim, hidden_dim)
        if self.temporal_pooling == "hidden_dim_attention":
            self.video_phase_embedding = nn.Parameter(torch.zeros(2, hidden_dim))
            self.video_query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            self.video_pair_norm = nn.LayerNorm(hidden_dim)
            self.video_pair_attention = nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=dropout, batch_first=True
            )
        else:
            self.video_temporal_pool = IdentityInitializedTemporalAttentionPool(
                input_dim,
                group_size=2,
                num_heads=self.temporal_pooling_heads,
            )
        self.spatial_embedding = nn.Parameter(torch.zeros(1, height * width, hidden_dim))
        self.set_embedding = nn.Embedding(48, hidden_dim)
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
        self.shared_mask_set = nn.Parameter(torch.zeros(1, 1, 1, 1, output_dim))

        nn.init.trunc_normal_(self.spatial_embedding, std=0.02)
        nn.init.trunc_normal_(self.type_embedding.weight, std=0.02)
        if self.temporal_pooling == "hidden_dim_attention":
            nn.init.trunc_normal_(self.video_phase_embedding, std=0.02)
            nn.init.trunc_normal_(self.video_query, std=0.02)

    def _mix_layers(self, group: PrefixGroupFeatures) -> Tensor:
        layers = group.layer_tokens or (group.tokens,)
        if len(layers) != self.num_input_layers:
            raise ValueError(f"Expected {self.num_input_layers} V-JEPA layers, got {len(layers)}")
        stacked = torch.stack(layers)
        if self.layer_fusion == "learned_softmax":
            weights = self.layer_mix_logits.softmax(dim=0).to(layers[0])
            mixed = (stacked * weights.view(-1, 1, 1, 1, 1, 1)).sum(dim=0)
        else:
            mixed = stacked.sum(dim=0)
        return self.layer_fusion_norm(mixed)

    def _reduce_group(self, group: PrefixGroupFeatures, mixed: Tensor) -> tuple[Tensor, int]:
        if group.latent_type == "image_first":
            if mixed.shape[1] != 1:
                raise ValueError("image_first must contain one selected tubelet")
            return self.first_projection(mixed[:, 0]), IMAGE_FIRST_ID
        if group.latent_type != "video_group" or mixed.shape[1] != 2:
            raise ValueError("video_group must contain two selected tubelets")

        if self.temporal_pooling == "input_dim_attention":
            return self.video_projection(self.video_temporal_pool(mixed)), VIDEO_GROUP_ID

        x = self.video_projection(mixed)
        b, pair, h, w, c = x.shape
        x = x + self.video_phase_embedding.view(1, pair, 1, 1, c)
        key_value = x.permute(0, 2, 3, 1, 4).reshape(b * h * w, pair, c)
        key_value = self.video_pair_norm(key_value)
        query = self.video_query.expand(b * h * w, -1, -1)
        fused, _ = self.video_pair_attention(query, key_value, key_value, need_weights=False)
        return fused[:, 0].reshape(b, h, w, c), VIDEO_GROUP_ID

    def _prepare_hidden(
        self,
        features: PrefixEncoderOutput,
        *,
        include_repa: bool,
    ) -> tuple[Tensor, Tensor, tuple[tuple[int, int, int], ...], RepaReference | None]:
        if not isinstance(features, PrefixEncoderOutput):
            raise TypeError("candidate v3 projector requires PrefixEncoderOutput")
        if features.spatial_grid != (self.height, self.width):
            raise ValueError(f"Expected spatial grid {(self.height, self.width)}")
        if features.max_context_frames != self.max_context_frames:
            raise ValueError("Encoder/projector max context mismatch")

        reduced: list[Tensor] = []
        anchors: list[Tensor] = []
        video_phases: list[Tensor] = []
        latent_type_ids: list[int] = []
        windows: list[tuple[int, int, int]] = []
        for group in features.groups:
            mixed = self._mix_layers(group)
            group_state, type_id = self._reduce_group(group, mixed)
            reduced.append(group_state)
            if include_repa:
                if type_id == IMAGE_FIRST_ID:
                    anchors.append(mixed[:, 0])
                else:
                    video_phases.append(mixed)
            latent_type_ids.append(type_id)
            windows.append((group.source_start, group.source_end, group.input_frames))

        x = torch.stack(reduced, dim=1)
        b, t, h, w, c = x.shape
        flat_grid = x.reshape(b, t, h * w, c)
        ordered = flat_grid.index_select(2, self.fps_permutation)
        spatial = self.spatial_embedding.index_select(1, self.fps_permutation)
        set_embedding = self.set_embedding(self.canonical_set_ids).view(1, 1, h * w, c)
        type_ids = torch.tensor(latent_type_ids, device=x.device, dtype=torch.long)
        type_embedding = self.type_embedding(type_ids).view(1, t, 1, c)
        hidden = ordered + spatial.view(1, 1, h * w, c) + set_embedding + type_embedding
        repa_reference = None
        if include_repa:
            anchor = torch.stack(anchors, dim=1)
            if video_phases:
                phases = torch.stack(video_phases, dim=1)
            else:
                phases = anchor.new_empty(b, 0, 2, h, w, features.embed_dim)
            repa_reference = RepaReference(
                anchor=anchor.detach(), video_phases=phases.detach()
            )
        return hidden, type_ids, tuple(windows), repa_reference

    def _project_hidden(self, hidden: Tensor, token_count: int) -> Tensor:
        b, t, _, c = hidden.shape
        selected = hidden[:, :, :token_count]
        attention_mask = None
        if self.spatial_attention_mode == "set_causal":
            attention_mask = self.causal_attention_mask[:token_count, :token_count].to(
                hidden.device
            )
        hidden = self.transformer(
            selected.reshape(b * t, token_count, c),
            mask=attention_mask,
        )
        canonical = self.state_norm(self.output_projection(self.output_norm(hidden)))
        return canonical.reshape(b, t, token_count // 30, 30, self.output_dim)

    def forward(self, features: PrefixEncoderOutput) -> ProjectorOutput:
        hidden, type_ids, windows, repa_reference = self._prepare_hidden(
            features, include_repa=True
        )
        canonical = self._project_hidden(hidden, self.height * self.width)
        state = ProgressiveState(
            tokens=canonical,
            layout_version=self.layout_version,
            layout_checksum=self.layout_checksum,
            latent_types=type_ids,
            contract=self.contract,
        )
        assert repa_reference is not None
        return ProjectorOutput(
            state=state,
            repa_reference=repa_reference,
            prefix_windows=windows,
        )

    def make_prefix_view(self, state: ProgressiveState, endpoint: int) -> SpatialPrefixView:
        self.contract.assert_compatible(state.contract)
        if not 0 <= endpoint < self.contract.num_sets - 1:
            raise ValueError("prefix endpoint must be in [0, 46]")
        set_index = torch.arange(self.contract.num_sets, device=state.tokens.device)
        visible = (set_index <= endpoint).view(1, 1, -1, 1, 1)
        tokens = torch.where(visible, state.tokens, self.shared_mask_set.to(state.tokens))
        return SpatialPrefixView(tokens=tokens, endpoint=int(endpoint), source=state)

    def grid_view(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 5 or tuple(tokens.shape[2:4]) != (48, 30):
            raise ValueError("Expected ordered state [B,T,48,30,C]")
        b, t, _, _, c = tokens.shape
        ordered = tokens.reshape(b, t, self.height * self.width, c)
        return ordered.index_select(2, self.inverse_fps_permutation).reshape(
            b, t, self.height, self.width, c
        )
