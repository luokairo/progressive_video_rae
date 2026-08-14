from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..model.model import ProgressiveVideoRAEOutput
from ..model.dct import dct_band_coefficients, progressive_frequency_terms
from ..model.types import RepaReference


HIERARCHICAL_REPA_LEVELS = (
    (1, 1),
    (2, 3),
    (4, 6),
    (8, 12),
    (15, 24),
    (30, 48),
)


@dataclass(frozen=True)
class HierarchicalRepaTerms:
    global_loss: Tensor
    local_loss: Tensor
    local_scale: Tensor
    level: int
    grid: tuple[int, int]
    anchor_global_error: Tensor
    anchor_local_error: Tensor
    group_phase_global_errors: Tensor
    group_phase_local_errors: Tensor


def hierarchical_repa_level(endpoint: int) -> tuple[int, tuple[int, int], float]:
    if not 0 <= endpoint <= 46:
        raise ValueError(f"Hierarchical REPA endpoint must be in [0,46], got {endpoint}")
    level = min(endpoint // 8, len(HIERARCHICAL_REPA_LEVELS) - 1)
    scale = level / (len(HIERARCHICAL_REPA_LEVELS) - 1)
    return level, HIERARCHICAL_REPA_LEVELS[level], scale


def _pool_repa(features: Tensor, grid: tuple[int, int]) -> Tensor:
    if features.ndim not in (5, 6):
        raise ValueError(
            "REPA pyramid expects anchor [B,1,H,W,C] or phases [B,G,2,H,W,C]"
        )
    height, width, channels = features.shape[-3:]
    flat = features.reshape(-1, height, width, channels).permute(0, 3, 1, 2)
    pooled = F.adaptive_avg_pool2d(flat.float(), grid)
    return pooled.permute(0, 2, 3, 1).reshape(*features.shape[:-3], *grid, channels)


def _repa_cosine_errors(
    prediction: Tensor,
    target: Tensor,
    grid: tuple[int, int],
) -> Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"REPA shape mismatch: {prediction.shape} vs {target.shape}")
    return 1.0 - F.cosine_similarity(
        _pool_repa(prediction, grid),
        _pool_repa(target.detach(), grid),
        dim=-1,
    )


def hierarchical_repa_terms(
    prediction: RepaReference,
    target: RepaReference,
    endpoint: int,
) -> HierarchicalRepaTerms:
    level, grid, local_scale_value = hierarchical_repa_level(endpoint)
    anchor_global_errors = _repa_cosine_errors(prediction.anchor, target.anchor, (1, 1))
    anchor_local_errors = _repa_cosine_errors(prediction.anchor, target.anchor, grid)
    anchor_global = anchor_global_errors.mean()
    anchor_local = anchor_local_errors.mean()

    if prediction.video_phases.shape != target.video_phases.shape:
        raise ValueError(
            "REPA video phase shape mismatch: "
            f"{prediction.video_phases.shape} vs {target.video_phases.shape}"
        )
    if prediction.video_phases.numel():
        phase_global_errors = _repa_cosine_errors(
            prediction.video_phases, target.video_phases, (1, 1)
        )
        phase_local_errors = _repa_cosine_errors(
            prediction.video_phases, target.video_phases, grid
        )
        video_global = phase_global_errors.mean()
        video_local = phase_local_errors.mean()
        global_loss = 0.5 * (anchor_global + video_global)
        local_loss = 0.5 * (anchor_local + video_local)
        group_phase_global_errors = phase_global_errors.mean(dim=(0, 3, 4))
        group_phase_local_errors = phase_local_errors.mean(dim=(0, 3, 4))
    else:
        global_loss = anchor_global
        local_loss = anchor_local
        group_phase_global_errors = anchor_global.new_empty((0, 2))
        group_phase_local_errors = anchor_local.new_empty((0, 2))

    return HierarchicalRepaTerms(
        global_loss=global_loss,
        local_loss=local_loss,
        local_scale=anchor_local.new_tensor(local_scale_value),
        level=level,
        grid=grid,
        anchor_global_error=anchor_global,
        anchor_local_error=anchor_local,
        group_phase_global_errors=group_phase_global_errors,
        group_phase_local_errors=group_phase_local_errors,
    )


class PatchDiscriminator(nn.Module):
    """2D PatchGAN applied to a deterministic subset of video frames."""

    def __init__(self, base_channels: int = 64, num_layers: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(3, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        channels = base_channels
        for layer_index in range(1, num_layers):
            out_channels = min(base_channels * 2**layer_index, 512)
            layers.extend(
                [
                    nn.Conv2d(channels, out_channels, 4, stride=2, padding=1, bias=False),
                    nn.GroupNorm(min(32, out_channels), out_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            channels = out_channels
        layers.append(nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1))
        self.net = nn.Sequential(*layers)

    @staticmethod
    def select_frames(video: Tensor, frames_per_clip: int = 4) -> Tensor:
        b, c, t, h, w = video.shape
        indices = torch.linspace(0, t - 1, min(t, frames_per_clip), device=video.device).long()
        return video.index_select(2, indices).permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)

    def forward(self, video: Tensor) -> Tensor:
        return self.net(self.select_frames(video))


class FrozenLPIPS(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        try:
            import lpips
        except ImportError as exc:
            raise ImportError("Install lpips to train or evaluate perceptual loss") from exc
        self.model = lpips.LPIPS(net="vgg").eval().requires_grad_(False)

    def forward(self, prediction: Tensor, target: Tensor, chunk_size: int = 4) -> Tensor:
        b, c, t, h, w = prediction.shape
        pred = prediction.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        real = target.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        values = []
        for start in range(0, pred.shape[0], chunk_size):
            pred_chunk = pred[start : start + chunk_size]
            real_chunk = real[start : start + chunk_size]
            if self.training and pred_chunk.requires_grad:
                values.append(
                    checkpoint(self.model, pred_chunk, real_chunk, use_reentrant=False)
                )
            else:
                values.append(self.model(pred_chunk, real_chunk))
        return torch.cat(values).mean()


@dataclass
class LossOutput:
    total: Tensor
    terms: dict[str, Tensor]
    weighted_terms: dict[str, Tensor] = field(default_factory=dict)
    statistics: dict[str, Tensor] = field(default_factory=dict)


def temporal_l1(prediction: Tensor, target: Tensor) -> Tensor:
    """Match first-order frame differences without penalizing single-frame clips."""

    if prediction.shape != target.shape:
        raise ValueError(
            f"Temporal L1 shape mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}"
        )
    if prediction.ndim != 5:
        raise ValueError(f"Temporal L1 expects [B,C,T,H,W], got {tuple(prediction.shape)}")
    if prediction.shape[2] < 2:
        return prediction.sum() * 0.0
    return F.l1_loss(
        prediction[:, :, 1:] - prediction[:, :, :-1],
        target[:, :, 1:] - target[:, :, :-1],
    )


class ProgressiveLosses(nn.Module):
    def __init__(
        self,
        *,
        l1_weight: float = 1.0,
        prefix_lpips_weight: float = 0.5,
        lpips_weight: float = 1.0,
        repa_local_weight: float = 1.0,
        repa_global_weight: float = 1.0,
        prefix_repa_local_weight: float = 0.0,
        prefix_repa_global_weight: float = 0.0,
        adversarial_weight: float = 0.1,
        temporal_l1_weight: float = 0.0,
        band_weight: float = 1.0,
        leakage_weight: float = 0.1,
        paired_delta_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.lpips = FrozenLPIPS()
        self.l1_weight = float(l1_weight)
        self.prefix_lpips_weight = float(prefix_lpips_weight)
        self.lpips_weight = float(lpips_weight)
        self.repa_local_weight = float(repa_local_weight)
        self.repa_global_weight = float(repa_global_weight)
        self.prefix_repa_local_weight = float(prefix_repa_local_weight)
        self.prefix_repa_global_weight = float(prefix_repa_global_weight)
        self.adversarial_weight = float(adversarial_weight)
        self.temporal_l1_weight = float(temporal_l1_weight)
        self.band_weight = float(band_weight)
        self.leakage_weight = float(leakage_weight)
        self.paired_delta_weight = float(paired_delta_weight)

    def _lpips(self, prediction: Tensor, target: Tensor) -> Tensor:
        return self.lpips(prediction, target)

    def prefix(
        self,
        output: ProgressiveVideoRAEOutput,
        full_target: Tensor,
        *,
        endpoint: int,
        previous_prediction: Tensor | None = None,
    ) -> LossOutput:
        needs_frequency_terms = (
            endpoint != 47
            or self.band_weight != 0.0
            or self.leakage_weight != 0.0
            or (previous_prediction is not None and self.paired_delta_weight != 0.0)
        )
        band = output.reconstruction.new_zeros(())
        leakage = output.reconstruction.new_zeros(())
        target_coeff = None
        band_mask = None
        if needs_frequency_terms:
            spectral = progressive_frequency_terms(
                output.reconstruction.float(), full_target.float(), endpoint
            )
            target = spectral.target
            target_coeff = spectral.target_coefficients
            prediction_coeff = spectral.prediction_coefficients
            band_mask = spectral.band_mask
            if bool(band_mask.any()):
                band = F.l1_loss(prediction_coeff[band_mask], target_coeff[band_mask])
            leakage = spectral.leakage
        else:
            target = full_target.float()

        l1 = F.l1_loss(output.reconstruction, target)
        perceptual = self._lpips(output.reconstruction, target)
        temporal = temporal_l1(output.reconstruction, target)
        paired_delta = output.reconstruction.new_zeros(())
        if previous_prediction is not None and self.paired_delta_weight != 0.0:
            delta_coeff, _ = dct_band_coefficients(
                (output.reconstruction - previous_prediction).float(), endpoint
            )
            if bool(band_mask.any()):
                paired_delta = F.l1_loss(delta_coeff[band_mask], target_coeff[band_mask])

        prefix_repa_global = output.reconstruction.new_zeros(())
        prefix_repa_local = output.reconstruction.new_zeros(())
        prefix_repa_local_scaled = output.reconstruction.new_zeros(())
        statistics: dict[str, Tensor] = {}
        if self.prefix_repa_global_weight or self.prefix_repa_local_weight:
            if output.repa_features is None or output.repa_reference is None:
                raise RuntimeError("Prefix hierarchical REPA requires decoder features and target")
            hierarchical = hierarchical_repa_terms(
                output.repa_features, output.repa_reference, endpoint
            )
            prefix_repa_global = hierarchical.global_loss
            prefix_repa_local = hierarchical.local_loss
            prefix_repa_local_scaled = hierarchical.local_scale * prefix_repa_local
            statistics = {
                "prefix_repa/level": output.reconstruction.new_tensor(hierarchical.level),
                "prefix_repa/grid_height": output.reconstruction.new_tensor(
                    hierarchical.grid[0]
                ),
                "prefix_repa/grid_width": output.reconstruction.new_tensor(
                    hierarchical.grid[1]
                ),
                "prefix_repa/local_scale": hierarchical.local_scale,
                "prefix_repa/anchor_global_error": hierarchical.anchor_global_error,
                "prefix_repa/anchor_local_error": hierarchical.anchor_local_error,
            }
            for group_index in range(hierarchical.group_phase_local_errors.shape[0]):
                for phase_index, phase_name in enumerate(("f0", "f1")):
                    statistics[
                        f"prefix_repa/group_{group_index:02d}/{phase_name}_global_error"
                    ] = hierarchical.group_phase_global_errors[group_index, phase_index]
                    statistics[
                        f"prefix_repa/group_{group_index:02d}/{phase_name}_local_error"
                    ] = hierarchical.group_phase_local_errors[group_index, phase_index]
        terms = {
            "l1": l1,
            "lpips": perceptual,
            "temporal_l1": temporal,
            "band": band,
            "leakage": leakage,
            "paired_delta": paired_delta,
            "prefix_repa_global": prefix_repa_global,
            "prefix_repa_local": prefix_repa_local,
            "prefix_repa_local_scaled": prefix_repa_local_scaled,
        }
        weighted_terms = {
            "l1": self.l1_weight * l1,
            "lpips": self.prefix_lpips_weight * perceptual,
            "temporal_l1": self.temporal_l1_weight * temporal,
            "band": self.band_weight * band,
            "leakage": self.leakage_weight * leakage,
            "paired_delta": self.paired_delta_weight * paired_delta,
            "prefix_repa_global": (
                self.prefix_repa_global_weight * prefix_repa_global
            ),
            "prefix_repa_local_scaled": (
                self.prefix_repa_local_weight * prefix_repa_local_scaled
            ),
        }
        total = sum(weighted_terms.values())
        return LossOutput(
            total=total,
            terms=terms,
            weighted_terms=weighted_terms,
            statistics=statistics,
        )

    def full_generator(
        self,
        output: ProgressiveVideoRAEOutput,
        discriminator: nn.Module,
        *,
        adversarial_factor: float = 1.0,
    ) -> LossOutput:
        l1 = F.l1_loss(output.reconstruction, output.target)
        perceptual = self._lpips(output.reconstruction, output.target)
        if output.repa_features is None:
            raise RuntimeError("Full-state training requires decoder REPA features")
        if output.repa_reference is None:
            raise RuntimeError("Full-state training requires group-aligned REPA targets")
        predicted = output.repa_features
        reference = output.repa_reference
        if predicted.anchor.shape != reference.anchor.shape:
            raise RuntimeError("REPA anchor shape mismatch")
        if predicted.video_phases.shape != reference.video_phases.shape:
            raise RuntimeError("REPA video phase shape mismatch")
        anchor_local = 1.0 - F.cosine_similarity(
            predicted.anchor, reference.anchor.detach(), dim=-1
        ).mean()
        if predicted.video_phases.numel():
            phase_errors = 1.0 - F.cosine_similarity(
                predicted.video_phases, reference.video_phases.detach(), dim=-1
            )
            video_local = phase_errors.mean()
            group_phase_errors = phase_errors.mean(dim=(0, 3, 4))
            local = 0.5 * (anchor_local + video_local)
            predicted_all = torch.cat(
                (predicted.anchor.flatten(1, 3), predicted.video_phases.flatten(1, 4)), dim=1
            )
            reference_all = torch.cat(
                (reference.anchor.flatten(1, 3), reference.video_phases.flatten(1, 4)), dim=1
            ).detach()
        else:
            video_local = anchor_local.new_zeros(())
            group_phase_errors = anchor_local.new_empty((0, 2))
            local = anchor_local
            predicted_all = predicted.anchor.flatten(1, 3)
            reference_all = reference.anchor.flatten(1, 3).detach()
        predicted_global = predicted_all.mean(dim=1)
        reference_global = reference_all.mean(dim=1)
        global_loss = 1.0 - F.cosine_similarity(
            predicted_global, reference_global, dim=-1
        ).mean()
        temporal = temporal_l1(output.reconstruction, output.target)
        adversarial = (
            -discriminator(output.reconstruction).mean()
            if adversarial_factor > 0.0 and self.adversarial_weight > 0.0
            else output.reconstruction.new_zeros(())
        )
        terms = {
            "l1": l1,
            "lpips": perceptual,
            "temporal_l1": temporal,
            "repa_local": local,
            "repa_anchor": anchor_local,
            "repa_video": video_local,
            "repa_global": global_loss,
            "adversarial": adversarial,
        }
        weighted_terms = {
            "l1": self.l1_weight * l1,
            "lpips": self.lpips_weight * perceptual,
            "temporal_l1": self.temporal_l1_weight * temporal,
            "repa_local": self.repa_local_weight * local,
            "repa_global": self.repa_global_weight * global_loss,
            "adversarial": self.adversarial_weight * float(adversarial_factor) * adversarial,
        }
        statistics = {"repa/anchor_error": anchor_local}
        for group_index in range(group_phase_errors.shape[0]):
            statistics[f"repa/group_{group_index:02d}/f0_error"] = group_phase_errors[
                group_index, 0
            ]
            statistics[f"repa/group_{group_index:02d}/f1_error"] = group_phase_errors[
                group_index, 1
            ]
        if group_phase_errors.shape[0] >= 4:
            tail_four = group_phase_errors[-4:]
            statistics["repa/tail_four_mean"] = tail_four.mean()
            statistics["repa/tail_four_worst"] = tail_four.max()
        total = sum(weighted_terms.values())
        return LossOutput(
            total=total, terms=terms, weighted_terms=weighted_terms, statistics=statistics
        )

    @staticmethod
    def discriminator(
        discriminator: nn.Module,
        prediction: Tensor,
        target: Tensor,
        *,
        adversarial_factor: float = 1.0,
    ) -> LossOutput:
        real_logits = discriminator(target)
        fake_logits = discriminator(prediction.detach())
        real_loss = F.relu(1.0 - real_logits).mean()
        fake_loss = F.relu(1.0 + fake_logits).mean()
        raw_total = 0.5 * (real_loss + fake_loss)
        total = float(adversarial_factor) * raw_total
        return LossOutput(
            total=total,
            terms={
                "disc_real": real_loss,
                "disc_fake": fake_loss,
                "disc_total": raw_total,
                "disc_scaled_total": total,
            },
            statistics={
                "real_logits": real_logits,
                "fake_logits": fake_logits,
            },
        )


def scalar_terms(terms: dict[str, Tensor]) -> dict[str, float]:
    return {key: float(value.detach().float().cpu()) for key, value in terms.items()}
