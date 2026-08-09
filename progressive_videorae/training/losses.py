from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..model.model import ProgressiveVideoRAEOutput


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
            values.append(self.model(pred[start : start + chunk_size], real[start : start + chunk_size]))
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
        adversarial_weight: float = 0.1,
        temporal_l1_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.lpips = FrozenLPIPS()
        self.l1_weight = float(l1_weight)
        self.prefix_lpips_weight = float(prefix_lpips_weight)
        self.lpips_weight = float(lpips_weight)
        self.repa_local_weight = float(repa_local_weight)
        self.repa_global_weight = float(repa_global_weight)
        self.adversarial_weight = float(adversarial_weight)
        self.temporal_l1_weight = float(temporal_l1_weight)

    def prefix(self, output: ProgressiveVideoRAEOutput, target: Tensor) -> LossOutput:
        l1 = F.l1_loss(output.reconstruction, target)
        perceptual = self.lpips(output.reconstruction, target)
        temporal = temporal_l1(output.reconstruction, target)
        terms = {"l1": l1, "lpips": perceptual, "temporal_l1": temporal}
        weighted_terms = {
            "l1": self.l1_weight * l1,
            "lpips": self.prefix_lpips_weight * perceptual,
            "temporal_l1": self.temporal_l1_weight * temporal,
        }
        total = sum(weighted_terms.values())
        return LossOutput(total=total, terms=terms, weighted_terms=weighted_terms)

    def full_generator(
        self,
        output: ProgressiveVideoRAEOutput,
        discriminator: nn.Module,
        *,
        adversarial_factor: float = 1.0,
    ) -> LossOutput:
        l1 = F.l1_loss(output.reconstruction, output.target)
        perceptual = self.lpips(output.reconstruction, output.target)
        if output.repa_features is None:
            raise RuntimeError("Full-state training requires decoder REPA features")
        if output.repa_reference is None:
            raise RuntimeError("Full-state training requires group-aligned REPA targets")
        reference = output.repa_reference.detach()
        if output.repa_features.shape != reference.shape:
            raise RuntimeError(
                f"REPA shape mismatch: {tuple(output.repa_features.shape)} vs {tuple(reference.shape)}"
            )
        local = 1.0 - F.cosine_similarity(output.repa_features, reference, dim=-1).mean()
        predicted_global = output.repa_features.mean(dim=(1, 2, 3))
        reference_global = reference.mean(dim=(1, 2, 3))
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
        total = sum(weighted_terms.values())
        return LossOutput(total=total, terms=terms, weighted_terms=weighted_terms)

    @staticmethod
    def discriminator(
        discriminator: nn.Module,
        prediction: Tensor,
        target: Tensor,
    ) -> LossOutput:
        real_logits = discriminator(target)
        fake_logits = discriminator(prediction.detach())
        real_loss = F.relu(1.0 - real_logits).mean()
        fake_loss = F.relu(1.0 + fake_logits).mean()
        total = 0.5 * (real_loss + fake_loss)
        return LossOutput(
            total=total,
            terms={"disc_real": real_loss, "disc_fake": fake_loss, "disc_total": total},
            statistics={
                "real_logits": real_logits,
                "fake_logits": fake_logits,
            },
        )


def scalar_terms(terms: dict[str, Tensor]) -> dict[str, float]:
    return {key: float(value.detach().float().cpu()) for key, value in terms.items()}
