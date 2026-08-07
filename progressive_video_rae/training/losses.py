from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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


class ProgressiveLosses(nn.Module):
    def __init__(self, lpips_weight: float = 1.0, adversarial_weight: float = 0.1) -> None:
        super().__init__()
        self.lpips = FrozenLPIPS()
        self.lpips_weight = lpips_weight
        self.adversarial_weight = adversarial_weight

    def prefix(self, output: ProgressiveVideoRAEOutput, target: Tensor) -> LossOutput:
        l1 = F.l1_loss(output.reconstruction, target)
        perceptual = self.lpips(output.reconstruction, target)
        terms = {"l1": l1, "lpips": perceptual}
        return LossOutput(total=l1 + 0.5 * self.lpips_weight * perceptual, terms=terms)

    def full_generator(
        self,
        output: ProgressiveVideoRAEOutput,
        discriminator: nn.Module,
    ) -> LossOutput:
        l1 = F.l1_loss(output.reconstruction, output.target)
        perceptual = self.lpips(output.reconstruction, output.target)
        if output.repa_features is None:
            raise RuntimeError("Full-state training requires decoder REPA features")
        reference = output.encoder_output.tokens.detach()
        if output.repa_features.shape != reference.shape:
            raise RuntimeError(
                f"REPA shape mismatch: {tuple(output.repa_features.shape)} vs {tuple(reference.shape)}"
            )
        local = 1.0 - F.cosine_similarity(output.repa_features, reference, dim=-1).mean()
        predicted_global = output.repa_features.mean(dim=(1, 2, 3))
        reference_global = reference.mean(dim=(1, 2, 3))
        global_loss = 1.0 - F.cosine_similarity(predicted_global, reference_global, dim=-1).mean()
        adversarial = -discriminator(output.reconstruction).mean()
        terms = {
            "l1": l1,
            "lpips": perceptual,
            "repa_local": local,
            "repa_global": global_loss,
            "adversarial": adversarial,
        }
        total = (
            l1
            + self.lpips_weight * perceptual
            + local
            + global_loss
            + self.adversarial_weight * adversarial
        )
        return LossOutput(total=total, terms=terms)

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
        return LossOutput(total=total, terms={"disc_real": real_loss, "disc_fake": fake_loss})


def scalar_terms(terms: dict[str, Tensor]) -> dict[str, float]:
    return {key: float(value.detach().float().cpu()) for key, value in terms.items()}
