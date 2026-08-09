from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..model.dct import frequency_leakage
from ..model.types import EncoderOutput, PrefixEncoderOutput, ProgressiveState
from ..training.losses import FrozenLPIPS


def psnr(prediction: Tensor, target: Tensor, data_range: float = 2.0) -> Tensor:
    mse = (prediction - target).square().flatten(1).mean(1)
    return 10.0 * torch.log10(data_range**2 / mse.clamp_min(1e-12))


def ssim(prediction: Tensor, target: Tensor) -> Tensor:
    try:
        from torchmetrics.functional.image import structural_similarity_index_measure
    except ImportError as exc:
        raise ImportError("Install torchmetrics to compute SSIM") from exc
    b, c, t, h, w = prediction.shape
    pred = prediction.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    real = target.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    return structural_similarity_index_measure(
        pred, real, data_range=2.0, reduction="elementwise_mean"
    )


def temporal_l1(prediction: Tensor, target: Tensor) -> Tensor:
    return F.l1_loss(
        prediction[:, :, 1:] - prediction[:, :, :-1],
        target[:, :, 1:] - target[:, :, :-1],
    )


def _semantic_tokens(value: Tensor | EncoderOutput | PrefixEncoderOutput) -> Tensor:
    if isinstance(value, Tensor):
        return value
    if isinstance(value, EncoderOutput):
        return value.tokens
    return torch.stack([group.tokens.mean(dim=1) for group in value.groups], dim=1)


def encoder_cosine(
    reference: Tensor | EncoderOutput | PrefixEncoderOutput,
    reconstruction: Tensor | EncoderOutput | PrefixEncoderOutput,
) -> tuple[Tensor, Tensor]:
    reference_tokens = _semantic_tokens(reference)
    reconstruction_tokens = _semantic_tokens(reconstruction)
    local = F.cosine_similarity(reference_tokens, reconstruction_tokens, dim=-1).mean()
    ref_global = reference_tokens.mean(dim=(1, 2, 3))
    rec_global = reconstruction_tokens.mean(dim=(1, 2, 3))
    global_score = F.cosine_similarity(ref_global, rec_global, dim=-1).mean()
    return local, global_score


def effective_rank(matrix: Tensor, eps: float = 1e-8) -> Tensor:
    if matrix.shape[0] < 2:
        return matrix.new_tensor(1.0)
    singular = torch.linalg.svdvals(matrix.float())
    probabilities = singular / singular.sum().clamp_min(eps)
    entropy = -(probabilities * probabilities.clamp_min(eps).log()).sum()
    return entropy.exp()


def state_set_statistics(
    state: ProgressiveState, collapse_threshold: float = 1e-4
) -> list[dict[str, float]]:
    source = (state.metadata or {}).get("unmasked_tokens", state.tokens)
    rows = []
    for set_id in range(len(state.set_sizes)):
        mask = state.set_ids == set_id
        values = source[:, :, mask, :].reshape(-1, source.shape[-1])
        variance = values.float().var(dim=0, unbiased=False).mean()
        centered = values.float() - values.float().mean(dim=0, keepdim=True)
        rows.append(
            {
                "set_id": float(set_id),
                "set_size": float(mask.sum().item()),
                "variance": float(variance.cpu()),
                "effective_rank": float(effective_rank(centered).cpu()),
                "incremental_mean_norm": float(values.float().norm(dim=-1).mean().cpu()),
                "collapsed": float(variance < collapse_threshold),
            }
        )
    return rows


class MetricSuite(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lpips = FrozenLPIPS()

    @torch.no_grad()
    def reconstruction(
        self,
        prediction: Tensor,
        target: Tensor,
        *,
        prefix_len: int,
    ) -> dict[str, float]:
        metrics = {
            "psnr": float(psnr(prediction, target).mean().cpu()),
            "ssim": float(ssim(prediction, target).cpu()),
            "lpips": float(self.lpips(prediction, target).cpu()),
            "temporal_l1": float(temporal_l1(prediction, target).cpu()),
            "frequency_leakage": float(frequency_leakage(prediction, prefix_len).cpu()),
        }
        if prediction.shape[2] > 1:
            metrics["temporal_lpips"] = float(
                self.lpips(
                    prediction[:, :, 1:] - prediction[:, :, :-1],
                    target[:, :, 1:] - target[:, :, :-1],
                ).cpu()
            )
        return metrics


class I3DFeatureExtractor(nn.Module):
    def __init__(self, checkpoint_path: str) -> None:
        super().__init__()
        self.model = torch.jit.load(checkpoint_path, map_location="cpu").eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, video: Tensor) -> Tensor:
        b, _c, t, _h, _w = video.shape
        resized = F.interpolate(video, size=(t, 224, 224), mode="trilinear", align_corners=False)
        output = self.model(resized)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if isinstance(output, dict):
            output = next(iter(output.values()))
        return output.float().reshape(b, -1)


def frechet_distance(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    from scipy import linalg

    real_mean = np.mean(real_features, axis=0)
    fake_mean = np.mean(fake_features, axis=0)
    real_cov = np.cov(real_features, rowvar=False)
    fake_cov = np.cov(fake_features, rowvar=False)
    covariance_mean, _ = linalg.sqrtm(real_cov @ fake_cov, disp=False)
    if np.iscomplexobj(covariance_mean):
        covariance_mean = covariance_mean.real
    difference = real_mean - fake_mean
    return float(
        difference.dot(difference)
        + np.trace(real_cov)
        + np.trace(fake_cov)
        - 2.0 * np.trace(covariance_mean)
    )
