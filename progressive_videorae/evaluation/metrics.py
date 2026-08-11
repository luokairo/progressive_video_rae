from __future__ import annotations

import torch
from torch import Tensor

from ..model.types import ProgressiveState
from .full_metrics import (
    FullReconstructionMetricSuite,
    I3DFeatureExtractor,
    encoder_cosine,
    frame_mean_psnr,
    frechet_distance,
)


MetricSuite = FullReconstructionMetricSuite
psnr = frame_mean_psnr


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
    source = state.tokens
    rows = []
    for set_id in range(source.shape[2]):
        values = source[:, :, set_id].reshape(-1, source.shape[-1])
        variance = values.float().var(dim=0, unbiased=False).mean()
        centered = values.float() - values.float().mean(dim=0, keepdim=True)
        rows.append(
            {
                "set_id": float(set_id),
                "set_size": float(source.shape[3]),
                "variance": float(variance.cpu()),
                "effective_rank": float(effective_rank(centered).cpu()),
                "incremental_mean_norm": float(
                    values.float().norm(dim=-1).mean().cpu()
                ),
                "collapsed": float(variance < collapse_threshold),
            }
        )
    return rows


__all__ = [
    "I3DFeatureExtractor",
    "MetricSuite",
    "encoder_cosine",
    "frechet_distance",
    "psnr",
    "state_set_statistics",
]
