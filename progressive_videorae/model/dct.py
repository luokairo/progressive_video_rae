from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor


NUM_PROGRESSIVE_SETS = 48


@dataclass(frozen=True)
class FrequencyCutoff:
    endpoint: int
    height: int
    width: int


def frequency_cutoff(endpoint: int, height: int = 480, width: int = 768) -> FrequencyCutoff:
    if not 0 <= endpoint < NUM_PROGRESSIVE_SETS:
        raise ValueError(f"endpoint must be in [0,47], got {endpoint}")
    if endpoint == NUM_PROGRESSIVE_SETS - 1:
        return FrequencyCutoff(endpoint, height, width)
    ratio = 1.0 - math.cos(math.pi * (endpoint + 1) / (2 * NUM_PROGRESSIVE_SETS))
    return FrequencyCutoff(
        endpoint,
        min(height, math.ceil(height * ratio)),
        min(width, math.ceil(width * ratio)),
    )


def _dct(video: Tensor) -> tuple[Tensor, tuple[int, int, int, int, int]]:
    try:
        import torch_dct
    except ImportError as exc:
        raise ImportError("Install torch-dct to use progressive frequency supervision") from exc
    if video.ndim != 5:
        raise ValueError(f"Expected [B,C,T,H,W], got {tuple(video.shape)}")
    shape = tuple(video.shape)
    b, c, t, h, w = shape
    flat = video.permute(0, 2, 1, 3, 4).reshape(b * t * c, h, w)
    return torch_dct.dct_2d(flat, norm="ortho"), shape


def _restore(coeff: Tensor, shape: tuple[int, int, int, int, int]) -> Tensor:
    import torch_dct

    b, c, t, h, w = shape
    restored = torch_dct.idct_2d(coeff, norm="ortho")
    return restored.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)


def dct_lowpass_target(video: Tensor, endpoint: int) -> Tensor:
    coeff, shape = _dct(video)
    cutoff = frequency_cutoff(endpoint, shape[-2], shape[-1])
    masked = torch.zeros_like(coeff)
    masked[:, : cutoff.height, : cutoff.width] = coeff[:, : cutoff.height, : cutoff.width]
    return _restore(masked, shape)


def dct_band_coefficients(video: Tensor, endpoint: int) -> tuple[Tensor, Tensor]:
    coeff, shape = _dct(video)
    current = frequency_cutoff(endpoint, shape[-2], shape[-1])
    previous = frequency_cutoff(endpoint - 1, shape[-2], shape[-1]) if endpoint else None
    mask = torch.zeros_like(coeff, dtype=torch.bool)
    mask[:, : current.height, : current.width] = True
    if previous is not None:
        mask[:, : previous.height, : previous.width] = False
    return coeff, mask


def frequency_leakage(prediction: Tensor, endpoint: int, eps: float = 1e-8) -> Tensor:
    coeff, shape = _dct(prediction)
    cutoff = frequency_cutoff(endpoint, shape[-2], shape[-1])
    energy = coeff.square()
    total = energy.sum(dim=(-2, -1))
    allowed = energy[:, : cutoff.height, : cutoff.width].sum(dim=(-2, -1))
    return ((total - allowed) / total.clamp_min(eps)).mean()


@dataclass(frozen=True)
class ProgressiveFrequencyTerms:
    target: Tensor
    target_coefficients: Tensor
    prediction_coefficients: Tensor
    band_mask: Tensor
    leakage: Tensor


def progressive_frequency_terms(
    prediction: Tensor,
    full_target: Tensor,
    endpoint: int,
    *,
    eps: float = 1e-8,
) -> ProgressiveFrequencyTerms:
    if prediction.shape != full_target.shape:
        raise ValueError(
            f"Progressive frequency shape mismatch: {prediction.shape} vs {full_target.shape}"
        )

    target_coefficients, shape = _dct(full_target)
    prediction_coefficients, prediction_shape = _dct(prediction)
    if prediction_shape != shape:
        raise ValueError("Prediction and target DCT shapes must match")

    cutoff = frequency_cutoff(endpoint, shape[-2], shape[-1])
    previous = frequency_cutoff(endpoint - 1, shape[-2], shape[-1]) if endpoint else None
    band_mask = torch.zeros_like(target_coefficients, dtype=torch.bool)
    band_mask[:, : cutoff.height, : cutoff.width] = True
    if previous is not None:
        band_mask[:, : previous.height, : previous.width] = False

    if endpoint == NUM_PROGRESSIVE_SETS - 1:
        target = full_target
    else:
        masked = torch.zeros_like(target_coefficients)
        masked[:, : cutoff.height, : cutoff.width] = target_coefficients[
            :, : cutoff.height, : cutoff.width
        ]
        target = _restore(masked, shape)

    if endpoint == NUM_PROGRESSIVE_SETS - 1:
        leakage = prediction_coefficients.sum() * 0.0
    else:
        energy = prediction_coefficients.square()
        total = energy.sum(dim=(-2, -1))
        allowed = energy[:, : cutoff.height, : cutoff.width].sum(dim=(-2, -1))
        leakage = ((total - allowed) / total.clamp_min(eps)).mean()
    return ProgressiveFrequencyTerms(
        target=target,
        target_coefficients=target_coefficients,
        prediction_coefficients=prediction_coefficients,
        band_mask=band_mask,
        leakage=leakage,
    )
