from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor


@dataclass(frozen=True)
class FrequencyCutoff:
    prefix_len: int
    omega: int
    height: int
    width: int


def spectralar_omega(prefix_len: int) -> int:
    if prefix_len < 1 or prefix_len > 64:
        raise ValueError(f"prefix_len must be in [1, 64], got {prefix_len}")
    if prefix_len <= 32:
        return prefix_len
    if prefix_len <= 48:
        return 2 * prefix_len - 32
    return 12 * prefix_len - 512


def frequency_cutoff(prefix_len: int, height: int = 480, width: int = 768) -> FrequencyCutoff:
    omega = spectralar_omega(prefix_len)
    return FrequencyCutoff(
        prefix_len=prefix_len,
        omega=omega,
        height=min(height, math.ceil(height * omega / 256)),
        width=min(width, math.ceil(width * omega / 256)),
    )


def dct_lowpass_target(video: Tensor, prefix_len: int) -> Tensor:
    """Build a per-frame, per-channel orthonormal DCT-II prefix target."""

    try:
        import torch_dct
    except ImportError as exc:
        raise ImportError("Install torch-dct to use DCT prefix supervision") from exc
    if video.ndim != 5:
        raise ValueError(f"Expected [B,C,T,H,W], got {tuple(video.shape)}")
    b, c, t, h, w = video.shape
    cutoff = frequency_cutoff(prefix_len, h, w)
    flat = video.permute(0, 2, 1, 3, 4).reshape(b * t * c, h, w)
    coeff = torch_dct.dct_2d(flat, norm="ortho")
    masked = torch.zeros_like(coeff)
    masked[:, : cutoff.height, : cutoff.width] = coeff[:, : cutoff.height, : cutoff.width]
    restored = torch_dct.idct_2d(masked, norm="ortho")
    return restored.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)


def frequency_leakage(prediction: Tensor, prefix_len: int, eps: float = 1e-8) -> Tensor:
    try:
        import torch_dct
    except ImportError as exc:
        raise ImportError("Install torch-dct to compute frequency leakage") from exc
    b, c, t, h, w = prediction.shape
    cutoff = frequency_cutoff(prefix_len, h, w)
    flat = prediction.permute(0, 2, 1, 3, 4).reshape(b * t * c, h, w)
    energy = torch_dct.dct_2d(flat, norm="ortho").square()
    total = energy.sum(dim=(-2, -1))
    allowed = energy[:, : cutoff.height, : cutoff.width].sum(dim=(-2, -1))
    return ((total - allowed) / total.clamp_min(eps)).mean()

