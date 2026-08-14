from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..model.types import EncoderOutput, PrefixEncoderOutput
from .full import FVD_FEATURE_DIM


def frame_mse(prediction: Tensor, target: Tensor) -> Tensor:
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError(
            "frame_mse expects matching [B,C,T,H,W] tensors, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    return (prediction - target).square().mean(dim=(1, 3, 4))


def frame_mean_psnr(prediction: Tensor, target: Tensor, data_range: float = 2.0) -> Tensor:
    per_frame = 10.0 * torch.log10(
        data_range**2 / frame_mse(prediction, target).clamp_min(1e-12)
    )
    return per_frame.mean(dim=1)


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


def temporal_difference_l1(prediction: Tensor, target: Tensor) -> Tensor:
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("temporal_difference_l1 expects matching [B,C,T,H,W] tensors")
    if prediction.shape[2] < 2:
        return prediction.new_zeros(())
    return F.l1_loss(
        prediction[:, :, 1:] - prediction[:, :, :-1],
        target[:, :, 1:] - target[:, :, :-1],
    )


def clamp_visual_metric_inputs(
    prediction_raw: Tensor, target: Tensor
) -> tuple[Tensor, Tensor]:
    if prediction_raw.shape != target.shape or prediction_raw.ndim != 5:
        raise ValueError("Visual metrics require matching [B,C,T,H,W] tensors")
    if not torch.isfinite(prediction_raw).all() or not torch.isfinite(target).all():
        raise FloatingPointError("Visual metric inputs contain non-finite values")
    return (
        prediction_raw.float().clamp(-1.0, 1.0),
        target.float().clamp(-1.0, 1.0),
    )

def output_range_statistics(prediction_raw: Tensor) -> dict[str, float]:
    overshoot = (prediction_raw.detach().float().abs() - 1.0).clamp_min(0.0)
    return {
        "out_of_range_fraction": float((overshoot > 0).float().mean().cpu()),
        "mean_overshoot": float(overshoot.mean().cpu()),
        "max_overshoot": float(overshoot.max().cpu()),
    }


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


class EvaluationLPIPS(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        try:
            import lpips
        except ImportError as exc:
            raise ImportError("Install lpips to evaluate perceptual similarity") from exc
        self.model = lpips.LPIPS(net="vgg").eval().requires_grad_(False)

    def forward(self, prediction: Tensor, target: Tensor, chunk_size: int = 4) -> Tensor:
        b, c, t, h, w = prediction.shape
        pred = prediction.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        real = target.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        values = [
            self.model(pred[start : start + chunk_size], real[start : start + chunk_size])
            for start in range(0, pred.shape[0], chunk_size)
        ]
        return torch.cat(values).mean()


class FullReconstructionMetricSuite(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lpips = EvaluationLPIPS()

    @torch.no_grad()
    def forward(self, prediction_raw: Tensor, target: Tensor) -> dict[str, float]:
        prediction = prediction_raw.float().clamp(-1.0, 1.0)
        target = target.float().clamp(-1.0, 1.0)
        values = {
            "rgb_mse": float(frame_mse(prediction, target).mean().cpu()),
            "rgb_psnr": float(frame_mean_psnr(prediction, target).mean().cpu()),
            "rgb_ssim": float(ssim(prediction, target).cpu()),
            "rgb_lpips": float(self.lpips(prediction, target).cpu()),
            "temporal_difference_l1": float(
                temporal_difference_l1(prediction, target).cpu()
            ),
            **output_range_statistics(prediction_raw),
        }
        if not all(np.isfinite(value) for value in values.values()):
            raise FloatingPointError(f"Non-finite reconstruction metric: {values}")
        return values


def stylegan_v_i3d_preprocess(video: Tensor, resolution: int = 224) -> Tensor:
    if video.ndim != 5 or video.shape[1] != 3:
        raise ValueError(f"I3D input must be RGB BCTHW, got {tuple(video.shape)}")
    if not torch.isfinite(video).all():
        raise FloatingPointError("I3D input contains non-finite values")
    video = video.float().clamp(-1.0, 1.0)
    b, c, t, h, w = video.shape
    scale = resolution / min(h, w)
    resized_h = max(resolution, int(np.ceil(h * scale)))
    resized_w = max(resolution, int(np.ceil(w * scale)))
    frames = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    frames = F.interpolate(
        frames, size=(resized_h, resized_w), mode="bilinear", align_corners=False
    )
    top = (resized_h - resolution) // 2
    left = (resized_w - resolution) // 2
    frames = frames[:, :, top : top + resolution, left : left + resolution]
    return (
        frames.reshape(b, t, c, resolution, resolution)
        .permute(0, 2, 1, 3, 4)
        .contiguous()
    )


class I3DFeatureExtractor(nn.Module):
    def __init__(self, checkpoint_path: str) -> None:
        super().__init__()
        self.model = (
            torch.jit.load(checkpoint_path, map_location="cpu")
            .eval()
            .requires_grad_(False)
        )

    @torch.no_grad()
    def forward(self, video: Tensor) -> Tensor:
        prepared = stylegan_v_i3d_preprocess(video)
        output = self.model(
            prepared,
            rescale=False,
            resize=False,
            return_features=True,
        )
        if not isinstance(output, Tensor):
            raise TypeError(f"I3D must return a Tensor, got {type(output).__name__}")
        output = output.float()
        expected = (prepared.shape[0], FVD_FEATURE_DIM)
        if output.ndim != 2 or tuple(output.shape) != expected:
            raise ValueError(f"I3D features must be {expected}, got {tuple(output.shape)}")
        if not torch.isfinite(output).all():
            raise FloatingPointError("I3D returned non-finite features")
        return output


def frechet_distance(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    from scipy import linalg

    real = np.asarray(real_features, dtype=np.float64)
    fake = np.asarray(fake_features, dtype=np.float64)
    if real.ndim != 2 or fake.ndim != 2 or real.shape != fake.shape:
        raise ValueError(
            f"FVD features must have equal [N,D] shapes: {real.shape}, {fake.shape}"
        )
    if real.shape[0] < 2:
        raise ValueError("FVD requires at least two real and reconstructed samples")
    if not np.isfinite(real).all() or not np.isfinite(fake).all():
        raise FloatingPointError("FVD features contain non-finite values")
    real_mean = np.mean(real, axis=0)
    fake_mean = np.mean(fake, axis=0)
    real_cov = np.cov(real, rowvar=False)
    fake_cov = np.cov(fake, rowvar=False)
    covariance_mean, _ = linalg.sqrtm(real_cov @ fake_cov, disp=False)
    if np.iscomplexobj(covariance_mean):
        covariance_mean = covariance_mean.real
    difference = real_mean - fake_mean
    value = float(
        difference.dot(difference)
        + np.trace(real_cov)
        + np.trace(fake_cov)
        - 2.0 * np.trace(covariance_mean)
    )
    if not np.isfinite(value):
        raise FloatingPointError("FVD calculation produced a non-finite value")
    if value < -1e-6:
        raise FloatingPointError(f"FVD calculation produced a negative value: {value}")
    return max(value, 0.0)
