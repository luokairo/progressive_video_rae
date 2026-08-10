from __future__ import annotations

from pathlib import Path

import numpy as np
from torch import Tensor


def _to_uint8(video: Tensor) -> np.ndarray:
    video = video.detach().float().clamp(-1, 1).add(1).mul(127.5).byte()
    return video.permute(1, 2, 3, 0).cpu().numpy()


def save_comparison_video(
    path: str | Path,
    videos: list[Tensor],
    *,
    fps: float = 12.0,
) -> None:
    """Save horizontally concatenated [C,T,H,W] videos using PyAV."""

    import av

    arrays = [_to_uint8(video) for video in videos]
    grid = np.concatenate(arrays, axis=2)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=round(fps))
        stream.width = grid.shape[2]
        stream.height = grid.shape[1]
        stream.pix_fmt = "yuv420p"
        for frame_array in grid:
            frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def save_prefix_curve(rows: list[dict], output_path: str | Path, metric: str = "psnr_rgb") -> None:
    import matplotlib.pyplot as plt

    grouped: dict[int, list[float]] = {}
    for row in rows:
        if metric in row:
            grouped.setdefault(int(row["endpoint"]), []).append(float(row[metric]))
    endpoints = sorted(grouped)
    values = [sum(grouped[endpoint]) / len(grouped[endpoint]) for endpoint in endpoints]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(endpoints, values, marker="o")
    axis.set_xlabel("Inclusive set endpoint")
    axis.set_ylabel(metric)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
