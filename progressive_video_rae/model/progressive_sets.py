from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct
from typing import Iterable

import torch
from torch import Tensor


SET_SIZES: tuple[int, ...] = (
    35, 35, 35, 35, 35, 34, 34, 34, 34, 34, 34, 33, 33, 33, 33, 32,
    32, 32, 31, 31, 31, 30, 30, 29, 29, 28, 28, 27, 27, 26, 26, 25,
    25, 24, 23, 23, 22, 21, 21, 20, 19, 19, 18, 17, 17, 16, 15, 14,
    14, 13, 12, 11, 10, 10, 9, 8, 7, 6, 6, 5, 4, 3, 2, 1,
)
LAYOUT_VERSION = "fps_v1_h30_w48_s64"


@dataclass(frozen=True)
class ProgressiveLayout:
    height: int
    width: int
    set_sizes: tuple[int, ...]
    traversal: tuple[int, ...]
    set_ids_flat: tuple[int, ...]
    version: str
    checksum: str

    def set_ids_tensor(self, *, device=None) -> Tensor:
        return torch.tensor(self.set_ids_flat, dtype=torch.long, device=device).reshape(
            self.height, self.width
        )


def validate_set_sizes(set_sizes: Iterable[int], token_count: int) -> tuple[int, ...]:
    sizes = tuple(int(x) for x in set_sizes)
    if any(x <= 0 for x in sizes):
        raise ValueError("Every progressive set must be non-empty")
    if any(a < b for a, b in zip(sizes, sizes[1:])):
        raise ValueError("Progressive set sizes must be monotonically non-increasing")
    if sum(sizes) != token_count:
        raise ValueError(f"Set sizes sum to {sum(sizes)}, expected {token_count}")
    return sizes


def farthest_point_traversal(height: int, width: int) -> tuple[int, ...]:
    """Deterministic center-first farthest-point traversal with row-major ties."""

    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    count = height * width
    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0
    first = min(
        range(count),
        key=lambda idx: ((idx // width - center_y) ** 2 + (idx % width - center_x) ** 2, idx),
    )
    traversal = [first]
    selected = [False] * count
    selected[first] = True
    min_dist = [float("inf")] * count

    while len(traversal) < count:
        last_y, last_x = divmod(traversal[-1], width)
        for idx in range(count):
            if selected[idx]:
                continue
            y, x = divmod(idx, width)
            distance = float((y - last_y) ** 2 + (x - last_x) ** 2)
            if distance < min_dist[idx]:
                min_dist[idx] = distance
        best_distance = max(min_dist[idx] for idx in range(count) if not selected[idx])
        next_index = min(
            idx for idx in range(count) if not selected[idx] and min_dist[idx] == best_distance
        )
        selected[next_index] = True
        traversal.append(next_index)
    return tuple(traversal)


def build_progressive_layout(
    height: int = 30,
    width: int = 48,
    set_sizes: Iterable[int] = SET_SIZES,
    version: str = LAYOUT_VERSION,
) -> ProgressiveLayout:
    sizes = validate_set_sizes(set_sizes, height * width)
    traversal = farthest_point_traversal(height, width)
    set_ids = [-1] * (height * width)
    cursor = 0
    for set_id, size in enumerate(sizes):
        for flat_index in traversal[cursor : cursor + size]:
            set_ids[flat_index] = set_id
        cursor += size
    if any(x < 0 for x in set_ids):
        raise RuntimeError("Progressive layout assignment is incomplete")
    digest = hashlib.sha256()
    digest.update(version.encode("utf-8"))
    digest.update(struct.pack(f"<{len(sizes)}H", *sizes))
    digest.update(struct.pack(f"<{len(set_ids)}H", *set_ids))
    return ProgressiveLayout(
        height=height,
        width=width,
        set_sizes=sizes,
        traversal=traversal,
        set_ids_flat=tuple(set_ids),
        version=version,
        checksum=digest.hexdigest(),
    )


def build_causal_attention_mask(set_ids: Tensor) -> Tensor:
    flat = set_ids.reshape(-1)
    query_sets = flat[:, None]
    key_sets = flat[None, :]
    return key_sets > query_sets  # True means disallowed for PyTorch MHA.


def build_prefix_mask(set_ids: Tensor, prefix_len: int) -> Tensor:
    num_sets = int(set_ids.max().item()) + 1
    if prefix_len < 1 or prefix_len > num_sets:
        raise ValueError(f"prefix_len must be in [1, {num_sets}], got {prefix_len}")
    return set_ids < prefix_len

