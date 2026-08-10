from __future__ import annotations

from typing import Iterator, Sized

import torch
from torch.utils.data import Sampler


class DistributedFullDatasetSampler(Sampler[int]):
    """Shuffle every valid video without source balancing or cross-rank duplication."""

    def __init__(
        self,
        dataset: Sized,
        *,
        seed: int = 20260807,
        rank: int | None = None,
        world_size: int | None = None,
        samples_per_rank_multiple: int = 1,
    ) -> None:
        if rank is None or world_size is None:
            distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
            rank = torch.distributed.get_rank() if distributed else 0
            world_size = torch.distributed.get_world_size() if distributed else 1
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.epoch = 0
        self.multiple = int(samples_per_rank_multiple)
        self.dataset_size = len(dataset)
        if not 0 <= self.rank < self.world_size:
            raise ValueError("rank must be in [0, world_size)")
        if self.multiple <= 0:
            raise ValueError("samples_per_rank_multiple must be positive")
        unit = self.world_size * self.multiple
        self.usable = len(dataset) - len(dataset) % unit
        if self.usable == 0:
            raise ValueError("Dataset is too small for the requested distributed topology")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.usable // self.world_size

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.dataset_size, generator=generator)[: self.usable].tolist()
        return iter(order[self.rank :: self.world_size])


# Import compatibility only; training no longer uses category-balanced sampling.
BalancedHumanNonSpeechSampler = DistributedFullDatasetSampler
