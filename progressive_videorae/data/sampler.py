from __future__ import annotations

import math
from typing import Iterator, Sized

import torch
from torch.utils.data import Sampler


class BalancedHumanNonSpeechSampler(Sampler[int]):
    """Shard balanced human/non-speech pairs without cross-rank duplication."""

    def __init__(
        self,
        dataset: Sized,
        *,
        seed: int = 20260807,
        rank: int | None = None,
        world_size: int | None = None,
    ) -> None:
        if not hasattr(dataset, "category_to_indices"):
            raise TypeError("dataset must expose category_to_indices")
        self.human = list(dataset.category_to_indices["human"])
        self.non_speech = list(dataset.category_to_indices["non_speech"])
        if not self.human or len(self.non_speech) < len(self.human):
            raise ValueError("Balanced sampling requires non-empty human data and at least as much non-speech data")
        if rank is None or world_size is None:
            distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
            rank = torch.distributed.get_rank() if distributed else 0
            world_size = torch.distributed.get_world_size() if distributed else 1
        self.rank = int(rank)
        self.world_size = int(world_size)
        if not 0 <= self.rank < self.world_size:
            raise ValueError("rank must be in [0, world_size)")
        self.seed = int(seed)
        self.epoch = 0
        self.pairs_per_rank = len(self.human) // self.world_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return 2 * self.pairs_per_rank

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        human_order = torch.randperm(len(self.human), generator=generator).tolist()
        non_order = torch.randperm(len(self.non_speech), generator=generator)[: len(self.human)].tolist()
        pair_order = torch.randperm(len(self.human), generator=generator).tolist()
        pairs = [
            (self.human[human_order[index]], self.non_speech[non_order[index]])
            for index in pair_order
        ]
        usable = self.pairs_per_rank * self.world_size
        local_pairs = pairs[:usable][self.rank :: self.world_size]
        flattened = []
        for pair_index, (human_index, non_index) in enumerate(local_pairs):
            if pair_index % 2:
                flattened.extend((non_index, human_index))
            else:
                flattened.extend((human_index, non_index))
        return iter(flattened)

