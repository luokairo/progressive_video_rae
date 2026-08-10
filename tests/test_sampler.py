import pytest

torch = pytest.importorskip("torch")

from progressive_videorae.data.sampler import DistributedFullDatasetSampler


class DummyDataset:
    def __len__(self):
        return 25


def test_distributed_sampler_uses_all_sources_without_overlap():
    dataset = DummyDataset()
    rank0 = list(DistributedFullDatasetSampler(dataset, rank=0, world_size=2, seed=9))
    rank1 = list(DistributedFullDatasetSampler(dataset, rank=1, world_size=2, seed=9))
    assert set(rank0).isdisjoint(rank1)
    assert len(rank0) == len(rank1) == 12
    assert len(set(rank0 + rank1)) == 24


def test_epoch_shuffle_is_deterministic_and_changes_tail_membership():
    sampler = DistributedFullDatasetSampler(DummyDataset(), rank=0, world_size=1, seed=9)
    first = list(sampler)
    sampler.set_epoch(1)
    second = list(sampler)
    assert first != second
    duplicate = DistributedFullDatasetSampler(DummyDataset(), rank=0, world_size=1, seed=9)
    duplicate.set_epoch(1)
    assert second == list(duplicate)
