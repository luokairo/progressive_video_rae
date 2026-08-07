import pytest

torch = pytest.importorskip("torch")

from progressive_video_rae.data.sampler import BalancedHumanNonSpeechSampler


class DummyDataset:
    category_to_indices = {
        "human": list(range(8)),
        "non_speech": list(range(8, 24)),
    }

    def __len__(self):
        return 24


def test_distributed_sampler_is_balanced_and_non_overlapping():
    dataset = DummyDataset()
    rank0 = list(BalancedHumanNonSpeechSampler(dataset, rank=0, world_size=2, seed=9))
    rank1 = list(BalancedHumanNonSpeechSampler(dataset, rank=1, world_size=2, seed=9))
    assert set(rank0).isdisjoint(rank1)
    for indices in (rank0, rank1):
        assert sum(index < 8 for index in indices) == sum(index >= 8 for index in indices)


def test_epoch_changes_non_speech_draw_deterministically():
    sampler = BalancedHumanNonSpeechSampler(DummyDataset(), rank=0, world_size=1, seed=9)
    first = list(sampler)
    sampler.set_epoch(1)
    second = list(sampler)
    assert first != second

