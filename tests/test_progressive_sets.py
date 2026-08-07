import pytest

torch = pytest.importorskip("torch")

from progressive_video_rae.model.progressive_sets import (
    SET_SIZES,
    build_causal_attention_mask,
    build_progressive_layout,
)


def test_production_layout_is_fixed_and_complete():
    layout = build_progressive_layout()
    assert len(SET_SIZES) == 64
    assert sum(SET_SIZES) == 1440
    assert layout.traversal[0] == 14 * 48 + 23
    assert len(set(layout.traversal)) == 1440
    assert min(layout.set_ids_flat) == 0
    assert max(layout.set_ids_flat) == 63
    assert layout.checksum == build_progressive_layout().checksum


def test_causal_mask_allows_same_and_earlier_sets():
    set_ids = torch.tensor([[0, 0, 1, 2]])
    mask = build_causal_attention_mask(set_ids)
    assert not mask[0, 1]
    assert mask[0, 2]
    assert not mask[2, 0]
    assert not mask[2, 2]

