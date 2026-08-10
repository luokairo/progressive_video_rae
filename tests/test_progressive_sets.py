import pytest

torch = pytest.importorskip("torch")

from progressive_videorae.model.progressive_sets import (
    SET_SIZES,
    build_causal_attention_mask,
    build_prefix_mask,
    build_progressive_layout,
)


def test_production_layout_is_48_equal_sets_and_complete():
    layout = build_progressive_layout()
    assert SET_SIZES == (30,) * 48
    assert sum(SET_SIZES) == 1440
    assert layout.traversal[0] == 14 * 48 + 23
    assert len(set(layout.traversal)) == 1440
    assert min(layout.set_ids_flat) == 0
    assert max(layout.set_ids_flat) == 47
    assert layout.checksum == build_progressive_layout().checksum


def test_causal_mask_and_inclusive_prefix_endpoint():
    set_ids = torch.tensor([[0, 0, 1, 2]])
    mask = build_causal_attention_mask(set_ids)
    assert not mask[0, 1]
    assert mask[0, 2]
    assert not mask[2, 0]
    assert build_prefix_mask(set_ids, 1).tolist() == [[True, True, True, False]]
