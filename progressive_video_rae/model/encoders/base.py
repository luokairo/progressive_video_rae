from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
import sys
from typing import Iterator, Sequence

import torch
from torch import Tensor, nn

from ..pretrained import PretrainedLoadReport
from ..types import EncoderOutput


@contextmanager
def upstream_import_path(source_root: str | Path | None) -> Iterator[None]:
    """Temporarily prepend an explicitly configured upstream checkout."""

    if source_root is None:
        yield
        return
    root = str(Path(source_root).expanduser().resolve())
    if not Path(root).is_dir():
        raise FileNotFoundError(f"Upstream source root does not exist: {root}")
    inserted = root not in sys.path
    if inserted:
        sys.path.insert(0, root)
    try:
        yield
    finally:
        if inserted and sys.path and sys.path[0] == root:
            sys.path.pop(0)


def clean_state_dict_keys(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    cleaned: dict[str, Tensor] = {}
    for key, value in state_dict.items():
        for prefix in ("module.", "backbone.", "encoder."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        cleaned[key] = value
    return cleaned


class VideoFoundationEncoder(nn.Module, ABC):
    """Uniform interface for frozen V-JEPA2 and VideoMAEv2 encoders."""

    output_layers: tuple[int, ...]
    load_report: PretrainedLoadReport | None

    @abstractmethod
    def forward(
        self,
        pixel_values: Tensor,
        output_layers: Sequence[int] | None = None,
    ) -> EncoderOutput:
        raise NotImplementedError

    def freeze_backbone(self) -> None:
        self.eval()
        self.requires_grad_(False)

    @staticmethod
    def _validate_layers(layers: Sequence[int], depth: int) -> tuple[int, ...]:
        normalized = tuple(int(x) for x in layers)
        if not normalized or any(x < 1 or x > depth for x in normalized):
            raise ValueError(f"output_layers must use 1-based indices in [1, {depth}], got {normalized}")
        if tuple(sorted(set(normalized))) != normalized:
            raise ValueError(f"output_layers must be unique and increasing, got {normalized}")
        return normalized

    @staticmethod
    def _load_checkpoint(path: str | Path) -> dict:
        checkpoint_path = Path(path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Encoder checkpoint not found: {checkpoint_path}")
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
