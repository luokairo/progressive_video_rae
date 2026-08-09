from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor


_LOGIT_STAT_COUNT = 6


def timestamped_log_path(
    log_dir: str | Path, stage: str, *, now: datetime | None = None
) -> Path:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ValueError("Timestamp must be timezone-aware")
    timestamp = current.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return Path(log_dir) / f"{stage}_{timestamp}.train.jsonl"


def resolve_training_log_file(
    training: Mapping[str, Any],
    output_dir: str | Path,
    *,
    resume_log_file: str | Path | None = None,
    now: datetime | None = None,
) -> Path:
    """Choose a log file without changing checkpoint/output directory semantics.

    Callers pass ``resume_log_file`` only for ``--resume``. Consequently an
    ``--init-from`` run always receives a new timestamped Stage 1 log, even
    when its source checkpoint recorded a previous log file.
    """

    if resume_log_file:
        return Path(resume_log_file)
    if "log_dir" in training:
        return timestamped_log_path(training["log_dir"], str(training["stage"]), now=now)
    return Path(output_dir) / "train.jsonl"


def append_jsonl_record(
    log_file: Path | None, record: Mapping[str, Any], *, rank: int
) -> bool:
    """Append one record on rank zero only; return whether a write occurred."""

    if rank != 0:
        return False
    if log_file is None:
        raise ValueError("rank zero requires a training log file")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
    return True


def gradient_norm(parameters: Iterable[Tensor]) -> Tensor | None:
    squared_norm: Tensor | None = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float().square().sum()
        squared_norm = value if squared_norm is None else squared_norm + value
    return None if squared_norm is None else squared_norm.sqrt()


class GlobalMetricWindow:
    """Accumulate scalar metrics locally, then reduce a complete log window."""

    def __init__(
        self,
        metric_names: Iterable[str],
        *,
        device: torch.device,
        prefix_bins: int = 64,
    ) -> None:
        names = tuple(metric_names)
        if len(names) != len(set(names)):
            raise ValueError("Metric names must be unique")
        if prefix_bins <= 0:
            raise ValueError("prefix_bins must be positive")
        self.names = names
        self._index = {name: index for index, name in enumerate(names)}
        self.device = device
        self.prefix_bins = prefix_bins
        self._sums = torch.zeros(len(names), dtype=torch.float64, device=device)
        self._counts = torch.zeros(len(names), dtype=torch.float64, device=device)
        self._histogram = torch.zeros(prefix_bins, dtype=torch.float64, device=device)
        self._logits = torch.zeros(_LOGIT_STAT_COUNT, dtype=torch.float64, device=device)
        self.steps = 0

    def _scalar(self, value: Tensor | float) -> Tensor:
        if isinstance(value, Tensor):
            return value.detach().to(device=self.device, dtype=torch.float64).reshape(())
        return torch.tensor(float(value), device=self.device, dtype=torch.float64)

    def add_mean(self, name: str, value: Tensor | float, *, count: int | float = 1) -> None:
        if name not in self._index:
            raise KeyError(f"Unknown metric: {name}")
        if count < 0:
            raise ValueError("Metric count must be non-negative")
        if count == 0:
            return
        index = self._index[name]
        weight = float(count)
        self._sums[index].add_(self._scalar(value) * weight)
        self._counts[index].add_(weight)

    def add_prefix(self, prefix_len: int, *, count: int | float = 1) -> None:
        if prefix_len < 1 or prefix_len > self.prefix_bins:
            raise ValueError(f"prefix_len must be in [1, {self.prefix_bins}]")
        if count < 0:
            raise ValueError("Prefix count must be non-negative")
        self._histogram[prefix_len - 1].add_(float(count))

    def add_logits(self, real_logits: Tensor, fake_logits: Tensor) -> None:
        for offset, logits in ((0, real_logits), (3, fake_logits)):
            values = logits.detach().to(device=self.device, dtype=torch.float64)
            self._logits[offset].add_(values.sum())
            self._logits[offset + 1].add_(values.square().sum())
            self._logits[offset + 2].add_(values.numel())

    def step(self) -> None:
        self.steps += 1

    def reduce(self) -> tuple[dict[str, float], dict[str, int]]:
        payload = torch.cat((self._sums, self._counts, self._histogram, self._logits))
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(payload, op=torch.distributed.ReduceOp.SUM)

        split = len(self.names)
        sums = payload[:split]
        counts = payload[split : 2 * split]
        histogram = payload[2 * split : 2 * split + self.prefix_bins]
        logits = payload[2 * split + self.prefix_bins :]

        metrics = {
            name: float((sums[index] / counts[index]).cpu())
            for index, name in enumerate(self.names)
            if counts[index] > 0
        }
        for name, offset in (("real", 0), ("fake", 3)):
            total, squared_total, count = logits[offset : offset + 3]
            if count > 0:
                mean = total / count
                variance = (squared_total / count - mean.square()).clamp_min(0.0)
                metrics[f"disc/{name}_logit_mean"] = float(mean.cpu())
                metrics[f"disc/{name}_logit_std"] = float(variance.sqrt().cpu())

        prefix_histogram = {
            str(index + 1): int(value.item())
            for index, value in enumerate(histogram)
            if value > 0
        }
        return metrics, prefix_histogram

    def reset(self) -> None:
        self._sums.zero_()
        self._counts.zero_()
        self._histogram.zero_()
        self._logits.zero_()
        self.steps = 0

