from __future__ import annotations

from dataclasses import asdict, dataclass, field
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Mapping, Sequence

from torch import Tensor, nn


@dataclass(frozen=True)
class ShapeMismatch:
    key: str
    checkpoint_shape: tuple[int, ...]
    model_shape: tuple[int, ...]


@dataclass
class PretrainedLoadReport:
    """Auditable result of loading one pretrained model component."""

    component: str
    checkpoint_path: str
    minimum_coverage: float
    loaded_keys: list[str] = field(default_factory=list)
    missing_keys: list[str] = field(default_factory=list)
    unexpected_keys: list[str] = field(default_factory=list)
    shape_mismatches: list[ShapeMismatch] = field(default_factory=list)
    allowed_missing_keys: list[str] = field(default_factory=list)
    ignored_checkpoint_keys: list[str] = field(default_factory=list)
    allowed_shape_mismatches: list[ShapeMismatch] = field(default_factory=list)
    missing_required_groups: list[str] = field(default_factory=list)
    loaded_numel: int = 0
    expected_numel: int = 0

    @property
    def coverage(self) -> float:
        if self.expected_numel == 0:
            return 0.0
        return self.loaded_numel / self.expected_numel

    @property
    def ready(self) -> bool:
        return (
            bool(self.loaded_keys)
            and self.coverage >= self.minimum_coverage
            and not self.shape_mismatches
            and not self.missing_required_groups
        )

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["coverage"] = self.coverage
        payload["ready"] = self.ready
        return payload

    def assert_ready(self) -> None:
        if self.ready:
            return
        problems = []
        if not self.loaded_keys:
            problems.append("no compatible tensors were loaded")
        if self.coverage < self.minimum_coverage:
            problems.append(
                f"coverage {self.coverage:.4f} is below {self.minimum_coverage:.4f} "
                f"({self.loaded_numel}/{self.expected_numel} numel)"
            )
        if self.shape_mismatches:
            problems.append(f"{len(self.shape_mismatches)} non-whitelisted shape mismatches")
        if self.missing_required_groups:
            problems.append(f"missing required groups {self.missing_required_groups}")
        raise RuntimeError(
            f"Pretrained validation failed for {self.component} from "
            f"{self.checkpoint_path}: " + "; ".join(problems)
        )


def _matches(key: str, patterns: Sequence[str]) -> bool:
    return any(fnmatchcase(key, pattern) for pattern in patterns)


def load_validated_pretrained(
    module: nn.Module,
    checkpoint_state: Mapping[str, Tensor],
    *,
    component: str,
    checkpoint_path: str | Path,
    minimum_coverage: float = 0.95,
    required_groups: Mapping[str, Sequence[str]] | None = None,
    allowed_missing_patterns: Sequence[str] = (),
    ignored_checkpoint_patterns: Sequence[str] = (),
) -> PretrainedLoadReport:
    """Load only shape-compatible tensors and reject incomplete or wrong checkpoints.

    Patterns use shell-style matching. Expected omissions, such as a resolution-specific
    VideoMAEv2 positional embedding, are visible in the report but excluded from the
    coverage denominator. Missing and unexpected keys remain visible in the report, but
    normal upstream checkpoint differences do not block startup when enough weights load.
    """

    if not 0.0 < minimum_coverage <= 1.0:
        raise ValueError(f"minimum_coverage must be in (0, 1], got {minimum_coverage}")
    if not isinstance(checkpoint_state, Mapping):
        raise TypeError(f"{component} checkpoint state must be a mapping")

    model_state = module.state_dict()
    compatible: dict[str, Tensor] = {}
    ignored_checkpoint_keys: list[str] = []
    unexpected_keys: list[str] = []
    shape_mismatches: list[ShapeMismatch] = []
    allowed_shape_mismatches: list[ShapeMismatch] = []

    for key, value in checkpoint_state.items():
        if not isinstance(key, str) or not isinstance(value, Tensor):
            continue
        expected = model_state.get(key)
        if expected is None:
            if _matches(key, ignored_checkpoint_patterns):
                ignored_checkpoint_keys.append(key)
            else:
                unexpected_keys.append(key)
            continue
        if tuple(value.shape) != tuple(expected.shape):
            mismatch = ShapeMismatch(key, tuple(value.shape), tuple(expected.shape))
            if _matches(key, allowed_missing_patterns):
                allowed_shape_mismatches.append(mismatch)
            else:
                shape_mismatches.append(mismatch)
            continue
        compatible[key] = value

    missing = [key for key in model_state if key not in compatible]
    allowed_missing = [key for key in missing if _matches(key, allowed_missing_patterns)]
    incompatible_missing = [key for key in missing if key not in allowed_missing]

    expected_keys = [key for key in model_state if key not in allowed_missing]
    expected_numel = sum(model_state[key].numel() for key in expected_keys)
    loaded_numel = sum(model_state[key].numel() for key in compatible if key in expected_keys)

    required_groups = required_groups or {}
    missing_required_groups = []
    for group, patterns in required_groups.items():
        if not any(_matches(key, patterns) for key in compatible):
            missing_required_groups.append(group)

    report = PretrainedLoadReport(
        component=component,
        checkpoint_path=str(Path(checkpoint_path).expanduser().resolve()),
        minimum_coverage=float(minimum_coverage),
        loaded_keys=sorted(compatible),
        missing_keys=sorted(incompatible_missing),
        unexpected_keys=sorted(unexpected_keys),
        shape_mismatches=sorted(shape_mismatches, key=lambda item: item.key),
        allowed_missing_keys=sorted(allowed_missing),
        ignored_checkpoint_keys=sorted(ignored_checkpoint_keys),
        allowed_shape_mismatches=sorted(allowed_shape_mismatches, key=lambda item: item.key),
        missing_required_groups=sorted(missing_required_groups),
        loaded_numel=int(loaded_numel),
        expected_numel=int(expected_numel),
    )
    report.assert_ready()
    module.load_state_dict(compatible, strict=False)
    return report
