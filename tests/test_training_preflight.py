from __future__ import annotations

from pathlib import Path

import pytest
from torch import nn

from progressive_video_rae.model.pretrained import load_validated_pretrained
from progressive_video_rae.train import preflight_pretrained_weights


class StubModel:
    def __init__(self, ready: bool, checkpoint_path: Path):
        module = nn.Linear(2, 2)
        state = {key: value.detach().clone() for key, value in module.state_dict().items()}
        if not ready:
            state.pop("weight")
        self.report = None
        try:
            self.report = load_validated_pretrained(
                module,
                state,
                component="stub",
                checkpoint_path=checkpoint_path,
                minimum_coverage=1.0,
                required_groups={"input": ("weight",)},
            )
        except RuntimeError:
            # Keep a failed report-like object so preflight still exercises its gate.
            self.error = True
        else:
            self.error = False

    def assert_pretrained_ready(self) -> None:
        if self.error:
            raise RuntimeError("stub pretrained validation failed")
        assert self.report is not None

    def pretrained_load_report(self) -> dict:
        return {
            "ready": not self.error,
            "components": {"stub": self.report.to_dict() if self.report is not None else None},
            "random_initialized_components": [],
        }


def test_preflight_writes_report_before_training(tmp_path: Path):
    model = StubModel(ready=True, checkpoint_path=tmp_path / "encoder.pt")
    output = tmp_path / "run" / "pretrained_load_report.json"
    report = preflight_pretrained_weights(model, output)
    assert report["ready"] is True
    assert output.is_file()
    assert '"coverage"' in output.read_text(encoding="utf-8")


def test_preflight_rejects_incomplete_weights_before_optimizer(tmp_path: Path):
    model = StubModel(ready=False, checkpoint_path=tmp_path / "broken.pt")

    with pytest.raises(RuntimeError, match="pretrained validation failed"):
        preflight_pretrained_weights(model, tmp_path / "report.json")
    assert not (tmp_path / "report.json").exists()
