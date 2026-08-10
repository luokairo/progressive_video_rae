from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from progressive_videorae.model.pretrained import load_validated_pretrained
from progressive_videorae.train import preflight_pretrained_weights
from progressive_videorae.training.checkpoint import (
    checkpoint_model_state,
    load_decoder_from_checkpoint,
    load_model_state,
)
from progressive_videorae.model.types import StateContract


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


class TinyTrainingModel(nn.Module):
    def __init__(self, encoder_dim: int = 2):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.backbone = nn.Linear(encoder_dim, encoder_dim)
        self.encoder.frozen = True
        self.projector = nn.Linear(encoder_dim, 2)
        self.decoder = nn.Linear(2, 2)
        self.state_contract = StateContract()


def test_checkpoint_state_excludes_frozen_encoder_and_restores_external_encoder():
    source = TinyTrainingModel()
    state, excluded = checkpoint_model_state(source)
    assert excluded == ["encoder.backbone."]
    assert not any(key.startswith("encoder.backbone.") for key in state)

    target = TinyTrainingModel()
    encoder_before = target.encoder.backbone.weight.detach().clone()
    load_model_state(target, state)
    assert torch.equal(target.encoder.backbone.weight, encoder_before)
    assert torch.equal(target.projector.weight, source.projector.weight)


def test_decoder_only_initialization_does_not_replace_encoder_or_projector(tmp_path: Path):
    source = TinyTrainingModel()
    target = TinyTrainingModel()
    with torch.no_grad():
        source.encoder.backbone.weight.fill_(1.0)
        source.projector.weight.fill_(2.0)
        source.decoder.weight.fill_(3.0)
        target.encoder.backbone.weight.fill_(4.0)
        target.projector.weight.fill_(5.0)
        target.decoder.weight.fill_(6.0)
    checkpoint_path = tmp_path / "vitl_training.pt"
    torch.save({"model": source.state_dict(), "state_contract": source.state_contract.to_dict()}, checkpoint_path)

    load_decoder_from_checkpoint(checkpoint_path, model=target)

    assert torch.all(target.encoder.backbone.weight == 4.0)
    assert torch.all(target.projector.weight == 5.0)
    assert torch.all(target.decoder.weight == 3.0)


def test_full_checkpoint_rejects_cross_variant_shape_change():
    source = TinyTrainingModel(encoder_dim=2)
    state, _ = checkpoint_model_state(source)
    target = TinyTrainingModel(encoder_dim=3)
    with pytest.raises(RuntimeError):
        load_model_state(target, state)


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
