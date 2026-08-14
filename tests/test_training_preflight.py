from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from progressive_videorae.model.pretrained import load_validated_pretrained
import progressive_videorae.train as train_module
from progressive_videorae.checksums import sha256_file
from progressive_videorae.train import preflight_pretrained_weights
from progressive_videorae.training.checkpoint import (
    checkpoint_model_state,
    load_checkpoint,
    load_decoder_from_checkpoint,
    load_model_state,
    representation_identity,
    save_checkpoint,
    verify_pretrained_checkpoint_hashes,
    validate_checkpoint_transition,
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
    torch.save(
        {
            "model": source.state_dict(),
            "state_contract": source.state_contract.to_dict(),
        },
        checkpoint_path,
    )

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


class IdentityTrainingModel(nn.Module):
    def __init__(self, encoder_path: Path, wan_path: Path):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.backbone = nn.Linear(2, 2)
        self.encoder.frozen = True
        self.encoder.variant = "vitl"
        self.encoder.output_layers = (8, 12, 16, 20, 24)
        self.encoder.load_report = SimpleNamespace(checkpoint_path=str(encoder_path.resolve()))
        contract = StateContract()
        self.projector = nn.Module()
        self.projector.main = nn.Linear(2, 2)
        self.projector.layout_checksum = contract.layout_checksum
        self.projector.layout_version = contract.layout_version
        self.decoder = nn.Module()
        self.decoder.temporal_adapter = nn.Linear(2, 2)
        self.decoder.backend = nn.Linear(2, 2)
        self.decoder.codec_id = "codec-v4"
        self.decoder.decoder_id = "decoder-v4"
        self.decoder.load_report = SimpleNamespace(
            decoder=SimpleNamespace(checkpoint_path=str(wan_path.resolve()))
        )
        self.state_contract = contract

    def pretrained_load_report(self):
        return {"ready": True}


def _identity_files(tmp_path: Path):
    encoder = tmp_path / "encoder.pt"
    wan = tmp_path / "wan.pt"
    encoder.write_bytes(b"encoder")
    wan.write_bytes(b"wan")
    for path in (encoder, wan):
        path.with_suffix(".pt.sha256").write_text(
            f"{sha256_file(path)}  {path.name}\n", encoding="utf-8"
        )
    return encoder, wan


def _save_identity_checkpoint(
    path: Path,
    *,
    model: IdentityTrainingModel,
    optimizer_step: int = 10,
    max_steps: int = 10,
):
    discriminator = nn.Linear(2, 1)
    generator_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    discriminator_optimizer = torch.optim.SGD(discriminator.parameters(), lr=0.1)
    generator_scheduler = torch.optim.lr_scheduler.LambdaLR(
        generator_optimizer, lambda _step: 1.0
    )
    discriminator_scheduler = torch.optim.lr_scheduler.LambdaLR(
        discriminator_optimizer, lambda _step: 1.0
    )
    hashes = verify_pretrained_checkpoint_hashes(
        model, create_missing_sidecars=False
    )
    static_identity = representation_identity(model, checkpoint_hashes=hashes)
    save_checkpoint(
        path,
        model=model,
        discriminator=discriminator,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator_scheduler=generator_scheduler,
        discriminator_scheduler=discriminator_scheduler,
        optimizer_step=optimizer_step,
        discriminator_update_count=0,
        epoch=0,
        config={"training": {"max_steps": max_steps}},
        stage="stage1a",
        objective_mode="full_repa",
        static_identity=static_identity,
    )
    return static_identity


@pytest.mark.parametrize(
    ("target", "source"),
    (("stage1b", "stage1a"), ("stage2a", "stage1b"), ("stage2b", "stage2a")),
)
def test_checkpoint_transition_accepts_only_adjacent_completed_stage(target, source):
    checkpoint = {
        "checkpoint_schema_version": 4,
        "stage": source,
        "objective_mode": {
            "stage1a": "full_repa",
            "stage1b": "nested_spectral_hrepa_full_anchor",
            "stage2a": "full_repa",
        }[source],
        "optimizer_step": 10,
        "stage_max_steps": 10,
        "stage_complete": True,
    }

    validate_checkpoint_transition(
        checkpoint,
        target_stage=target,
        target_objective_mode="unused-for-init",
        mode="init",
        allow_smoke_checkpoint=False,
    )

    checkpoint["stage"] = "stage1a"
    if source != "stage1a":
        with pytest.raises(RuntimeError, match="must initialize"):
            validate_checkpoint_transition(
                checkpoint,
                target_stage=target,
                target_objective_mode="unused-for-init",
                mode="init",
                allow_smoke_checkpoint=False,
            )


@pytest.mark.parametrize("legacy_objective", ["full_repa_spatial_prefix", "nested_spectral_prefix"])
def test_stage2a_rejects_legacy_stage1b_objective(legacy_objective):
    checkpoint = {
        "checkpoint_schema_version": 4,
        "stage": "stage1b",
        "objective_mode": legacy_objective,
        "optimizer_step": 90000,
        "stage_max_steps": 90000,
        "stage_complete": True,
    }

    with pytest.raises(RuntimeError, match="stage1b checkpoint objective mismatch"):
        validate_checkpoint_transition(
            checkpoint,
            target_stage="stage2a",
            target_objective_mode="full_repa",
            mode="init",
            allow_smoke_checkpoint=False,
        )


def test_incomplete_and_schema3_checkpoints_require_explicit_smoke_mode():
    incomplete = {
        "checkpoint_schema_version": 4,
        "stage": "stage1a",
        "objective_mode": "full_repa",
        "optimizer_step": 5,
        "stage_max_steps": 10,
        "stage_complete": False,
    }
    with pytest.raises(RuntimeError, match="completed"):
        validate_checkpoint_transition(
            incomplete,
            target_stage="stage1b",
            target_objective_mode="nested_spectral_hrepa_full_anchor",
            mode="init",
            allow_smoke_checkpoint=False,
        )
    validate_checkpoint_transition(
        incomplete,
        target_stage="stage1b",
        target_objective_mode="nested_spectral_hrepa_full_anchor",
        mode="init",
        allow_smoke_checkpoint=True,
    )

    legacy = {
        "checkpoint_schema_version": 3,
        "stage": "stage1a",
        "objective_mode": "full_repa",
    }
    with pytest.raises(RuntimeError, match="schema"):
        validate_checkpoint_transition(
            legacy,
            target_stage="stage1b",
            target_objective_mode="nested_spectral_hrepa_full_anchor",
            mode="init",
            allow_smoke_checkpoint=False,
        )
    validate_checkpoint_transition(
        legacy,
        target_stage="stage1b",
        target_objective_mode="nested_spectral_hrepa_full_anchor",
        mode="init",
        allow_smoke_checkpoint=True,
    )


@pytest.mark.parametrize(
    ("tensor_prefix", "error"),
    (
        ("projector.", "projector_state_sha256"),
        ("decoder.temporal_adapter.", "bridge_state_sha256"),
    ),
)
def test_checkpoint_rejects_tampered_projector_or_bridge_hash(
    tmp_path: Path, tensor_prefix: str, error: str
):
    encoder, wan = _identity_files(tmp_path)
    source = IdentityTrainingModel(encoder, wan)
    checkpoint_path = tmp_path / "stage1a.pt"
    static_identity = _save_identity_checkpoint(checkpoint_path, model=source)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    key = next(key for key in payload["model"] if key.startswith(tensor_prefix))
    payload["model"][key] = payload["model"][key] + 1.0
    torch.save(payload, checkpoint_path)

    target = IdentityTrainingModel(encoder, wan)
    with pytest.raises(RuntimeError, match=error):
        load_checkpoint(
            checkpoint_path,
            model=target,
            restore_rng=False,
            static_identity=static_identity,
        )


def test_checkpoint_rejects_changed_external_static_hash(tmp_path: Path):
    encoder, wan = _identity_files(tmp_path)
    source = IdentityTrainingModel(encoder, wan)
    checkpoint_path = tmp_path / "stage1a.pt"
    _save_identity_checkpoint(checkpoint_path, model=source)
    encoder.write_bytes(b"changed encoder")
    encoder.with_suffix(".pt.sha256").write_text(
        f"{sha256_file(encoder)}  {encoder.name}\n", encoding="utf-8"
    )

    target = IdentityTrainingModel(encoder, wan)
    changed_hashes = verify_pretrained_checkpoint_hashes(
        target, create_missing_sidecars=False
    )
    changed_identity = representation_identity(target, checkpoint_hashes=changed_hashes)
    with pytest.raises(RuntimeError, match="static representation identity"):
        load_checkpoint(
            checkpoint_path,
            model=target,
            restore_rng=False,
            static_identity=changed_identity,
        )


def test_nonzero_rank_consumes_broadcast_hashes_without_reading_files(tmp_path, monkeypatch):
    encoder, wan = _identity_files(tmp_path)
    model = IdentityTrainingModel(encoder, wan)
    hashes = {
        "vjepa_checkpoint_sha256": "1" * 64,
        "wan_checkpoint_sha256": "2" * 64,
    }
    calls = []

    def fake_verify(*_args, **_kwargs):
        calls.append("hashed")
        return hashes

    monkeypatch.setattr(train_module, "verify_pretrained_checkpoint_hashes", fake_verify)
    monkeypatch.setattr(
        train_module,
        "broadcast_rank0_object",
        lambda _value, *, rank: {"ok": True, "hashes": hashes},
    )
    identity = train_module.distributed_verified_representation_identity(
        model, rank=1
    )
    assert calls == []
    assert identity["vjepa_checkpoint_sha256"] == "1" * 64


def test_rank0_hash_error_is_broadcast_before_all_ranks_fail(tmp_path, monkeypatch):
    encoder, wan = _identity_files(tmp_path)
    model = IdentityTrainingModel(encoder, wan)
    broadcasted = []

    def fail_hash(*_args, **_kwargs):
        raise RuntimeError("digest mismatch")

    def capture(value, *, rank):
        broadcasted.append((rank, value))
        return value

    monkeypatch.setattr(train_module, "verify_pretrained_checkpoint_hashes", fail_hash)
    monkeypatch.setattr(train_module, "broadcast_rank0_object", capture)
    with pytest.raises(RuntimeError, match="Rank 0 pretrained SHA256 preflight failed"):
        train_module.distributed_verified_representation_identity(model, rank=0)
    assert broadcasted[0][0] == 0
    assert "digest mismatch" in broadcasted[0][1]["error"]


def test_schema4_checkpoint_missing_new_contract_field_is_rejected(tmp_path: Path):
    encoder, wan = _identity_files(tmp_path)
    source = IdentityTrainingModel(encoder, wan)
    checkpoint_path = tmp_path / "schema4.pt"
    static_identity = _save_identity_checkpoint(checkpoint_path, model=source)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    payload["state_contract"].pop("layout_checksum")
    torch.save(payload, checkpoint_path)
    target = IdentityTrainingModel(encoder, wan)
    with pytest.raises(RuntimeError, match="incompatible StateContract"):
        load_checkpoint(
            checkpoint_path,
            model=target,
            restore_rng=False,
            static_identity=static_identity,
        )


def test_schema3_smoke_checkpoint_migrates_fixed_prefix_and_identity_checksum(tmp_path: Path):
    encoder, wan = _identity_files(tmp_path)
    source = IdentityTrainingModel(encoder, wan)
    checkpoint_path = tmp_path / "schema3.pt"
    _save_identity_checkpoint(checkpoint_path, model=source)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    payload["checkpoint_schema_version"] = 3
    payload["state_contract"].pop("prefix_indexing")
    payload["state_contract"].pop("layout_checksum")
    payload["representation_identity"] = representation_identity(source)
    torch.save(payload, checkpoint_path)
    target = IdentityTrainingModel(encoder, wan)
    loaded = load_checkpoint(
        checkpoint_path,
        model=target,
        restore_rng=False,
        allow_smoke_checkpoint=True,
    )
    assert loaded["checkpoint_schema_version"] == 3


@pytest.mark.parametrize("checksum", [None, "tampered"])
def test_schema3_smoke_checkpoint_rejects_missing_or_wrong_layout_checksum(tmp_path, checksum):
    encoder, wan = _identity_files(tmp_path)
    source = IdentityTrainingModel(encoder, wan)
    checkpoint_path = tmp_path / "schema3-bad.pt"
    _save_identity_checkpoint(checkpoint_path, model=source)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    payload["checkpoint_schema_version"] = 3
    payload["state_contract"].pop("prefix_indexing")
    payload["state_contract"].pop("layout_checksum")
    payload["representation_identity"] = representation_identity(source)
    payload["representation_identity"]["layout_checksum"] = checksum
    torch.save(payload, checkpoint_path)
    target = IdentityTrainingModel(encoder, wan)
    with pytest.raises(RuntimeError, match="schema v3 layout identity mismatch"):
        load_checkpoint(
            checkpoint_path,
            model=target,
            restore_rng=False,
            allow_smoke_checkpoint=True,
        )
