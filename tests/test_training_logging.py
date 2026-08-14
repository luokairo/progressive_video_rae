from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from progressive_videorae.train import (
    PREFIX_WINDOW_TERMS,
    _prefix_window_metric_names,
    record_prefix_window,
)
from progressive_videorae.training.checkpoint import representation_identity, save_checkpoint
from progressive_videorae.model.types import StateContract
from progressive_videorae.training.logging import (
    GlobalMetricWindow,
    append_jsonl_record,
    gradient_norm,
    resolve_training_log_file,
    timestamped_log_path,
)


def test_metric_window_reports_global_means_logits_and_prefix_histogram():
    window = GlobalMetricWindow(("loss", "grad"), device=torch.device("cpu"))
    window.add_mean("loss", 2.0, count=2)
    window.add_mean("loss", 4.0)
    window.add_mean("grad", 5.0)
    window.add_prefix(1, count=2)
    window.add_prefix(47, kind="paired")
    window.add_logits(torch.tensor([1.0, 3.0]), torch.tensor([-1.0, 1.0]))

    metrics, histogram = window.reduce()

    assert metrics["loss"] == pytest.approx(8.0 / 3.0)
    assert metrics["grad"] == pytest.approx(5.0)
    assert metrics["disc/real_logit_mean"] == pytest.approx(2.0)
    assert metrics["disc/real_logit_std"] == pytest.approx(1.0)
    assert metrics["disc/fake_logit_mean"] == pytest.approx(0.0)
    assert metrics["disc/fake_logit_std"] == pytest.approx(1.0)
    assert histogram == {"single": {"1": 2}, "paired": {"47": 1}}


def test_prefix_metric_window_records_hrepa_statistics_by_endpoint():
    window = GlobalMetricWindow(
        _prefix_window_metric_names(),
        device=torch.device("cpu"),
        prefix_bins=48,
    )
    loss = SimpleNamespace(
        terms={name: torch.tensor(float(index)) for index, name in enumerate(PREFIX_WINDOW_TERMS)},
        statistics={
            "prefix_repa/level": torch.tensor(2.0),
            "prefix_repa/grid_height": torch.tensor(4.0),
            "prefix_repa/grid_width": torch.tensor(6.0),
            "prefix_repa/local_scale": torch.tensor(0.4),
            "prefix_repa/anchor_global_error": torch.tensor(0.25),
            "prefix_repa/group_00/f0_local_error": torch.tensor(0.5),
        },
    )

    record_prefix_window(window, kind="single", endpoint=16, loss=loss)
    metrics, histogram = window.reduce()

    base = "prefix/single/endpoint_16"
    assert metrics[f"{base}/prefix_repa_global"] == pytest.approx(4.0)
    assert metrics[f"{base}/prefix_repa/level"] == pytest.approx(2.0)
    assert metrics[f"{base}/prefix_repa/grid_height"] == pytest.approx(4.0)
    assert metrics[f"{base}/prefix_repa/local_scale"] == pytest.approx(0.4)
    assert metrics[f"{base}/prefix_repa/group_00/f0_local_error"] == pytest.approx(0.5)
    assert histogram == {"single": {"16": 1}, "paired": {}}


def test_metric_window_uses_distributed_sum_when_initialized(monkeypatch):
    window = GlobalMetricWindow(("loss",), device=torch.device("cpu"))
    window.add_mean("loss", 3.0)
    window.add_prefix(4)
    window.add_logits(torch.tensor([2.0, 4.0]), torch.tensor([-2.0, 0.0]))

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda tensor, op: tensor.mul_(2.0),
    )

    metrics, histogram = window.reduce()

    assert metrics["loss"] == pytest.approx(3.0)
    assert metrics["disc/real_logit_mean"] == pytest.approx(3.0)
    assert metrics["disc/real_logit_std"] == pytest.approx(1.0)
    assert metrics["disc/fake_logit_mean"] == pytest.approx(-1.0)
    assert metrics["disc/fake_logit_std"] == pytest.approx(1.0)
    assert histogram == {"single": {"4": 2}, "paired": {}}


def test_gradient_norm_uses_only_present_gradients():
    parameter = nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([3.0, 4.0])
    unused = nn.Parameter(torch.zeros(1))

    value = gradient_norm((parameter, unused))

    assert value is not None
    assert value.item() == pytest.approx(5.0)
    assert gradient_norm((unused,)) is None


def test_timestamped_log_path_uses_utc_microseconds():
    path = timestamped_log_path(
        "/tmp/logs",
        "stage1b",
        now=datetime(2026, 8, 8, 12, 12, 34, 123456, tzinfo=timezone.utc),
    )

    assert path.name == "stage1b_20260808T121234123456Z.train.jsonl"

def test_resolve_log_file_uses_external_stage1_directory_for_new_and_init_runs(tmp_path):
    training = {"stage": "stage1a", "log_dir": str(tmp_path / "external_logs")}
    output_dir = tmp_path / "outputs" / "stage1a"
    now = datetime(2026, 8, 8, 12, 12, 34, 123456, tzinfo=timezone.utc)

    new_path = resolve_training_log_file(training, output_dir, now=now)
    init_path = resolve_training_log_file(training, output_dir, now=now)

    assert new_path == init_path
    assert new_path.parent == tmp_path / "external_logs"
    assert new_path.name == "stage1a_20260808T121234123456Z.train.jsonl"
    assert not output_dir.exists()


def test_resolve_log_file_reuses_checkpoint_path_only_for_resume(tmp_path):
    training = {"stage": "stage1b", "log_dir": str(tmp_path / "external_logs")}
    output_dir = tmp_path / "outputs" / "stage1b"
    restored = tmp_path / "previous" / "stage1b_existing.train.jsonl"
    now = datetime(2026, 8, 8, 12, 12, 34, 123456, tzinfo=timezone.utc)

    resumed_path = resolve_training_log_file(
        training, output_dir, resume_log_file=restored, now=now
    )
    init_path = resolve_training_log_file(training, output_dir, now=now)

    assert resumed_path == restored
    assert init_path != restored
    assert init_path.parent == tmp_path / "external_logs"


def test_only_rank_zero_appends_jsonl_record(tmp_path):
    path = tmp_path / "logs" / "run.train.jsonl"
    record = {"step": 10, "generator/total": 1.25}

    assert not append_jsonl_record(path, record, rank=1)
    assert not path.exists()
    assert append_jsonl_record(path, record, rank=0)
    assert path.read_text(encoding="utf-8") == '{"step": 10, "generator/total": 1.25}\n'



def test_representation_identity_binds_encoder_layers_and_codec():
    model = SimpleNamespace(
        encoder=SimpleNamespace(
            variant="vitl",
            output_layers=(8, 12, 16, 20, 24),
            load_report=SimpleNamespace(checkpoint_path="/weights/vjepa2.pth"),
        ),
        projector=SimpleNamespace(layout_checksum="checksum", layout_version="layout-v3"),
        decoder=SimpleNamespace(codec_id="codec-v3", decoder_id="decoder-v1"),
    )

    assert representation_identity(model) == {
        "encoder_checkpoint": "/weights/vjepa2.pth",
        "encoder_variant": "vitl",
        "selected_vjepa_layers": [8, 12, 16, 20, 24],
        "layout_checksum": "checksum",
        "layout_version": "layout-v3",
        "codec_id": "codec-v3",
        "decoder_id": "decoder-v1",
    }


def test_checkpoint_stores_log_file(tmp_path):
    model = nn.Linear(2, 2)
    model.state_contract = StateContract()
    discriminator = nn.Linear(2, 1)
    generator_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    discriminator_optimizer = torch.optim.SGD(discriminator.parameters(), lr=0.1)
    generator_scheduler = torch.optim.lr_scheduler.LambdaLR(
        generator_optimizer, lambda _step: 1.0
    )
    discriminator_scheduler = torch.optim.lr_scheduler.LambdaLR(
        discriminator_optimizer, lambda _step: 1.0
    )
    run_id = "20260808T121234123456Z"
    log_file = f"/share/project/liujingyi/logs/waverae/progressive_video_rae/stage1a_{run_id}.train.jsonl"
    checkpoint_path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        checkpoint_path,
        model=model,
        discriminator=discriminator,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator_scheduler=generator_scheduler,
        discriminator_scheduler=discriminator_scheduler,
        optimizer_step=10,
        discriminator_update_count=5,
        epoch=1,
        config={},
        log_file=log_file,
        run_id=run_id,
        checkpoint_dir=str(tmp_path),
        source_checkpoint="/external/source.pt",
        update_latest=True,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["log_file"] == log_file
    assert checkpoint["run_id"] == run_id
    assert checkpoint["checkpoint_dir"] == str(tmp_path)
    assert checkpoint["source_checkpoint"] == "/external/source.pt"
    latest = tmp_path / "latest.pt"
    assert latest.is_symlink()
    assert latest.resolve() == checkpoint_path.resolve()
    assert not tuple(tmp_path.glob(".*.tmp"))
