from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from progressive_videorae.config import load_yaml
from progressive_videorae.training.stages import configure_stage, stage1a_phase
from scripts.stage1a_ablation import (
    GateError,
    MODEL_CONFIGS,
    choose_quality_candidate,
    initialize_manifest,
    training_command,
)


class DummyWanCore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Linear(1, 1)
        self.time_conv = nn.Linear(1, 1)
        self.spatial = nn.Linear(1, 1)


class DummyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.temporal_adapter = nn.Linear(1, 1)
        self.pre_decoder = nn.Linear(1, 1)
        self.decoder = DummyWanCore()


class DummyProjector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.main = nn.Linear(1, 1)
        self.shared_mask_set = nn.Parameter(torch.zeros(1))


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(1, 1)
        self.projector = DummyProjector()
        self.repa_projection = nn.Linear(1, 1)
        self.decoder = DummyDecoder()


def test_k_configs_change_only_the_selected_layer_range():
    expected = {
        "k7": [12, 14, 16, 18, 20, 22, 24],
        "k17": list(range(8, 25)),
        "k23": list(range(2, 25)),
    }
    configs = {name: load_yaml(path) for name, path in MODEL_CONFIGS.items()}
    for name, config in configs.items():
        assert config["encoder"]["output_layers"] == expected[name]
        assert config["projector"]["hidden_dim"] == 512
        assert config["projector"]["layer_fusion"] == "fixed_sum"
        assert config["projector"]["layer_fusion_norm"] == "non_affine_layer_norm"
        assert config["projector"]["temporal_pooling"] == "input_dim_attention"
        assert config["state"]["channels"] == 48


def test_formal_phase_boundary_has_no_full_update_in_a_200_step_loop():
    assert {stage1a_phase(step, wan_interface_step=0, wan_full_step=200) for step in range(200)} == {
        "interface"
    }
    assert stage1a_phase(200, wan_interface_step=0, wan_full_step=200) == "full"


def test_stage1a_trainable_ownership_at_interface_and_full_boundaries():
    model = DummyModel()
    configure_stage(
        model,
        "stage1a",
        optimizer_step=199,
        wan_interface_step=0,
        wan_full_step=200,
    )
    assert not any(parameter.requires_grad for parameter in model.encoder.parameters())
    assert model.projector.main.weight.requires_grad
    assert not model.projector.shared_mask_set.requires_grad
    assert model.repa_projection.weight.requires_grad
    assert model.decoder.temporal_adapter.weight.requires_grad
    assert model.decoder.pre_decoder.weight.requires_grad
    assert model.decoder.decoder.conv1.weight.requires_grad
    assert model.decoder.decoder.time_conv.weight.requires_grad
    assert not model.decoder.decoder.spatial.weight.requires_grad

    configure_stage(
        model,
        "stage1a",
        optimizer_step=200,
        wan_interface_step=0,
        wan_full_step=200,
    )
    assert all(parameter.requires_grad for parameter in model.decoder.parameters())


def test_reconstruction_only_freezes_repa_head_and_shared_mask():
    model = DummyModel()
    configure_stage(
        model,
        "stage1a",
        optimizer_step=0,
        wan_interface_step=0,
        wan_full_step=0,
        repa_trainable=False,
    )
    assert not any(parameter.requires_grad for parameter in model.repa_projection.parameters())
    assert not model.projector.shared_mask_set.requires_grad


def test_ablation_command_forces_full_from_start_and_preserves_global_batch(tmp_path: Path):
    command = training_command(
        model_config=MODEL_CONFIGS["k17"],
        run_root=tmp_path,
        max_steps=200,
        micro_batch=2,
        accumulation=4,
        checkpointing=True,
        log_every=10,
        save_at_steps=[100],
        warmup_steps=50,
        verify_ddp=False,
    )
    joined = " ".join(command)
    assert "training.max_steps=200" in joined
    assert "training.global_batch_size=64" in joined
    assert "training.micro_batch_size=2" in joined
    assert "training.gradient_accumulation_steps=4" in joined
    assert "training.wan_full_step=0" in joined
    assert "training.repa_max_factor=0.0" in joined
    assert "training.adversarial_weight=0.0" in joined
    assert "12000" not in joined


def test_ablation_scheduler_refuses_formal_training_command(tmp_path: Path):
    with pytest.raises(GateError, match="formal training"):
        training_command(
            model_config=MODEL_CONFIGS["k17"],
            run_root=tmp_path,
            max_steps=12000,
            micro_batch=1,
            accumulation=8,
            checkpointing=True,
            log_every=10,
            save_at_steps=[],
            warmup_steps=500,
            verify_ddp=False,
        )


def _metrics(lpips: float, psnr: float = 25.0, ssim: float = 0.8, temporal: float = 0.1):
    return {
        "rgb_lpips": lpips,
        "rgb_psnr": psnr,
        "rgb_ssim": ssim,
        "temporal_difference_l1": temporal,
    }


def test_quality_selection_requires_stable_one_percent_lpips_gain():
    metrics100 = {
        "k17": _metrics(0.30),
        "k7": _metrics(0.29),
        "k23": _metrics(0.28),
    }
    metrics200 = {
        "k17": _metrics(0.20),
        "k7": _metrics(0.19),
        "k23": _metrics(0.18, psnr=24.0),
    }
    winner, decision = choose_quality_candidate(
        names=metrics200,
        metrics100=metrics100,
        metrics200=metrics200,
        baseline_name="k17",
        prefer_small=["k7", "k17", "k23"],
    )
    assert winner == "k7"
    assert decision["reasons"]["k23"] == "PSNR/SSIM/temporal guard failed"


def test_manifest_never_queues_formal_training(tmp_path: Path):
    manifest_path = initialize_manifest(tmp_path, {"source_sha256": "test"})
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["formal_training_allowed"] is False
    assert not any("12000" in name or "formal" in name for name in manifest["queue"])
    training_plans = [
        task for task in manifest["task_plan"].values() if task["kind"] == "training"
    ]
    assert all(task["max_steps"] <= 200 for task in training_plans)
    assert all(task["wan_full_step"] == 0 for task in training_plans)
    assert all(task["repa_max_factor"] == 0.0 for task in training_plans)
    assert all(task["adversarial_weight"] == 0.0 for task in training_plans)
    evaluation_plans = [task for task in manifest["task_plan"].values() if task["kind"] == "evaluation"]
    assert all(task["max_clips"] == 128 for task in evaluation_plans)
