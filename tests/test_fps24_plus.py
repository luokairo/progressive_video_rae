from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from progressive_videorae.config import load_training_bundle, load_yaml
from progressive_videorae.data.dataset import native_fps_is_supported
from progressive_videorae.evaluation.full import validate_evaluation_config
from progressive_videorae.evaluation import full_metrics
from progressive_videorae.model.types import (
    IMAGE_FIRST_ID,
    VIDEO_GROUP_ID,
    ProgressiveState,
    ProgressiveStateChunk,
    StateContract,
)
from progressive_videorae.model.wan_decoder import (
    RAETemporalCache,
    WanCacheState,
    WanVideoDecoder,
)
from progressive_videorae.train import (
    validate_init_sampling_contract,
    validate_resume_training_contract,
)
from progressive_videorae.training.checkpoint import validate_checkpoint_transition
from progressive_videorae.training.stages import (
    stage1a_phase,
    validate_training_bundle,
)


def _state(latent_types: list[int]) -> ProgressiveState:
    contract = StateContract()
    return ProgressiveState(
        tokens=torch.zeros(1, len(latent_types), 48, 30, 48),
        layout_version=contract.layout_version,
        layout_checksum=contract.layout_checksum,
        latent_types=torch.tensor(latent_types),
        contract=contract,
    )


def test_native_fps_policy_accepts_23976_and_rejects_low_fps():
    assert native_fps_is_supported(23.976, 24.0, 0.999)
    assert native_fps_is_supported(24.0, 24.0, 0.999)
    assert not native_fps_is_supported(23.0, 24.0, 0.999)


def test_stage1a_24fps_bundle_is_k23_h512_and_preserves_formal_boundary():
    bundle = load_training_bundle(
        "configs/train/stage1a_recon_k23_h512_24fps_12k.yaml"
    )
    validate_training_bundle(bundle["training"], bundle["model"], bundle["data"])
    assert bundle["model"]["encoder"]["output_layers"] == list(range(2, 25))
    assert bundle["model"]["projector"]["hidden_dim"] == 512
    assert bundle["model"]["video"]["target_fps"] == 24.0
    assert bundle["data"]["target_fps"] == 24.0
    assert bundle["data"]["min_native_fps_ratio"] == pytest.approx(0.999)
    assert bundle["training"]["max_steps"] == 12000
    assert bundle["training"]["warmup_steps"] == 500
    assert bundle["training"]["min_lr_ratio"] == pytest.approx(0.1)
    assert bundle["training"]["temporal_l1_weight"] == pytest.approx(0.1)
    assert bundle["training"]["repa_start_step"] == 1000
    assert bundle["training"]["repa_ramp_steps"] == 1500
    assert bundle["training"]["repa_max_factor"] == pytest.approx(0.5)
    assert bundle["training"]["adversarial_weight"] == 0.0
    assert bundle["training"]["disc_start"] == 12000
    assert bundle["training"]["adversarial_ramp_steps"] == 0
    assert stage1a_phase(199, wan_interface_step=0, wan_full_step=200) == "interface"
    assert stage1a_phase(200, wan_interface_step=0, wan_full_step=200) == "full"


def test_stage2b_24fps_bundle_keeps_k23_h512_with_33_frames():
    bundle = load_training_bundle("configs/train/stage2b_24fps.yaml")
    validate_training_bundle(bundle["training"], bundle["model"], bundle["data"])
    assert bundle["model"]["video"]["num_frames"] == 33
    assert bundle["model"]["state"]["num_frames"] == 9
    assert bundle["model"]["encoder"]["output_layers"] == list(range(2, 25))
    assert bundle["model"]["projector"]["hidden_dim"] == 512
    assert bundle["data"]["target_fps"] == 24.0


def test_stage1a_plus_bundle_uses_33_rgb_frames_and_17_frame_window_model():
    bundle = load_training_bundle(
        "configs/train/stage1a_plus_k23_h512_24fps_2k.yaml"
    )
    validate_training_bundle(bundle["training"], bundle["model"], bundle["data"])
    assert bundle["training"]["stage"] == "stage1a_plus"
    assert bundle["training"]["max_steps"] == 2000
    assert bundle["training"]["global_batch_size"] == 32
    assert bundle["training"]["gradient_accumulation_steps"] == 4
    assert bundle["data"]["num_frames"] == 33
    assert bundle["model"]["video"]["num_frames"] == 17


def test_model_and_data_fps_mismatch_is_rejected():
    bundle = load_training_bundle(
        "configs/train/stage1a_recon_k23_h512_24fps_12k.yaml"
    )
    bundle["data"]["target_fps"] = 12.0
    with pytest.raises(ValueError, match="target_fps"):
        validate_training_bundle(bundle["training"], bundle["model"], bundle["data"])


def test_resume_and_init_sampling_contracts_reject_fps_switch():
    training = {"stage": "stage1a", "micro_batch_size": 1,
                "gradient_accumulation_steps": 8, "global_batch_size": 64,
                "max_steps": 12000}
    saved_data = {
        "num_frames": 17,
        "target_fps": 12.0,
        "min_native_fps_ratio": 0.0,
    }
    checkpoint = {"config": {"training": dict(training), "data": saved_data}}
    current = {
        "num_frames": 17,
        "target_fps": 24.0,
        "min_native_fps_ratio": 0.999,
    }
    with pytest.raises(RuntimeError, match="Resume sampling contract mismatch"):
        validate_resume_training_contract(checkpoint, training, current)
    with pytest.raises(RuntimeError, match="Init sampling contract mismatch"):
        validate_init_sampling_contract(checkpoint, current)


def test_progressive_state_chunk_enforces_sequence_types_and_offsets():
    first = ProgressiveStateChunk(
        state=_state([IMAGE_FIRST_ID, VIDEO_GROUP_ID]),
        sequence_id="sequence",
        latent_start=0,
        is_sequence_start=True,
        target_fps=24.0,
        start_timestamp=1.0,
    )
    assert first.latent_end == 2
    continuation = ProgressiveStateChunk(
        state=_state([VIDEO_GROUP_ID, VIDEO_GROUP_ID]),
        sequence_id="sequence",
        latent_start=2,
        is_sequence_start=False,
        target_fps=24.0,
        start_timestamp=1.0 + 8.0 / 24.0,
    )
    assert continuation.latent_end == 4
    with pytest.raises(ValueError, match="video_group"):
        ProgressiveStateChunk(
            state=_state([IMAGE_FIRST_ID]),
            sequence_id="sequence",
            latent_start=2,
            is_sequence_start=False,
            target_fps=24.0,
            start_timestamp=1.0,
        )


def test_detached_cache_state_removes_graph_and_preserves_sequence_metadata():
    value = torch.ones(1, requires_grad=True) * 2.0
    cache = WanCacheState(
        features=[[value]],
        latents_seen=5,
        rae=RAETemporalCache(raw_history=value.view(1, 1, 1, 1, 1)),
        sequence_id="sequence",
        next_latent_offset=5,
        target_fps=24.0,
    )
    detached = WanVideoDecoder.detached_cache_state(cache)
    assert detached.features[0][0].grad_fn is None
    assert detached.rae.raw_history.grad_fn is None
    assert detached.sequence_id == "sequence"
    assert detached.next_latent_offset == 5
    assert detached.target_fps == 24.0
    assert cache.features[0][0].grad_fn is not None



def test_smoke_checkpoint_cannot_enter_formal_transition():
    checkpoint = {
        "checkpoint_schema_version": 4,
        "stage": "stage1a",
        "objective_mode": "full_repa",
        "optimizer_step": 2,
        "stage_max_steps": 2,
        "stage_complete": True,
        "run_mode": "smoke",
    }
    with pytest.raises(RuntimeError, match="Smoke checkpoint"):
        validate_checkpoint_transition(
            checkpoint,
            target_stage="stage1b",
            target_objective_mode="unused-for-init",
            mode="init",
            allow_smoke_checkpoint=False,
        )
    validate_checkpoint_transition(
        checkpoint,
        target_stage="stage1b",
        target_objective_mode="unused-for-init",
        mode="init",
        allow_smoke_checkpoint=True,
    )


class _ZeroLPIPS(nn.Module):
    def forward(self, prediction, target, chunk_size=4):
        del target, chunk_size
        return prediction.sum() * 0.0


def test_24fps_metric_reports_per_second_temporal_difference(monkeypatch):
    monkeypatch.setattr(full_metrics, "EvaluationLPIPS", _ZeroLPIPS)
    suite = full_metrics.FullReconstructionMetricSuite(target_fps=24.0)
    prediction = torch.zeros(1, 3, 2, 8, 8)
    prediction[:, :, 1] = 0.25
    target = torch.zeros_like(prediction)
    metrics = suite(prediction, target)
    assert metrics["temporal_difference_l1_per_second"] == pytest.approx(
        metrics["temporal_difference_l1"] * 24.0
    )


def test_24fps_evaluation_config_is_valid_without_cuda():
    evaluation = load_yaml("configs/eval/stage1a_24fps_128.yaml")
    evaluation["max_clips"] = 2
    model = load_yaml(evaluation["model_config"])
    data = load_yaml(evaluation["data_config"])
    normalized = validate_evaluation_config(
        evaluation,
        model,
        data,
        require_runtime_device=False,
        check_paths=False,
    )
    assert normalized["max_clips"] == 2
    assert data["target_fps"] == 24.0
