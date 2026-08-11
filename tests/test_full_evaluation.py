from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from progressive_videorae.evaluation.full import (
    append_jsonl,
    prepare_run_manifest,
    read_jsonl_repair,
    stable_sample_key,
    summary_statistics,
    validate_evaluation_config,
)
from progressive_videorae.evaluation.full_metrics import (
    I3DFeatureExtractor,
    frame_mean_psnr,
    output_range_statistics,
    stylegan_v_i3d_preprocess,
)
from progressive_videorae.evaluation.runner import _cache_equivalence
from progressive_videorae.model.types import ProgressiveState, StateContract


def _evaluation_config():
    return {
        "model_config": "model.yaml",
        "data_config": "data.yaml",
        "split": "test",
        "device": "cuda",
        "batch_size": 1,
        "precision": "bf16",
        "max_clips": 2048,
        "sampling_seed": 20260807,
        "save_videos": True,
        "save_video_limit": 8,
        "compute_fvd": True,
        "i3d_checkpoint": "/missing/i3d.pt",
        "compute_state_statistics": False,
        "check_cache_equivalence": True,
        "cache_check_samples": 8,
        "cache_split_patterns": [[1, 4], [3, 2]],
        "cache_equivalence_atol": 1e-5,
    }


def _model_config():
    return {"state": {"num_frames": 5}}


def _data_config():
    return {
        "num_frames": 17,
        "target_fps": 12.0,
        "height": 480,
        "width": 768,
    }


def test_stable_sample_order_does_not_depend_on_manifest_row_order():
    sample_ids = ["c", "a", "d", "b"]
    forward = sorted(sample_ids, key=lambda value: stable_sample_key(value, 17))
    backward = sorted(reversed(sample_ids), key=lambda value: stable_sample_key(value, 17))
    assert forward == backward


def test_full_config_rejects_prefix_fields_and_invalid_precision():
    config = _evaluation_config()
    config["endpoints"] = [47]
    with pytest.raises(ValueError, match="Unknown full-evaluation"):
        validate_evaluation_config(
            config,
            _model_config(),
            _data_config(),
            require_runtime_device=False,
            check_paths=False,
        )

    config = _evaluation_config()
    config["precision"] = "bfloat"
    with pytest.raises(ValueError, match="precision"):
        validate_evaluation_config(
            config,
            _model_config(),
            _data_config(),
            require_runtime_device=False,
            check_paths=False,
        )


def test_full_config_normalizes_cache_protocol_without_touching_cuda():
    config = validate_evaluation_config(
        _evaluation_config(),
        _model_config(),
        _data_config(),
        require_runtime_device=False,
        check_paths=False,
    )
    assert config["cache_split_patterns"] == [[1, 4], [3, 2]]
    assert config["cache_equivalence_atol"] == pytest.approx(1e-5)


def test_frame_mean_psnr_averages_db_values_per_frame():
    prediction = torch.zeros(1, 1, 2, 1, 1)
    target = torch.tensor([[[[[0.5]], [[1.0]]]]])
    expected = (10 * np.log10(4 / 0.25) + 10 * np.log10(4 / 1.0)) / 2
    assert float(frame_mean_psnr(prediction, target)) == pytest.approx(expected)


def test_output_range_diagnostics_preserve_raw_overshoot():
    raw = torch.tensor([-1.2, 0.0, 1.5])
    values = output_range_statistics(raw)
    assert values["out_of_range_fraction"] == pytest.approx(2 / 3)
    assert values["mean_overshoot"] == pytest.approx((0.2 + 0.5) / 3)
    assert values["max_overshoot"] == pytest.approx(0.5)


def test_stylegan_v_preprocess_preserves_time_and_center_crops_wide_video():
    video = torch.linspace(-1, 1, 3 * 2 * 480 * 768).reshape(1, 3, 2, 480, 768)
    prepared = stylegan_v_i3d_preprocess(video)
    assert prepared.shape == (1, 3, 2, 224, 224)
    assert torch.isfinite(prepared).all()
    assert prepared.min() >= -1
    assert prepared.max() <= 1


class FakeI3D(nn.Module):
    def __init__(self, feature_dim=400):
        super().__init__()
        self.feature_dim = feature_dim
        self.last_call = None

    def forward(self, video, rescale=False, resize=False, return_features=False):
        self.last_call = {
            "shape": tuple(video.shape),
            "rescale": rescale,
            "resize": resize,
            "return_features": return_features,
        }
        return torch.zeros(video.shape[0], self.feature_dim)


def _fake_extractor(feature_dim=400):
    extractor = I3DFeatureExtractor.__new__(I3DFeatureExtractor)
    nn.Module.__init__(extractor)
    extractor.model = FakeI3D(feature_dim)
    return extractor


def test_i3d_extractor_requests_stylegan_v_features():
    extractor = _fake_extractor()
    features = extractor(torch.zeros(2, 3, 17, 32, 48))
    assert features.shape == (2, 400)
    assert extractor.model.last_call == {
        "shape": (2, 3, 17, 224, 224),
        "rescale": False,
        "resize": False,
        "return_features": True,
    }


def test_i3d_extractor_rejects_non_400_dimensional_output():
    extractor = _fake_extractor(feature_dim=399)
    with pytest.raises(ValueError, match="I3D features"):
        extractor(torch.zeros(1, 3, 17, 32, 48))


def test_summary_statistics_reports_distribution_and_rejects_nonfinite():
    values = summary_statistics([1.0, 2.0, 3.0])
    assert values["count"] == 3
    assert values["mean"] == pytest.approx(2.0)
    assert values["median"] == pytest.approx(2.0)
    assert values["min"] == pytest.approx(1.0)
    assert values["max"] == pytest.approx(3.0)
    with pytest.raises(FloatingPointError):
        summary_statistics([1.0, float("nan")])


def test_jsonl_resume_repairs_only_an_incomplete_tail(tmp_path):
    path = tmp_path / "samples.jsonl"
    append_jsonl(path, {"sample_id": "a"})
    with path.open("ab") as handle:
        handle.write(b'{"sample_id":"partial"')
    assert read_jsonl_repair(path) == [{"sample_id": "a"}]
    assert path.read_bytes().endswith(b"\n")


def test_run_manifest_resume_requires_exact_identity(tmp_path):
    output = tmp_path / "run"
    identity = {"result_schema_version": 2, "checkpoint_sha256": "a" * 64}
    manifest = prepare_run_manifest(output, identity, resume=False)
    assert manifest["status"] == "running"
    resumed = prepare_run_manifest(output, identity, resume=True)
    assert resumed["run_identity"] == identity
    with pytest.raises(RuntimeError, match="provenance"):
        prepare_run_manifest(
            output,
            {"result_schema_version": 2, "checkpoint_sha256": "b" * 64},
            resume=True,
        )


class FakeChunkDecoder:
    def decode(
        self,
        state,
        *,
        cache_mode,
        cache_state=None,
        sequence_id=None,
    ):
        del cache_mode, sequence_id
        video = state.tokens.mean(dim=(2, 3, 4), keepdim=False)
        video = video[:, None].expand(-1, 3, -1)[:, :, :, None, None]
        return SimpleNamespace(video=video, cache_state=object())


def test_cache_equivalence_checks_both_cross_call_splits_on_cpu():
    contract = StateContract()
    latent_types = torch.tensor([0, 1, 1, 1, 1])
    state = ProgressiveState(
        tokens=torch.randn(1, 5, 48, 30, 48),
        layout_version=contract.layout_version,
        layout_checksum=contract.layout_checksum,
        latent_types=latent_types,
        contract=contract,
    )
    decoder = FakeChunkDecoder()
    reference = decoder.decode(
        state, cache_mode="disabled", sequence_id="sample"
    ).video
    values = _cache_equivalence(
        decoder=decoder,
        state=state,
        reference_raw=reference,
        split_patterns=[[1, 4], [3, 2]],
        sequence_id="sample",
        precision=torch.float16,
        atol=1e-5,
    )
    assert set(values) == {"1+4", "3+2"}
    assert values["1+4"]["max_abs_error"] == 0
    assert values["3+2"]["mean_abs_error"] == 0
