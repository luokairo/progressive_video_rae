from __future__ import annotations

import pytest

from progressive_videorae.evaluation.benchmarks import DAVIS_NAME, TOKENBENCH_NAME
from progressive_videorae.evaluation.compare import STAGE_ORDER, build_stage_comparison


def _run(benchmark, stage, index):
    metrics = {
        "rgb_psnr": 20.0 + index,
        "rgb_ssim": 0.7 + index * 0.01,
        "rgb_lpips": 0.3 - index * 0.01,
    }
    if benchmark == TOKENBENCH_NAME:
        metrics["reconstruction_fvd"] = 100.0 - index
    return {
        "run_dir": f"/{benchmark}/{stage}",
        "benchmark": benchmark,
        "protocol_id": f"{benchmark}-protocol",
        "manifest_sha256": f"{benchmark}-manifest",
        "manifest_rows_digest": f"{benchmark}-rows",
        "sample_id_digest": f"{benchmark}-samples",
        "stage": stage,
        "objective": f"objective-{stage}",
        "optimizer_step": index,
        "metrics": metrics,
    }


def _all_runs():
    return [
        _run(benchmark, stage, index)
        for benchmark in (TOKENBENCH_NAME, DAVIS_NAME)
        for index, stage in enumerate(STAGE_ORDER)
    ]


def test_comparison_requires_all_stages_and_computes_both_deltas():
    result = build_stage_comparison(_all_runs())
    row = next(
        row
        for row in result["rows"]
        if row["benchmark"] == TOKENBENCH_NAME and row["stage"] == "stage2a"
    )
    assert row["rgb_psnr_delta_vs_stage1a"] == pytest.approx(2.0)
    assert row["rgb_psnr_delta_vs_previous"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="missing stages"):
        build_stage_comparison(_all_runs()[:-1])


def test_comparison_rejects_duplicate_stage_and_provenance_mismatch():
    runs = _all_runs()
    with pytest.raises(ValueError, match="Duplicate"):
        build_stage_comparison([*runs, runs[0]])
    runs[-1] = {**runs[-1], "sample_id_digest": "different"}
    with pytest.raises(RuntimeError, match="sample_id_digest"):
        build_stage_comparison(runs)
