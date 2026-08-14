from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

from .benchmarks import DAVIS_NAME, TOKENBENCH_NAME
from .full import RESULT_SCHEMA_VERSION, atomic_write_json, utc_now


STAGE_ORDER = ("stage1a", "stage1b", "stage2a", "stage2b")
SUMMARY_FIELDS = {
    "rgb_psnr": "rgb_psnr",
    "rgb_ssim": "rgb_ssim",
    "rgb_lpips": "rgb_lpips",
}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def load_completed_run(run_dir: str | Path) -> dict[str, Any]:
    directory = Path(run_dir).expanduser().resolve()
    manifest = _load_json(directory / "run_manifest.json")
    metrics = _load_json(directory / "metrics.json")
    if manifest.get("status") != "completed":
        raise RuntimeError(f"Comparison requires a completed run: {directory}")
    identity = manifest.get("run_identity")
    if not isinstance(identity, dict) or identity.get("result_schema_version") != RESULT_SCHEMA_VERSION:
        raise RuntimeError(f"Run does not use result schema v2: {directory}")
    benchmark = identity.get("benchmark")
    if not isinstance(benchmark, dict):
        raise RuntimeError(f"Run is not a benchmark evaluation: {directory}")
    if metrics.get("sample_id_digest") != manifest.get("sample_id_digest"):
        raise RuntimeError(f"Final sample digest mismatch inside run: {directory}")
    values: dict[str, float] = {}
    summaries = metrics.get("metrics")
    if not isinstance(summaries, dict):
        raise RuntimeError(f"Run has no metric summaries: {directory}")
    for output_name, metric_name in SUMMARY_FIELDS.items():
        summary = summaries.get(metric_name)
        if not isinstance(summary, dict) or "mean" not in summary:
            raise RuntimeError(f"Run is missing {metric_name}: {directory}")
        value = float(summary["mean"])
        if not math.isfinite(value):
            raise FloatingPointError(f"Run has non-finite {metric_name}: {directory}")
        values[output_name] = value
    fvd = metrics.get("reconstruction_fvd")
    if fvd is not None:
        if not isinstance(fvd, dict) or "value" not in fvd:
            raise RuntimeError(f"Malformed reconstruction_fvd: {directory}")
        value = float(fvd["value"])
        if not math.isfinite(value):
            raise FloatingPointError(f"Run has non-finite reconstruction_fvd: {directory}")
        values["reconstruction_fvd"] = value
    expected_count = 500 if benchmark["name"] == TOKENBENCH_NAME else 30
    if int(benchmark.get("sample_count", -1)) != expected_count:
        raise RuntimeError(f"Run has the wrong benchmark sample count: {directory}")
    if int(metrics.get("num_clips", -1)) != expected_count:
        raise RuntimeError(f"Run metrics have the wrong clip count: {directory}")
    if benchmark["name"] == TOKENBENCH_NAME:
        if not isinstance(fvd, dict) or int(fvd.get("sample_count", -1)) != 500:
            raise RuntimeError(f"TokenBench rFVD must use exactly 500 samples: {directory}")
    elif fvd is not None:
        raise RuntimeError(f"DAVIS formal protocol must not report rFVD: {directory}")
    return {
        "run_dir": str(directory),
        "benchmark": str(benchmark["name"]),
        "protocol_id": str(benchmark["protocol_id"]),
        "manifest_sha256": str(benchmark["manifest_sha256"]),
        "manifest_rows_digest": str(benchmark["manifest_rows_digest"]),
        "sample_id_digest": str(manifest["sample_id_digest"]),
        "stage": str(benchmark["stage"]),
        "objective": str(benchmark["objective"]),
        "optimizer_step": int(benchmark["optimizer_step"]),
        "metrics": values,
    }


def build_stage_comparison(runs: Iterable[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, dict[str, Any]]] = {}
    for run in runs:
        benchmark = run["benchmark"]
        stage = run["stage"]
        if stage not in STAGE_ORDER:
            raise ValueError(f"Unsupported comparison stage: {stage}")
        stages = groups.setdefault(benchmark, {})
        if stage in stages:
            raise ValueError(f"Duplicate {benchmark}/{stage} run")
        stages[stage] = run
    if set(groups) != {TOKENBENCH_NAME, DAVIS_NAME}:
        raise ValueError(
            "Formal comparison requires exactly TokenBench-PVR and DAVIS17-Val-PVR"
        )
    rows: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {}
    for benchmark in (TOKENBENCH_NAME, DAVIS_NAME):
        stages = groups[benchmark]
        missing = [stage for stage in STAGE_ORDER if stage not in stages]
        if missing:
            raise ValueError(f"{benchmark} is missing stages: {missing}")
        reference = stages[STAGE_ORDER[0]]
        for key in (
            "protocol_id",
            "manifest_sha256",
            "manifest_rows_digest",
            "sample_id_digest",
        ):
            values = {stages[stage][key] for stage in STAGE_ORDER}
            if len(values) != 1:
                raise RuntimeError(f"{benchmark} has inconsistent {key}")
        provenance[benchmark] = {
            key: reference[key]
            for key in (
                "protocol_id",
                "manifest_sha256",
                "manifest_rows_digest",
                "sample_id_digest",
            )
        }
        for index, stage in enumerate(STAGE_ORDER):
            run = stages[stage]
            row: dict[str, Any] = {
                "benchmark": benchmark,
                "stage": stage,
                "objective": run["objective"],
                "optimizer_step": run["optimizer_step"],
                "run_dir": run["run_dir"],
            }
            previous = stages[STAGE_ORDER[index - 1]] if index else None
            for metric, value in run["metrics"].items():
                row[metric] = value
                row[f"{metric}_delta_vs_stage1a"] = value - reference["metrics"][metric]
                row[f"{metric}_delta_vs_previous"] = (
                    0.0 if previous is None else value - previous["metrics"][metric]
                )
            if benchmark == TOKENBENCH_NAME and "reconstruction_fvd" not in row:
                raise RuntimeError("TokenBench comparison requires reconstruction_fvd")
            rows.append(row)
    return {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "comparison_protocol": "pvr_four_stage_full_state_v1",
        "generated_at": utc_now(),
        "interpretation": (
            "Stage trajectory only; without a full-only control training run, deltas must not "
            "be attributed solely to coarse-to-fine supervision."
        ),
        "provenance": provenance,
        "rows": rows,
    }


def _atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def compare_run_directories(run_dirs: list[str], output_dir: str | Path) -> dict[str, Any]:
    if len(run_dirs) != 8:
        raise ValueError("Formal stage comparison requires exactly eight --run-dir values")
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Comparison output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    comparison = build_stage_comparison(load_completed_run(path) for path in run_dirs)
    atomic_write_json(output / "stage_comparison.json", comparison)
    _atomic_write_csv(output / "stage_comparison.csv", comparison["rows"])
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare completed PVR benchmark stages on CPU")
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = compare_run_directories(args.run_dir, args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
