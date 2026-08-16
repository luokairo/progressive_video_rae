#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import tempfile
from typing import Any, Iterable

import yaml


REPO = Path(__file__).resolve().parents[1]
BASE_TRAIN_CONFIG = REPO / "configs/train/stage1a_recon_ablation.yaml"
FORMAL_TRAIN_CONFIG = REPO / "configs/train/stage1a_recon_last17_12k.yaml"
BASE_EVAL_CONFIG = REPO / "configs/eval/stage1a_ablation_128.yaml"
MODEL_CONFIGS = {
    "k7": REPO / "configs/model/full_480p_vjepa2_k7.yaml",
    "k17": REPO / "configs/model/full_480p_vjepa2_last17.yaml",
    "k23": REPO / "configs/model/full_480p_vjepa2_k23.yaml",
}
DEFAULT_SWEEP_PARENT = Path(
    "/share/project/liujingyi/ckpts/progressive_video_rae/stage1a_sweeps"
)
ACTIVATE_SCRIPT = Path("/share/project/liujingyi/activate_conda.sh")
SCHEMA_VERSION = 1


class GateError(RuntimeError):
    pass


def utc_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return value


def write_yaml(path: Path, value: dict[str, Any]) -> None:
    atomic_write_text(path, yaml.safe_dump(value, sort_keys=False))


def source_inventory() -> list[Path]:
    patterns = (
        "progressive_videorae/**/*.py",
        "configs/**/*.yaml",
        "scripts/stage1a_ablation.py",
        "scripts/run_stage1a_*",
        "tests/test_stage1a*.py",
    )
    files: set[Path] = set()
    for pattern in patterns:
        files.update(path for path in REPO.glob(pattern) if path.is_file())
    return sorted(files)


def source_digest() -> tuple[str, list[str]]:
    digest = hashlib.sha256()
    relative_paths = []
    for path in source_inventory():
        relative = path.relative_to(REPO).as_posix()
        relative_paths.append(relative)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest(), relative_paths


def command_output(command: list[str]) -> str:
    return subprocess.run(
        command, check=True, capture_output=True, text=True
    ).stdout.strip()


def hardware_identity() -> dict[str, Any]:
    query = command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus = [line.strip() for line in query.splitlines() if line.strip()]
    return {"count": len(gpus), "gpus": gpus}


def upstream_identity() -> dict[str, str]:
    return {
        "vjepa2": command_output(
            [
                "git",
                "-C",
                str(REPO / "third_party/upstream/vjepa2"),
                "rev-parse",
                "HEAD",
            ]
        ),
        "wan2.2": command_output(
            ["git", "-C", "/share/project/lgy/Wan2.2", "rev-parse", "HEAD"]
        ),
    }


def checkpoint_sidecar_identity() -> dict[str, dict[str, Any]]:
    model = load_yaml(MODEL_CONFIGS["k17"])
    paths = {
        "vjepa2": Path(model["encoder"]["checkpoint_path"]),
        "wan2.2": Path(model["decoder"]["checkpoint_path"]),
    }
    result: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        sidecars = [
            path.with_suffix(path.suffix + ".sha256"),
            path.with_name(path.name + ".sha256"),
            path.with_suffix(".sha256"),
        ]
        sidecar = next((candidate for candidate in sidecars if candidate.is_file()), None)
        result[name] = {
            "path": str(path.resolve()),
            "size": path.stat().st_size,
            "sidecar": str(sidecar.resolve()) if sidecar else None,
            "sidecar_value": sidecar.read_text(encoding="utf-8").strip() if sidecar else None,
        }
    return result




def configured_representation_identities() -> dict[str, dict[str, Any]]:
    result = {}
    for name, path in MODEL_CONFIGS.items():
        config = load_yaml(path)
        result[name] = {
            "selected_vjepa_layers": config["encoder"]["output_layers"],
            "layer_fusion": config["projector"]["layer_fusion"],
            "layer_fusion_norm": config["projector"]["layer_fusion_norm"],
            "temporal_pooling": config["projector"]["temporal_pooling"],
            "projector_hidden_dim": config["projector"]["hidden_dim"],
            "state_channels": config["state"]["channels"],
        }
    return result
def current_identity() -> dict[str, Any]:
    digest, files = source_digest()
    return {
        "source_sha256": digest,
        "source_files": files,
        "upstream": upstream_identity(),
        "hardware": hardware_identity(),
        "pretrained": checkpoint_sidecar_identity(),
        "representations": configured_representation_identities(),
    }


def run_command(command: list[str], log_path: Path, *, cwd: Path = REPO) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"$ {shlex.join(command)}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {shlex.join(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return process.wait()


def set_expression(path: str, value: Any) -> str:
    # JSON is a YAML subset and avoids PyYAML's scalar document terminator (`...`).
    return f"{path}={json.dumps(value)}"


def training_command(
    *,
    model_config: Path,
    run_root: Path,
    max_steps: int,
    micro_batch: int,
    accumulation: int,
    checkpointing: bool,
    log_every: int,
    save_at_steps: list[int],
    warmup_steps: int,
    verify_ddp: bool,
) -> list[str]:
    if max_steps > 200:
        raise GateError("Ablation scheduler refuses to construct a formal training command")
    overrides = {
        "training.max_steps": max_steps,
        "training.global_batch_size": 64 if accumulation * micro_batch * 8 == 64 else 8,
        "training.micro_batch_size": micro_batch,
        "training.gradient_accumulation_steps": accumulation,
        "training.wan_interface_step": 0,
        "training.wan_full_step": 0,
        "training.warmup_steps": warmup_steps,
        "training.min_lr_ratio": 1.0,
        "training.repa_start_step": 0,
        "training.repa_ramp_steps": 0,
        "training.repa_max_factor": 0.0,
        "training.adversarial_weight": 0.0,
        "training.disc_start": max_steps,
        "training.adversarial_ramp_steps": 0,
        "training.gradient_checkpointing_by_phase.full": checkpointing,
        "training.verify_ddp_gradient_sync": verify_ddp,
        "training.save_every": max_steps,
        "training.save_at_steps": save_at_steps,
        "training.log_every": log_every,
        "training.checkpoint_root": str((run_root / "checkpoints").resolve()),
        "training.log_dir": str((run_root / "logs").resolve()),
    }
    command = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=8",
        "-m",
        "progressive_videorae.train",
        "--config",
        str(BASE_TRAIN_CONFIG),
        "--model-config",
        str(model_config),
        "--allow-smoke-checkpoint",
        "--seed",
        "20260807",
    ]
    for path, value in overrides.items():
        command.extend(("--set", set_expression(path, value)))
    return command


def candidate_run_dir(run_root: Path) -> Path:
    stage_root = run_root / "checkpoints/stage1a"
    candidates = sorted(path for path in stage_root.glob("*") if path.is_dir())
    if len(candidates) != 1:
        raise GateError(f"Expected one run directory under {stage_root}, got {candidates}")
    return candidates[0]


def candidate_log_file(run_root: Path) -> Path:
    logs = sorted((run_root / "logs").glob("stage1a_*.train.jsonl"))
    if len(logs) != 1:
        raise GateError(f"Expected one training log under {run_root / 'logs'}, got {logs}")
    return logs[0]


def training_records(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        # The JSONL stream also contains lifecycle events such as
        # training_start(step=0); only metric-bearing optimizer updates belong
        # in the numerical gates below.
        if (
            isinstance(value, dict)
            and isinstance(value.get("step"), int)
            and "generator/total" in value
        ):
            records.append(value)
    return records


def assert_smoke_records(records: list[dict[str, Any]]) -> None:
    if [record["step"] for record in records] != [1, 2]:
        raise GateError(f"Smoke expected records at steps 1 and 2, got {records}")
    for record in records:
        if record.get("schedule/phase") != "full":
            raise GateError("Smoke did not train the full decoder from the first update")
        if record.get("objective/repa_active") != 0:
            raise GateError("Smoke unexpectedly enabled REPA")
        if record.get("objective/gan_active") != 0:
            raise GateError("Smoke unexpectedly enabled GAN")
        if record.get("discriminator/update_count") != 0:
            raise GateError("Smoke unexpectedly updated the discriminator")
        if record.get("system/ddp_gradient_sync_verified") != 1:
            raise GateError("Smoke did not verify DDP gradient synchronization")
        positive_components = ("projector", "temporal_adapter", "pre_decoder", "wan_conv1", "wan_time_conv", "wan_spatial")
        negative_components = ("encoder", "repa_projection", "shared_mask", "discriminator")
        for component in positive_components:
            if record.get(f"gradient_present/{component}") != 1:
                raise GateError(f"Smoke lacks an expected gradient for {component}")
        for component in negative_components:
            if record.get(f"gradient_present/{component}") != 0:
                raise GateError(f"Smoke has an unexpected gradient for {component}")
        for group in ("rae_fast", "wan_temporal", "wan_spatial"):
            value = float(record.get(f"grad_norm/{group}", math.nan))
            if not math.isfinite(value) or value <= 0.0:
                raise GateError(f"Smoke has invalid gradient norm for {group}: {value}")
            post = float(record.get(f"grad_norm_post/{group}", math.nan))
            if not math.isfinite(post):
                raise GateError(f"Smoke has invalid post-clip gradient for {group}")
        expected_total = (
            float(record["l1"])
            + float(record["lpips"])
            + 0.1 * float(record["temporal_l1"])
        )
        if abs(float(record["generator/total"]) - expected_total) > 5.0e-3:
            raise GateError(
                f"Smoke generator total does not match reconstruction terms: {record}"
            )


def runtime_eval_config(root: Path, *, max_clips: int) -> Path:
    config = load_yaml(BASE_EVAL_CONFIG)
    config["model_config"] = str(MODEL_CONFIGS["k17"])
    config["data_config"] = str(REPO / "configs/data/default_17f.yaml")
    config["max_clips"] = max_clips
    path = root / f"eval_{max_clips}.yaml"
    write_yaml(path, config)
    return path


def evaluation_command(
    *,
    config: Path,
    model_config: Path,
    checkpoint: Path,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "progressive_videorae.evaluate",
        "--config",
        str(config),
        "--model-config",
        str(model_config),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
        "--allow-smoke-checkpoint",
    ]


def metric_means(output_dir: Path) -> dict[str, float]:
    summary = load_json(output_dir / "metrics.json")
    if int(summary.get("num_clips", -1)) <= 0:
        raise GateError(f"Evaluation contains no clips: {output_dir}")
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        raise GateError(f"Evaluation contains no metrics: {output_dir}")
    names = ("rgb_lpips", "rgb_psnr", "rgb_ssim", "temporal_difference_l1")
    result = {}
    for name in names:
        value = metrics.get(name)
        if not isinstance(value, dict) or not math.isfinite(float(value.get("mean", math.nan))):
            raise GateError(f"Evaluation metric {name} is invalid: {output_dir}")
        result[name] = float(value["mean"])
    result["num_clips"] = int(summary["num_clips"])
    result["sample_id_digest"] = str(summary["sample_id_digest"])
    return result


def update_task(
    manifest_path: Path,
    name: str,
    *,
    status: str,
    **values: Any,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    task = manifest.setdefault("tasks", {}).setdefault(name, {})
    task.update(status=status, updated_at=datetime.now(timezone.utc).isoformat(), **values)
    atomic_write_json(manifest_path, manifest)
    return task


def tail(path: Path, limit: int = 40) -> str:
    if not path.is_file():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:])


def run_training_task(
    *,
    manifest_path: Path,
    name: str,
    model_config: Path,
    root: Path,
    max_steps: int,
    micro_batch: int,
    accumulation: int,
    checkpointing: bool,
    log_every: int,
    save_at_steps: list[int],
    warmup_steps: int,
    verify_ddp: bool = False,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    existing = manifest.get("tasks", {}).get(name, {})
    if existing.get("status") in {"succeeded", "failed"}:
        return existing
    attempt = int(existing.get("attempt", 0)) + 1
    attempt_name = name if attempt == 1 else f"{name}_attempt_{attempt:02d}"
    run_root = root / "runs" / attempt_name
    launcher_log = run_root / "launcher.log"
    command = training_command(
        model_config=model_config,
        run_root=run_root,
        max_steps=max_steps,
        micro_batch=micro_batch,
        accumulation=accumulation,
        checkpointing=checkpointing,
        log_every=log_every,
        save_at_steps=save_at_steps,
        warmup_steps=warmup_steps,
        verify_ddp=verify_ddp,
    )
    update_task(
        manifest_path,
        name,
        status="running",
        kind="training",
        attempt=attempt,
        command=command,
        model_config=str(model_config),
        max_steps=max_steps,
        micro_batch=micro_batch,
        accumulation=accumulation,
        checkpointing=checkpointing,
        launcher_log=str(launcher_log),
    )
    return_code = run_command(command, launcher_log)
    if return_code:
        return update_task(
            manifest_path,
            name,
            status="failed",
            return_code=return_code,
            error_tail=tail(launcher_log),
        )
    try:
        run_dir = candidate_run_dir(run_root)
        training_log = candidate_log_file(run_root)
        records = training_records(training_log)
        checkpoints = {
            step: run_dir / f"step_{step:08d}.pt"
            for step in save_at_steps + [max_steps]
        }
        missing = [str(path) for path in checkpoints.values() if not path.is_file()]
        if missing:
            raise GateError(f"Training task {name} is missing checkpoints: {missing}")
        task = update_task(
            manifest_path,
            name,
            status="succeeded",
            run_dir=str(run_dir),
            training_log=str(training_log),
            records=len(records),
            last_step=records[-1]["step"] if records else None,
            checkpoints={str(step): str(path) for step, path in checkpoints.items()},
        )
        return task
    except Exception as exc:
        return update_task(
            manifest_path,
            name,
            status="failed",
            return_code=0,
            error=f"{type(exc).__name__}: {exc}",
        )


def run_evaluation_task(
    *,
    manifest_path: Path,
    name: str,
    model_config: Path,
    checkpoint: Path,
    eval_config: Path,
    root: Path,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    existing = manifest.get("tasks", {}).get(name, {})
    if existing.get("status") in {"succeeded", "failed"}:
        return existing
    attempt = int(existing.get("attempt", 0)) + 1
    attempt_name = name if attempt == 1 else f"{name}_attempt_{attempt:02d}"
    output_dir = root / "evaluations" / attempt_name
    launcher_log = root / "evaluation_logs" / f"{attempt_name}.log"
    command = evaluation_command(
        config=eval_config,
        model_config=model_config,
        checkpoint=checkpoint,
        output_dir=output_dir,
    )
    update_task(
        manifest_path,
        name,
        status="running",
        kind="evaluation",
        attempt=attempt,
        command=command,
        checkpoint=str(checkpoint),
        model_config=str(model_config),
        output_dir=str(output_dir),
        launcher_log=str(launcher_log),
    )
    return_code = run_command(command, launcher_log)
    if return_code:
        return update_task(
            manifest_path,
            name,
            status="failed",
            return_code=return_code,
            error_tail=tail(launcher_log),
        )
    try:
        metrics = metric_means(output_dir)
        expected_clips = int(load_yaml(eval_config)["max_clips"])
        if metrics["num_clips"] != expected_clips:
            raise GateError(f"Expected {expected_clips} clips, got {metrics['num_clips']}")
        return update_task(
            manifest_path,
            name,
            status="succeeded",
            output_dir=str(output_dir),
            metrics=metrics,
        )
    except Exception as exc:
        return update_task(
            manifest_path,
            name,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )


def efficiency_summary(task: dict[str, Any]) -> dict[str, float]:
    records = training_records(Path(task["training_log"]))
    selected = [record for record in records if record["step"] in (10, 20, 30)]
    if len(selected) != 3:
        raise GateError(f"Efficiency task lacks step 10/20/30 records: {task}")
    median_seconds = statistics.median(float(record["system/step_seconds"]) for record in selected)
    peak = max(float(record["system/max_memory_gib"]) for record in selected)
    return {
        "median_step_seconds": median_seconds,
        "true_clips_per_second": 64.0 / median_seconds,
        "peak_memory_gib": peak,
    }


def quality_eligible(candidate: dict[str, float], baseline: dict[str, float]) -> bool:
    return (
        candidate["rgb_psnr"] >= baseline["rgb_psnr"] - 0.05
        and candidate["rgb_ssim"] >= baseline["rgb_ssim"] - 0.001
        and candidate["temporal_difference_l1"]
        <= baseline["temporal_difference_l1"] * 1.02
    )


def choose_quality_candidate(
    *,
    names: Iterable[str],
    metrics100: dict[str, dict[str, float]],
    metrics200: dict[str, dict[str, float]],
    baseline_name: str,
    prefer_small: list[str],
) -> tuple[str, dict[str, Any]]:
    baseline100 = metrics100[baseline_name]
    baseline200 = metrics200[baseline_name]
    eligible = [baseline_name]
    reasons: dict[str, str] = {baseline_name: "baseline"}
    for name in names:
        if name == baseline_name:
            continue
        improvement = (
            baseline200["rgb_lpips"] - metrics200[name]["rgb_lpips"]
        ) / baseline200["rgb_lpips"]
        if improvement < 0.01:
            reasons[name] = f"LPIPS improvement {improvement:.4%} is below 1%"
            continue
        if metrics100[name]["rgb_lpips"] >= baseline100["rgb_lpips"]:
            reasons[name] = "step-100 LPIPS does not beat the baseline"
            continue
        if not quality_eligible(metrics200[name], baseline200):
            reasons[name] = "PSNR/SSIM/temporal guard failed"
            continue
        eligible.append(name)
        reasons[name] = "eligible"
    best_lpips = min(metrics200[name]["rgb_lpips"] for name in eligible)
    close = [
        name
        for name in eligible
        if (metrics200[name]["rgb_lpips"] - best_lpips) / best_lpips < 0.005
    ]
    order = {name: index for index, name in enumerate(prefer_small)}
    winner = min(close, key=lambda name: order.get(name, len(order)))
    return winner, {"eligible": eligible, "reasons": reasons}


def model_variant(root: Path, *, k_name: str, hidden_dim: int) -> Path:
    config = load_yaml(MODEL_CONFIGS[k_name])
    config["projector"]["hidden_dim"] = hidden_dim
    path = root / "runtime_configs" / f"model_{k_name}_h{hidden_dim}.yaml"
    write_yaml(path, config)
    return path


def initialize_manifest(root: Path, identity: dict[str, Any]) -> Path:
    path = root / "sweep_manifest.json"
    if path.is_file():
        return path
    queue = [
        "eff_e0",
        "eff_e1",
        "eff_e2",
        "eff_e3_conditional",
        "k17_train",
        "k17_eval_100",
        "k17_eval_200",
        "k7_train",
        "k7_eval_100",
        "k7_eval_200",
        "k23_train",
        "k23_eval_100",
        "k23_eval_200",
        "select_k",
        "hidden_768_train",
        "hidden_768_eval_100",
        "hidden_768_eval_200",
        "hidden_1024_train",
        "hidden_1024_eval_100",
        "hidden_1024_eval_200",
        "select_hidden",
        "finalize",
    ]
    training_common = {"global_batch_size": 64, "wan_full_step": 0, "repa_max_factor": 0.0, "adversarial_weight": 0.0}
    efficiency_plan = {
        "eff_e0": (1, 8, True),
        "eff_e1": (2, 4, True),
        "eff_e2": (1, 8, False),
        "eff_e3_conditional": (4, 2, True),
    }
    task_plan: dict[str, dict[str, Any]] = {}
    for name in queue:
        if name in efficiency_plan:
            micro, accumulation, checkpointing = efficiency_plan[name]
            task_plan[name] = {**training_common, "kind": "training", "max_steps": 30, "micro_batch": micro, "accumulation": accumulation, "checkpointing": checkpointing}
        elif name.endswith("_train"):
            task_plan[name] = {**training_common, "kind": "training", "max_steps": 200, "warmup_steps": 50, "save_at_steps": [100, 200], "batch_policy": "selected_efficiency"}
        elif "_eval_" in name:
            task_plan[name] = {"kind": "evaluation", "checkpoint_step": int(name.rsplit("_", 1)[1]), "max_clips": 128, "sampling_seed": 20260807}
        else:
            task_plan[name] = {"kind": "control"}
    task_plan["eff_e3_conditional"]["condition"] = "eff_e1 succeeded and peak_memory_gib <= 58"
    atomic_write_json(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "sweep_id": root.name,
            "root": str(root.resolve()),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "identity": identity,
            "queue": queue,
            "task_plan": task_plan,
            "tasks": {name: {"status": "pending"} for name in queue},
            "formal_training_allowed": False,
        },
    )
    return path


def manifest_has_started(manifest: dict[str, Any]) -> bool:
    return any(task.get("status") != "pending" for task in manifest.get("tasks", {}).values())


def smoke(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser().resolve() if args.root else DEFAULT_SWEEP_PARENT / utc_id()
    smoke_root = root / "smoke"
    smoke_root.mkdir(parents=True, exist_ok=True)
    report_path = smoke_root / "smoke_validation_report.json"
    identity = current_identity()
    if identity["hardware"]["count"] != 8 or any(
        "A800-SXM4-80GB" not in line for line in identity["hardware"]["gpus"]
    ):
        raise GateError(f"Smoke requires exactly 8 A800 80GB GPUs: {identity['hardware']}")

    test_log = smoke_root / "static_tests.log"
    tests = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_stage1a_last17.py",
        "tests/test_stage1a_ablation.py",
    ]
    if run_command(tests, test_log):
        atomic_write_json(
            report_path,
            {"passed": False, "identity": identity, "failure": "static tests failed"},
        )
        return 1

    manifest_path = initialize_manifest(smoke_root, identity)
    task = run_training_task(
        manifest_path=manifest_path,
        name="smoke_train",
        model_config=MODEL_CONFIGS["k17"],
        root=smoke_root,
        max_steps=2,
        micro_batch=1,
        accumulation=1,
        checkpointing=True,
        log_every=1,
        save_at_steps=[],
        warmup_steps=1,
        verify_ddp=True,
    )
    if task.get("status") != "succeeded":
        atomic_write_json(
            report_path,
            {"passed": False, "identity": identity, "failure": task},
        )
        return 1
    records = training_records(Path(task["training_log"]))
    assert_smoke_records(records)
    eval_config = runtime_eval_config(smoke_root / "runtime_configs", max_clips=2)
    checkpoint = Path(task["checkpoints"]["2"])
    evaluated = run_evaluation_task(
        manifest_path=manifest_path,
        name="smoke_eval",
        model_config=MODEL_CONFIGS["k17"],
        checkpoint=checkpoint,
        eval_config=eval_config,
        root=smoke_root,
    )
    if evaluated.get("status") != "succeeded":
        atomic_write_json(
            report_path,
            {"passed": False, "identity": identity, "failure": evaluated},
        )
        return 1
    report = {
        "schema_version": SCHEMA_VERSION,
        "passed": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "identity": identity,
        "training_log": task["training_log"],
        "checkpoint": str(checkpoint),
        "evaluation": evaluated,
    }
    atomic_write_json(report_path, report)
    atomic_write_text(root / "SMOKE_VALIDATED", str(report_path) + "\n")
    print(json.dumps({"smoke_report": str(report_path), "root": str(root)}, indent=2))
    return 0


def validate_smoke_report(path: Path) -> tuple[dict[str, Any], Path]:
    report = load_json(path)
    if report.get("passed") is not True:
        raise GateError(f"Smoke report did not pass: {path}")
    current = current_identity()
    if report.get("identity") != current:
        raise GateError("Smoke identity is stale; source, hardware, upstream, or weights changed")
    root = Path(str(report["root"])).resolve()
    if not path.resolve().is_relative_to(root):
        raise GateError("Smoke report is outside its recorded sweep root")
    return report, root


def evaluate_quality_candidate(
    *,
    manifest_path: Path,
    root: Path,
    train_name: str,
    model_config: Path,
    eval_config: Path,
) -> tuple[dict[str, float], dict[str, float]]:
    task = load_json(manifest_path)["tasks"][train_name]
    if task.get("status") != "succeeded":
        raise GateError(f"Cannot evaluate failed training task {train_name}")
    values = []
    for step in (100, 200):
        name = f"{train_name.removesuffix('_train')}_eval_{step}"
        result = run_evaluation_task(
            manifest_path=manifest_path,
            name=name,
            model_config=model_config,
            checkpoint=Path(task["checkpoints"][str(step)]),
            eval_config=eval_config,
            root=root,
        )
        if result.get("status") != "succeeded":
            raise GateError(f"Required evaluation failed: {name}")
        values.append(result["metrics"])
    if values[0]["sample_id_digest"] != values[1]["sample_id_digest"]:
        raise GateError(f"Evaluation sample digest changed for {train_name}")
    return values[0], values[1]


def assert_shared_sample_digest(candidate100, candidate200, baseline100, baseline200) -> None:
    for step, candidate, baseline in (
        (100, candidate100, baseline100),
        (200, candidate200, baseline200),
    ):
        if candidate["sample_id_digest"] != baseline["sample_id_digest"]:
            raise GateError(f"Evaluation sample digest differs from baseline at step {step}")


def write_final_outputs(root: Path, manifest_path: Path, winner: dict[str, Any]) -> None:
    manifest = load_json(manifest_path)
    rows = []
    for name, task in manifest["tasks"].items():
        if task.get("kind") != "training":
            continue
        row = {
            "name": name,
            "status": task.get("status"),
            "micro_batch": task.get("micro_batch"),
            "accumulation": task.get("accumulation"),
            "checkpointing": task.get("checkpointing"),
            "last_step": task.get("last_step"),
        }
        row.update(task.get("efficiency", {}))
        rows.append(row)
        row["error"] = task.get("error") or task.get("error_tail")
        candidate = name.removesuffix("_train")
        for step in (100, 200):
            evaluation = manifest["tasks"].get(f"{candidate}_eval_{step}", {})
            if evaluation.get("status") != "succeeded":
                continue
            for metric, value in evaluation.get("metrics", {}).items():
                row[f"{metric}_step_{step}"] = value
        row["attempt"] = task.get("attempt")
    csv_path = root / "candidate_table.csv"
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    winning_model = Path(winner["model_config"])
    formal_model = root / "artifacts/formal_candidate_model.yaml"
    write_yaml(formal_model, load_yaml(winning_model))
    formal_training = load_yaml(FORMAL_TRAIN_CONFIG)
    formal_training["model_config"] = str(formal_model)
    formal_training["data_config"] = str(REPO / "configs/data/default_17f.yaml")
    formal_training["micro_batch_size"] = winner["micro_batch"]
    formal_training["gradient_accumulation_steps"] = winner["accumulation"]
    formal_training["gradient_checkpointing_by_phase"]["full"] = winner["checkpointing"]
    formal_training["max_steps"] = 12000
    formal_training["wan_full_step"] = 200
    resolved = root / "resolved_candidate.yaml"
    write_yaml(resolved, formal_training)
    formal_command = (
        "torchrun --standalone --nproc_per_node=8 -m progressive_videorae.train "
        f"--config {shlex.quote(str(resolved))}\n"
    )
    atomic_write_text(root / "formal_launch_command.txt", formal_command)
    os.chmod(root / "formal_launch_command.txt", 0o600)
    atomic_write_text(root / "FORMAL_TRAINING_NOT_STARTED", "manual approval required\n")
    summary = [
        "# Stage1-A ablation summary",
        "",
        "Formal training was NOT started.",
        "",
        f"- Efficiency winner: {winner['efficiency_name']}",
        f"- K winner: {winner['k_name']}",
        f"- Hidden winner: {winner['hidden_dim']}",
        f"- Micro/accumulation: {winner['micro_batch']}/{winner['accumulation']}",
        f"- Full checkpointing: {winner['checkpointing']}",
        f"- Model config: {formal_model}",
        "",
        "The command preview requires explicit manual approval before use.",
    ]
    summary.extend(
        [
            "",
            "## Selection evidence",
            f"- K decision: `{json.dumps(manifest['tasks'].get('select_k', {}).get('decision', {}), ensure_ascii=False)}`",
            f"- Hidden decision: `{json.dumps(manifest['tasks'].get('select_hidden', {}).get('decision', {}), ensure_ascii=False)}`",
        ]
    )
    exceptional = {name: task.get("status") for name, task in manifest["tasks"].items() if task.get("status") in {"failed", "skipped"}}
    summary.extend(["", "## Failed or skipped tasks"])
    summary.extend(f"- {name}: {status}" for name, status in exceptional.items())
    summary.append("- None" if not exceptional else "")
    atomic_write_text(root / "review_summary.md", "\n".join(summary) + "\n")
    manifest["winner"] = winner
    manifest["status"] = "completed"
    manifest["formal_training_allowed"] = False
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(manifest_path, manifest)


def sweep(args: argparse.Namespace) -> int:
    report, recorded_root = validate_smoke_report(Path(args.smoke_report).resolve())
    root = Path(args.root).resolve() if args.root else recorded_root
    if root != recorded_root:
        raise GateError(f"Sweep root {root} differs from smoke root {recorded_root}")
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "sweep.lock"
    lock = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise GateError(f"Sweep is already running: {root}") from exc
    manifest_path = initialize_manifest(root, report["identity"])
    manifest = load_json(manifest_path)
    if manifest.get("identity") != report["identity"]:
        raise GateError("Persisted sweep identity differs from the validated Smoke")
    started = manifest_has_started(manifest)
    if args.resume_sweep and not started:
        raise GateError("--resume-sweep requires an interrupted or partially completed sweep")
    if started and not args.resume_sweep and not args.dry_run:
        raise GateError("Sweep state already exists; use --resume-sweep")
    if args.dry_run:
        print(json.dumps(load_json(manifest_path)["queue"], indent=2))
        return 0

    eval_config = runtime_eval_config(root / "runtime_configs", max_clips=128)
    efficiency_specs = [
        ("eff_e0", 1, 8, True),
        ("eff_e1", 2, 4, True),
        ("eff_e2", 1, 8, False),
    ]
    efficiency: dict[str, dict[str, Any]] = {}
    for name, micro, accumulation, checkpointing in efficiency_specs:
        task = run_training_task(
            manifest_path=manifest_path,
            name=name,
            model_config=MODEL_CONFIGS["k17"],
            root=root,
            max_steps=30,
            micro_batch=micro,
            accumulation=accumulation,
            checkpointing=checkpointing,
            log_every=10,
            save_at_steps=[],
            warmup_steps=5,
        )
        if name == "eff_e0" and task.get("status") != "succeeded":
            raise GateError("Mandatory efficiency baseline E0 failed")
        if task.get("status") == "succeeded":
            summary = efficiency_summary(task)
            task = update_task(manifest_path, name, status="succeeded", efficiency=summary)
            efficiency[name] = {**task, **summary}
    e1 = efficiency.get("eff_e1")
    if e1 is not None and e1["peak_memory_gib"] <= 58.0:
        task = run_training_task(
            manifest_path=manifest_path,
            name="eff_e3_conditional",
            model_config=MODEL_CONFIGS["k17"],
            root=root,
            max_steps=30,
            micro_batch=4,
            accumulation=2,
            checkpointing=True,
            log_every=10,
            save_at_steps=[],
            warmup_steps=5,
        )
        if task.get("status") == "succeeded":
            summary = efficiency_summary(task)
            task = update_task(
                manifest_path, "eff_e3_conditional", status="succeeded", efficiency=summary
            )
            efficiency["eff_e3_conditional"] = {**task, **summary}
    else:
        update_task(
            manifest_path,
            "eff_e3_conditional",
            status="skipped",
            reason="E1 did not pass with peak memory <= 58 GiB",
        )
    eligible_efficiency = {
        name: task
        for name, task in efficiency.items()
        if task["peak_memory_gib"] <= 72.0
    }
    if "eff_e0" not in eligible_efficiency:
        raise GateError("E0 exceeded the 72 GiB safety ceiling")
    best_efficiency = max(
        eligible_efficiency, key=lambda name: eligible_efficiency[name]["true_clips_per_second"]
    )
    baseline_speed = eligible_efficiency["eff_e0"]["true_clips_per_second"]
    if eligible_efficiency[best_efficiency]["true_clips_per_second"] < baseline_speed * 1.05:
        best_efficiency = "eff_e0"
    best_spec = eligible_efficiency[best_efficiency]
    micro = int(best_spec["micro_batch"])
    accumulation = int(best_spec["accumulation"])
    checkpointing = bool(best_spec["checkpointing"])
    update_task(
        manifest_path,
        "select_efficiency",
        status="succeeded",
        winner=best_efficiency,
        eligible=list(eligible_efficiency),
    )

    metrics100: dict[str, dict[str, float]] = {}
    metrics200: dict[str, dict[str, float]] = {}
    for k_name in ("k17", "k7", "k23"):
        train_name = f"{k_name}_train"
        task = run_training_task(
            manifest_path=manifest_path,
            name=train_name,
            model_config=MODEL_CONFIGS[k_name],
            root=root,
            max_steps=200,
            micro_batch=micro,
            accumulation=accumulation,
            checkpointing=checkpointing,
            log_every=10,
            save_at_steps=[100],
            warmup_steps=50,
        )
        if k_name == "k17" and task.get("status") != "succeeded":
            raise GateError("Mandatory K17 baseline failed")
        if task.get("status") != "succeeded":
            continue
        try:
            metrics100[k_name], metrics200[k_name] = evaluate_quality_candidate(
                manifest_path=manifest_path,
                root=root,
                train_name=train_name,
                model_config=MODEL_CONFIGS[k_name],
                eval_config=eval_config,
            )
            if k_name != "k17":
                assert_shared_sample_digest(
                    metrics100[k_name], metrics200[k_name],
                    metrics100["k17"], metrics200["k17"],
                )
        except Exception as exc:
            if k_name == "k17":
                raise
            metrics100.pop(k_name, None)
            metrics200.pop(k_name, None)
            update_task(
                manifest_path,
                f"{k_name}_evaluation",
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
    k_winner, k_decision = choose_quality_candidate(
        names=metrics200,
        metrics100=metrics100,
        metrics200=metrics200,
        baseline_name="k17",
        prefer_small=["k7", "k17", "k23"],
    )
    update_task(
        manifest_path,
        "select_k",
        status="succeeded",
        winner=k_winner,
        decision=k_decision,
    )

    hidden100 = {"h512": metrics100[k_winner]}
    hidden200 = {"h512": metrics200[k_winner]}
    hidden_models = {"h512": MODEL_CONFIGS[k_winner]}
    for hidden_dim in (768, 1024):
        hidden_name = f"h{hidden_dim}"
        model_config = model_variant(root, k_name=k_winner, hidden_dim=hidden_dim)
        hidden_models[hidden_name] = model_config
        train_name = f"hidden_{hidden_dim}_train"
        task = run_training_task(
            manifest_path=manifest_path,
            name=train_name,
            model_config=model_config,
            root=root,
            max_steps=200,
            micro_batch=micro,
            accumulation=accumulation,
            checkpointing=checkpointing,
            log_every=10,
            save_at_steps=[100],
            warmup_steps=50,
        )
        if task.get("status") != "succeeded":
            continue
        try:
            hidden100[hidden_name], hidden200[hidden_name] = evaluate_quality_candidate(
                manifest_path=manifest_path,
                root=root,
                train_name=train_name,
                model_config=model_config,
                eval_config=eval_config,
            )
            assert_shared_sample_digest(
                hidden100[hidden_name], hidden200[hidden_name],
                hidden100["h512"], hidden200["h512"],
            )
        except Exception as exc:
            hidden100.pop(hidden_name, None)
            hidden200.pop(hidden_name, None)
            update_task(
                manifest_path,
                f"hidden_{hidden_dim}_evaluation",
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
    hidden_winner, hidden_decision = choose_quality_candidate(
        names=hidden200,
        metrics100=hidden100,
        metrics200=hidden200,
        baseline_name="h512",
        prefer_small=["h512", "h768", "h1024"],
    )
    update_task(
        manifest_path,
        "select_hidden",
        status="succeeded",
        winner=hidden_winner,
        decision=hidden_decision,
    )
    winner = {
        "efficiency_name": best_efficiency,
        "k_name": k_winner,
        "hidden_dim": int(hidden_winner.removeprefix("h")),
        "micro_batch": micro,
        "accumulation": accumulation,
        "checkpointing": checkpointing,
        "model_config": str(hidden_models[hidden_winner]),
    }
    write_final_outputs(root, manifest_path, winner)
    print("Sweep complete. Formal training was NOT started. Manual approval is required.")
    return 0


def launch(args: argparse.Namespace) -> int:
    report, root = validate_smoke_report(Path(args.smoke_report).resolve())
    del report
    manifest_path = root / "sweep_manifest.json"
    started = manifest_path.is_file() and manifest_has_started(load_json(manifest_path))
    if args.resume_sweep and not started:
        raise GateError("--resume-sweep requires persisted in-progress sweep state")
    if started and not args.resume_sweep and not args.dry_run:
        raise GateError("Sweep state already exists; relaunch with --resume-sweep")
    session = args.session or f"pvr_stage1a_sweep_{root.name[-18:]}"
    session = "".join(character if character.isalnum() or character in "_-" else "_" for character in session)
    sweep_log = root / "sweep_tmux.log"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "sweep",
        "--root",
        str(root),
        "--smoke-report",
        str(Path(args.smoke_report).resolve()),
    ]
    if args.resume_sweep:
        command.append("--resume-sweep")
    if args.dry_run:
        print(json.dumps({"session": session, "root": str(root), "command": command}, indent=2))
        return 0
    shell_command = (
        f"cd {shlex.quote(str(REPO))} && "
        f"{shlex.join(command)} >> {shlex.quote(str(sweep_log))} 2>&1"
    )
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session, shell_command],
        check=True,
    )
    atomic_write_text(root / "tmux_session.txt", session + "\n")
    atomic_write_text(root / "sweep_launcher_command.txt", shlex.join(command) + "\n")
    print(json.dumps({"session": session, "root": str(root), "log": str(sweep_log)}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage1-A smoke and serial ablation scheduler")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--root", default=None)
    smoke_parser.set_defaults(function=smoke)

    sweep_parser = subparsers.add_parser("sweep")
    sweep_parser.add_argument("--root", required=True)
    sweep_parser.add_argument("--smoke-report", required=True)
    sweep_parser.add_argument("--dry-run", action="store_true")
    sweep_parser.add_argument("--resume-sweep", action="store_true")
    sweep_parser.set_defaults(function=sweep)

    launch_parser = subparsers.add_parser("launch")
    launch_parser.add_argument("--smoke-report", required=True)
    launch_parser.add_argument("--session", default=None)
    launch_parser.add_argument("--dry-run", action="store_true")
    launch_parser.set_defaults(function=launch)
    launch_parser.add_argument("--resume-sweep", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        code = int(args.function(args))
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
    raise SystemExit(code)


if __name__ == "__main__":
    main()
