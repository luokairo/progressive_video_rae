from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..checksums import sha256_file, sha256_sidecar_path, verify_checkpoint_sha256
from ..config import load_yaml, project_root, resolve_config_path
from ..model.factory import build_model
from ..model.types import ProgressiveState
from .benchmarks import load_benchmark_records, manifest_rows_digest
from .checkpoint_metadata import load_stage_checkpoint_metadata
from .full import (
    consume_reserve_replacement,
    FVD_FEATURE_DIM,
    ExactEvaluationDataset,
    SUMMARY_METRICS,
    append_jsonl,
    atomic_write_json,
    collate_exact_evaluation,
    load_ranked_records,
    measure_cuda_phase,
    prepare_run_manifest,
    read_jsonl_repair,
    run_identity,
    sample_id_digest,
    require_exact_sample_count,
    slice_progressive_state,
    summarize_rows,
    utc_now,
    update_run_manifest,
    validate_evaluation_config,
)
from .full_metrics import (
    FullReconstructionMetricSuite,
    I3DFeatureExtractor,
    encoder_cosine,
    frechet_distance,
)
from .io import save_comparison_video
from .metrics import state_set_statistics


def _atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
    temporary.replace(path)


def _require_file_and_sidecar(path: str | Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is unavailable: {resolved}")
    sidecar = sha256_sidecar_path(resolved)
    if not sidecar.is_file():
        raise FileNotFoundError(f"{label} SHA256 sidecar is unavailable: {sidecar}")
    return resolved


def _preflight_model_paths(model_config: dict[str, Any]) -> dict[str, str]:
    encoder_checkpoint = _require_file_and_sidecar(
        model_config["encoder"]["checkpoint_path"], "Encoder checkpoint"
    )
    decoder_checkpoint = _require_file_and_sidecar(
        model_config["decoder"]["checkpoint_path"], "Wan checkpoint"
    )
    return {
        "encoder": verify_checkpoint_sha256(encoder_checkpoint, create_missing_sidecar=False),
        "decoder": verify_checkpoint_sha256(decoder_checkpoint, create_missing_sidecar=False),
    }


def _autocast_context(precision: torch.dtype):
    return torch.autocast("cuda", dtype=precision, enabled=True)


def _require_full_projected_state(projected: Any) -> ProgressiveState:
    state = projected.state
    if not isinstance(state, ProgressiveState):
        raise TypeError("Projector must produce a ProgressiveState")
    if state.full_endpoint != 47 or state.tokens.shape[2] != 48:
        raise RuntimeError("Projector did not produce the complete P_47 state")
    return state


def _cache_equivalence(
    *,
    decoder,
    state,
    reference_raw: torch.Tensor,
    split_patterns: list[list[int]],
    sequence_id: str,
    precision: torch.dtype,
    atol: float,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for pattern in split_patterns:
        cache_state = None
        videos = []
        start = 0
        for chunk_index, size in enumerate(pattern):
            chunk = slice_progressive_state(state, start, start + size)
            device_type = state.tokens.device.type
            autocast = (
                torch.autocast("cuda", dtype=precision)
                if device_type == "cuda"
                else nullcontext()
            )
            with torch.inference_mode(), autocast:
                output = decoder.decode(
                    chunk,
                    cache_mode="reset" if chunk_index == 0 else "reuse",
                    cache_state=cache_state,
                    sequence_id=sequence_id,
                )
            videos.append(output.video)
            cache_state = output.cache_state
            start += size
        cached_raw = torch.cat(videos, dim=2)
        if cached_raw.shape != reference_raw.shape:
            raise RuntimeError(
                f"Cache split {pattern} produced {tuple(cached_raw.shape)}, "
                f"expected {tuple(reference_raw.shape)}"
            )
        difference = (cached_raw.float() - reference_raw.float()).abs()
        maximum = float(difference.max().cpu())
        mean = float(difference.mean().cpu())
        if not np.isfinite(maximum) or not np.isfinite(mean):
            raise FloatingPointError(f"Cache split {pattern} produced non-finite errors")
        if maximum > atol:
            raise RuntimeError(
                f"Cache split {pattern} max_abs_error={maximum} exceeds atol={atol}"
            )
        results["+".join(str(size) for size in pattern)] = {
            "max_abs_error": maximum,
            "mean_abs_error": mean,
        }
    return results


def _cache_summary(records: list[dict[str, Any]], atol: float) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for record in records:
        for pattern, values in record.get("cache_equivalence", {}).items():
            group = grouped.setdefault(pattern, {"max_abs_error": [], "mean_abs_error": []})
            group["max_abs_error"].append(float(values["max_abs_error"]))
            group["mean_abs_error"].append(float(values["mean_abs_error"]))
    return {
        "atol": atol,
        "patterns": {
            pattern: {
                "sample_count": len(values["max_abs_error"]),
                "maximum_abs_error": max(values["max_abs_error"]),
                "mean_abs_error": float(np.mean(values["mean_abs_error"])),
            }
            for pattern, values in grouped.items()
        },
    }


def _load_progress(output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    samples = read_jsonl_repair(output_dir / "progress" / "samples.jsonl")
    failures = read_jsonl_repair(output_dir / "progress" / "failures.jsonl")
    sample_ids = [str(record["row"]["sample_id"]) for record in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("Progress contains duplicate completed sample IDs")
    failure_ids = [str(record["sample_id"]) for record in failures]
    if len(failure_ids) != len(set(failure_ids)):
        raise RuntimeError("Progress contains duplicate failed candidate IDs")
    return samples, failures


def _open_fvd_memmap(
    output_dir: Path, max_clips: int, *, resume: bool
) -> np.memmap:
    path = output_dir / "progress" / "fvd_features.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = (max_clips, 2, FVD_FEATURE_DIM)
    if resume:
        if not path.is_file():
            raise FileNotFoundError("FVD resume requires progress/fvd_features.npy")
        value = np.load(path, mmap_mode="r+")
        if value.shape != expected or value.dtype != np.float32:
            raise RuntimeError(
                f"FVD progress has {value.shape}/{value.dtype}, expected {expected}/float32"
            )
        return value
    return np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=expected
    )


def _finalize(
    *,
    output_dir: Path,
    records: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    fvd_features: np.memmap | None,
    evaluation: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    rows = [record["row"] for record in records]
    final_digest = sample_id_digest(str(row["sample_id"]) for row in rows)
    summary: dict[str, Any] = {
        "result_schema_version": 2,
        "num_clips": len(rows),
        "sample_id_digest": final_digest,
        "metrics": summarize_rows(rows, SUMMARY_METRICS),
        "decode_failures": len(failures),
    }
    if evaluation["check_cache_equivalence"]:
        summary["cache_equivalence"] = _cache_summary(
            records, float(evaluation["cache_equivalence_atol"])
        )
    if fvd_features is not None:
        fvd_features.flush()
        count = len(rows)
        value = frechet_distance(
            np.asarray(fvd_features[:count, 0]),
            np.asarray(fvd_features[:count, 1]),
        )
        summary["reconstruction_fvd"] = {
            "value": value,
            "sample_count": count,
            "feature_dim": FVD_FEATURE_DIM,
        }
    _atomic_write_csv(output_dir / "metrics.csv", rows)
    if evaluation["compute_state_statistics"]:
        state_rows = [
            row
            for record in records
            for row in record.get("state_statistics", [])
        ]
        _atomic_write_csv(output_dir / "state_statistics.csv", state_rows)
    atomic_write_json(output_dir / "metrics.json", summary)
    update_run_manifest(
        output_dir,
        manifest,
        status="completed",
        fvd_feature_sample_count=len(rows) if fvd_features is not None else None,
        completed_at=utc_now(),
        completed_samples=len(rows),
        sample_id_digest=final_digest,
        decode_failures=len(failures),
    )
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    evaluation_path = resolve_config_path(args.config)
    evaluation = load_yaml(evaluation_path)
    model_path = resolve_config_path(
        args.model_config or evaluation["model_config"],
        relative_to=evaluation_path.parent,
    )
    data_path = resolve_config_path(
        evaluation["data_config"], relative_to=evaluation_path.parent
    )
    model_config = load_yaml(model_path)
    data_config = load_yaml(data_path)
    evaluation = dict(evaluation)
    evaluation["model_config"] = str(model_path)
    evaluation["data_config"] = str(data_path)
    if evaluation.get("i3d_checkpoint") is not None:
        evaluation["i3d_checkpoint"] = str(
            Path(evaluation["i3d_checkpoint"]).expanduser().resolve()
        )
    evaluation = validate_evaluation_config(
        evaluation,
        model_config,
        data_config,
        require_runtime_device=True,
    )
    benchmark = evaluation.get("benchmark")
    benchmark_mode = benchmark is not None
    selection_mode = evaluation["purpose"] in {
        "checkpoint_selection",
        "checkpoint_selection_smoke",
    }
    expected_stage = getattr(args, "expected_stage", None)
    benchmark_manifest = getattr(args, "benchmark_manifest", None)
    selection_manifest = getattr(args, "selection_manifest", None)
    if benchmark_mode:
        if not expected_stage:
            raise ValueError("Benchmark evaluation requires --expected-stage")
        if not benchmark_manifest:
            raise ValueError("Benchmark evaluation requires --benchmark-manifest")
        if bool(args.allow_smoke_checkpoint):
            raise ValueError("Formal benchmark evaluation forbids smoke checkpoints")
        if selection_manifest:
            raise ValueError("Benchmark evaluation cannot use --selection-manifest")
    elif selection_mode:
        if not expected_stage or not selection_manifest:
            raise ValueError(
                "Checkpoint selection requires --expected-stage and --selection-manifest"
            )
        if benchmark_manifest:
            raise ValueError("Checkpoint selection cannot use --benchmark-manifest")
        if bool(args.allow_smoke_checkpoint):
            raise ValueError("Checkpoint selection forbids smoke checkpoints")
    elif expected_stage or benchmark_manifest or selection_manifest:
        raise ValueError(
            "Stage and external manifest arguments require benchmark or selection configs"
        )
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Training checkpoint is unavailable: {checkpoint_path}")
    if benchmark_mode:
        manifest_path = Path(benchmark_manifest).expanduser().resolve()
    elif selection_mode:
        manifest_path = Path(selection_manifest).expanduser().resolve()
    else:
        manifest_path = (
            Path(data_config["manifest_dir"]).expanduser().resolve()
            / f"{evaluation['split']}.parquet"
        )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Evaluation manifest is unavailable: {manifest_path}")

    checkpoint_sha = sha256_file(checkpoint_path)
    manifest_sha = sha256_file(manifest_path)
    checkpoint_metadata = None
    selection_report = None
    if benchmark_mode:
        checkpoint_metadata = load_stage_checkpoint_metadata(
            checkpoint_path,
            expected_stage=str(expected_stage),
            evaluation_frames=int(data_config["num_frames"]),
            evaluation_latents=int(model_config["state"]["num_frames"]),
        )
    elif selection_mode:
        from .selection import (
            load_selection_candidate_metadata,
            load_selection_manifest,
        )

        resolved_path = checkpoint_path.parent / "resolved_config.json"
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint resolved config: {resolved_path}")
        resolved_checkpoint_config = json.loads(resolved_path.read_text(encoding="utf-8"))
        try:
            checkpoint_step = int(checkpoint_path.stem.removeprefix("step_"))
        except ValueError as exc:
            raise ValueError("Selection checkpoint must be named step_XXXXXXXX.pt") from exc
        checkpoint_metadata = load_selection_candidate_metadata(
            checkpoint_path,
            resolved_config=resolved_checkpoint_config,
            stage=str(expected_stage),
            expected_step=checkpoint_step,
        )
    i3d_sha = (
        verify_checkpoint_sha256(
            evaluation["i3d_checkpoint"], create_missing_sidecar=False
        )
        if evaluation["compute_fvd"]
        else None
    )
    pretrained_checkpoint_sha256 = _preflight_model_paths(model_config)
    if benchmark_mode:
        records = load_benchmark_records(manifest_path, benchmark)
    elif selection_mode:
        from .selection import (
            SELECTION_PROTOCOL,
            SELECTION_SMOKE_PROTOCOL,
        )

        records, selection_report = load_selection_manifest(
            manifest_path,
            expected_count=int(evaluation["max_clips"]),
            expected_protocol=(
                SELECTION_SMOKE_PROTOCOL
                if evaluation["purpose"] == "checkpoint_selection_smoke"
                else SELECTION_PROTOCOL
            ),
        )
    else:
        records = load_ranked_records(
            manifest_path,
            num_frames=int(data_config["num_frames"]),
            target_fps=float(data_config["target_fps"]),
            sampling_seed=int(evaluation["sampling_seed"]),
            min_native_fps_ratio=float(data_config.get("min_native_fps_ratio", 0.0)),
        )
    if len(records) < evaluation["max_clips"]:
        raise RuntimeError(f"Only {len(records)} valid manifest rows are available")
    sampling = (
        {
            "mode": "exhaustive",
            "candidate_count": len(records),
            "initial_selection_count": len(records),
            "initial_sample_id_digest": sample_id_digest(
                str(row["sample_id"]) for row in records
            ),
        }
        if benchmark_mode or selection_mode
        else {
            "seed": evaluation["sampling_seed"],
            "ranked_candidate_count": len(records),
            "initial_selection_count": evaluation["max_clips"],
            "initial_sample_id_digest": sample_id_digest(
                str(row["sample_id"]) for row in records[: evaluation["max_clips"]]
            ),
        }
    )
    identity = run_identity(
        evaluation=evaluation,
        model_config=model_config,
        data_config=data_config,
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=checkpoint_sha,
        manifest_path=str(manifest_path),
        manifest_sha256=manifest_sha,
        i3d_sha256=i3d_sha,
        pretrained_checkpoint_sha256=pretrained_checkpoint_sha256,
        sampling=sampling,
        project_root=project_root(),
    )
    identity["allow_smoke_checkpoint"] = bool(args.allow_smoke_checkpoint)
    if benchmark_mode:
        official_hashes = {str(row["official_list_sha256"]) for row in records}
        identity["benchmark"] = {
            "name": benchmark["name"],
            "protocol_id": benchmark["protocol_id"],
            "official_list_sha256": next(iter(official_hashes)),
            "manifest_sha256": manifest_sha,
            "manifest_rows_digest": manifest_rows_digest(records),
            "sample_count": len(records),
            "stage": checkpoint_metadata["stage"],
            "objective": checkpoint_metadata["objective"],
            "optimizer_step": checkpoint_metadata["optimizer_step"],
            "training_geometry": checkpoint_metadata["training_geometry"],
            "evaluation_geometry": checkpoint_metadata["evaluation_geometry"],
            "sample_id_digest": sampling["initial_sample_id_digest"],
            "exhaustive": True,
        }
    if selection_mode:
        identity["checkpoint_selection"] = {
            "protocol": selection_report["protocol"],
            "selection_manifest_sha256": selection_report["selection_manifest_sha256"],
            "sample_id_digest": selection_report["sample_id_digest"],
            "sample_count": selection_report["selection_count"],
            "stage": checkpoint_metadata["stage"],
            "objective": checkpoint_metadata["objective"],
            "optimizer_step": checkpoint_metadata["optimizer_step"],
            "stage_complete": checkpoint_metadata["stage_complete"],
            "evaluation_geometry": {
                "rgb_frames": int(data_config["num_frames"]),
                "temporal_latents": int(model_config["state"]["num_frames"]),
            },
            "full_endpoint": 47,
            "exhaustive": True,
        }
    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest = prepare_run_manifest(output_dir, identity, resume=args.resume)

    try:
        device = torch.device("cuda")
        precision = torch.bfloat16
        i3d = None
        if evaluation["compute_fvd"]:
            i3d = I3DFeatureExtractor(evaluation["i3d_checkpoint"]).to(device).eval()
            with torch.inference_mode():
                contract_features = i3d(
                    torch.zeros(
                        1,
                        3,
                        int(data_config["num_frames"]),
                        224,
                        224,
                        device=device,
                    )
                )
            if tuple(contract_features.shape) != (1, FVD_FEATURE_DIM):
                raise RuntimeError("I3D contract preflight failed")

        from ..training.checkpoint import (
            load_checkpoint,
            representation_identity,
            verify_pretrained_checkpoint_hashes,
        )

        model = build_model(model_config).to(device).eval()
        checkpoint_hashes = verify_pretrained_checkpoint_hashes(
            model, create_missing_sidecars=False
        )
        static_identity = representation_identity(
            model, checkpoint_hashes=checkpoint_hashes
        )
        loaded_checkpoint = load_checkpoint(
            checkpoint_path,
            model=model,
            restore_rng=False,
            static_identity=static_identity,
            allow_smoke_checkpoint=args.allow_smoke_checkpoint,
            mmap=True,
        )
        model.decoder.enable_gradient_checkpointing(False)
        loaded_representation = loaded_checkpoint["representation_identity"]
        existing_representation = manifest.get("representation_identity")
        if (
            existing_representation is not None
            and existing_representation != loaded_representation
        ):
            raise RuntimeError("Resume representation identity does not match checkpoint")
        manifest["representation_identity"] = loaded_representation
        update_run_manifest(output_dir, manifest)

        dataset = ExactEvaluationDataset(
            records,
            num_frames=int(data_config["num_frames"]),
            target_fps=float(data_config["target_fps"]),
            height=int(data_config["height"]),
            width=int(data_config["width"]),
            benchmark_mode=benchmark_mode,
            min_native_fps_ratio=float(data_config.get("min_native_fps_ratio", 0.0)),
        )
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=int(data_config["num_workers"]),
            pin_memory=True,
            persistent_workers=int(data_config["num_workers"]) > 0,
            collate_fn=collate_exact_evaluation,
        )
        metric_suite = FullReconstructionMetricSuite(
            target_fps=float(data_config["target_fps"])
        ).to(device).eval()

        sample_records, failures = _load_progress(output_dir)
        strict_manifest_mode = benchmark_mode or selection_mode
        if strict_manifest_mode and failures:
            raise RuntimeError(
                "An exhaustive run with recorded decode failures cannot resume"
            )
        completed_ids = {
            str(record["row"]["sample_id"]) for record in sample_records
        }
        failed_ids = {str(record["sample_id"]) for record in failures}
        replaced_ids = {
            str(record["row"]["replacement_for"])
            for record in sample_records
            if record["row"].get("replacement_for")
        }
        unresolved_initial_failures = [
            str(failure["sample_id"])
            for failure in failures
            if failure["initial_selection"]
            and str(failure["sample_id"]) not in replaced_ids
        ]
        fvd_features = (
            _open_fvd_memmap(
                output_dir, evaluation["max_clips"], resume=args.resume
            )
            if evaluation["compute_fvd"]
            else None
        )
        saved_videos = sum(
            bool(record["row"].get("video_saved")) for record in sample_records
        )

        try:
            from tqdm import tqdm

            iterator = tqdm(loader, total=len(dataset), desc="full reconstruction")
        except ImportError:
            iterator = loader

        for sample in iterator:
            if len(sample_records) >= evaluation["max_clips"]:
                break
            sample_id = str(sample["sample_id"])
            if sample_id in completed_ids or sample_id in failed_ids:
                continue
            candidate_index = int(sample["candidate_index"])
            if not bool(sample["decode_ok"]):
                failure = {
                    "sample_id": sample_id,
                    "path": str(sample["path"]),
                    "candidate_index": candidate_index,
                    "initial_selection": candidate_index < evaluation["max_clips"],
                    "error": str(sample["decode_error"]),
                }
                append_jsonl(output_dir / "progress" / "failures.jsonl", failure)
                failures.append(failure)
                failed_ids.add(sample_id)
                if failure["initial_selection"]:
                    unresolved_initial_failures.append(sample_id)
                update_run_manifest(
                    output_dir,
                    manifest,
                    completed_samples=len(sample_records),
                    decode_failures=len(failures),
                )
                if strict_manifest_mode:
                    raise RuntimeError(
                        f"Frozen exhaustive sample {sample_id} failed to decode: "
                        f"{sample['decode_error']}"
                    )
                continue

            pixel_values = sample["pixel_values"].to(device, non_blocking=True)
            target_rgb = pixel_values.mul(2).sub(1).float()

            def encode_call():
                with torch.inference_mode(), _autocast_context(precision):
                    return model.encode_features(pixel_values)

            encoder_output, encoder_measurement = measure_cuda_phase(encode_call)

            def project_call():
                with torch.inference_mode(), _autocast_context(precision):
                    return model.projector(encoder_output)

            projected, projector_measurement = measure_cuda_phase(project_call)
            state = _require_full_projected_state(projected)

            def decode_call():
                with torch.inference_mode(), _autocast_context(precision):
                    return model.decoder.decode(state, cache_mode="disabled")

            decoder_output, decoder_measurement = measure_cuda_phase(decode_call)
            prediction_raw = decoder_output.video.float()
            row = {
                "sample_id": sample_id,
                "path": str(sample["path"]),
                "category": str(sample["category"]),
                "source_tags": json.dumps(sample["source_tags"], ensure_ascii=False),
                "codec_sequence_id": str(sample["codec_sequence_id"]),
                "candidate_index": candidate_index,
                "replacement_for": (
                    None
                    if strict_manifest_mode
                    else consume_reserve_replacement(
                        candidate_index,
                        evaluation["max_clips"],
                        unresolved_initial_failures,
                    )
                ),
                "native_fps": float(sample["native_fps"]),
                "encoder_seconds": encoder_measurement.seconds,
                "projector_seconds": projector_measurement.seconds,
                "decoder_seconds": decoder_measurement.seconds,
                "model_forward_seconds": (
                    encoder_measurement.seconds
                    + projector_measurement.seconds
                    + decoder_measurement.seconds
                ),
                "encoder_start_allocated_gb": encoder_measurement.start_allocated_gb,
                "encoder_peak_allocated_gb": encoder_measurement.peak_allocated_gb,
                "encoder_incremental_peak_gb": encoder_measurement.incremental_peak_gb,
                "projector_start_allocated_gb": projector_measurement.start_allocated_gb,
                "projector_peak_allocated_gb": projector_measurement.peak_allocated_gb,
                "projector_incremental_peak_gb": (
                    projector_measurement.incremental_peak_gb
                ),
                "decoder_start_allocated_gb": decoder_measurement.start_allocated_gb,
                "decoder_peak_allocated_gb": decoder_measurement.peak_allocated_gb,
                "decoder_incremental_peak_gb": decoder_measurement.incremental_peak_gb,
                **metric_suite(prediction_raw, target_rgb),
            }

            clamped = prediction_raw.clamp(-1.0, 1.0)
            if evaluation["compute_vjepa"]:
                with torch.inference_mode(), _autocast_context(precision):
                    reconstructed_features = model.encode_features(clamped.add(1).mul(0.5))
                local, global_score = encoder_cosine(
                    encoder_output, reconstructed_features
                )
                row["vjepa_local_cosine"] = float(local.cpu())
                row["vjepa_global_cosine"] = float(global_score.cpu())

            state_rows = []
            if evaluation["compute_state_statistics"]:
                state_rows = [
                    {"sample_id": sample_id, **values}
                    for values in state_set_statistics(state)
                ]

            fvd_index = None
            if i3d is not None and fvd_features is not None:
                real_feature = i3d(target_rgb.clamp(-1.0, 1.0)).cpu().numpy()[0]
                fake_feature = i3d(clamped).cpu().numpy()[0]
                fvd_index = len(sample_records)
                fvd_features[fvd_index, 0] = real_feature
                fvd_features[fvd_index, 1] = fake_feature
                fvd_features.flush()

            cache_results = {}
            if (
                evaluation["check_cache_equivalence"]
                and len(sample_records) < evaluation["cache_check_samples"]
            ):
                cache_results = _cache_equivalence(
                    decoder=model.decoder,
                    state=state,
                    reference_raw=prediction_raw,
                    split_patterns=evaluation["cache_split_patterns"],
                    sequence_id=str(sample["codec_sequence_id"]),
                    precision=precision,
                    atol=float(evaluation["cache_equivalence_atol"]),
                )

            video_saved = False
            if (
                evaluation["save_videos"]
                and saved_videos < evaluation["save_video_limit"]
            ):
                save_comparison_video(
                    output_dir / "videos" / f"{sample_id}.mp4",
                    [target_rgb[0], clamped[0]],
                    fps=float(data_config["target_fps"]),
                )
                saved_videos += 1
                video_saved = True
            row["video_saved"] = video_saved

            record = {
                "row": row,
                "state_statistics": state_rows,
                "cache_equivalence": cache_results,
                "fvd_index": fvd_index,
            }
            append_jsonl(output_dir / "progress" / "samples.jsonl", record)
            sample_records.append(record)
            completed_ids.add(sample_id)
            update_run_manifest(
                output_dir,
                manifest,
                completed_samples=len(sample_records),
                decode_failures=len(failures),
            )

        if len(sample_records) != evaluation["max_clips"]:
            raise RuntimeError(
                f"Only {len(sample_records)}/{evaluation['max_clips']} samples decoded successfully"
            )
        return _finalize(
            output_dir=output_dir,
            records=sample_records,
            failures=failures,
            fvd_features=fvd_features,
            evaluation=evaluation,
            manifest=manifest,
        )
    except Exception as exc:
        update_run_manifest(
            output_dir,
            manifest,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate complete-state Progressive VideoRAE RGB reconstruction"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--benchmark-manifest", default=None)
    parser.add_argument("--selection-manifest", default=None)
    parser.add_argument(
        "--expected-stage",
        choices=("stage1a", "stage1b", "stage2a", "stage2b"),
        default=None,
    )
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-smoke-checkpoint", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
