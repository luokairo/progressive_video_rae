from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from progressive_videorae.evaluation.benchmarks import (
    DAVIS_NAME,
    DAVIS_PROTOCOL_ID,
    TOKENBENCH_COUNTS,
    TOKENBENCH_NAME,
    TOKENBENCH_PROTOCOL_ID,
    center_sample_indices,
    decode_frame_directory,
    manifest_rows_digest,
    parse_tokenbench_list,
    prepare_davis_rows,
    resize_to_cover_center_crop,
    validate_benchmark_rows,
)
from progressive_videorae.evaluation.checkpoint_metadata import (
    validate_stage_checkpoint_metadata,
)
from progressive_videorae.evaluation.runner import _require_full_projected_state
from progressive_videorae.model.types import ProgressiveState, StateContract
from progressive_videorae.training.checkpoint import OBJECTIVE_BY_STAGE


def _write_official_tokenbench_list(path):
    headers = {
        "bdd100k": "BDD100K",
        "bridgedata_v2": "BridgeData V2",
        "panda_70m": "Panda-70M",
        "egoexo_4d": "EgoExo-4D",
    }
    lines = []
    for group, count in TOKENBENCH_COUNTS.items():
        lines.append(f"### {headers[group]}")
        lines.extend(f"{group}/sample-{index:03d}.mp4" for index in range(count))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_tokenbench_list_requires_official_counts_and_unique_ids(tmp_path):
    path = tmp_path / "list.txt"
    _write_official_tokenbench_list(path)
    entries = parse_tokenbench_list(path)
    assert len(entries) == 500
    assert len(set(entries)) == 500
    path.write_text(path.read_text(encoding="utf-8").replace("sample-099.mp4", "sample-098.mp4", 1))
    with pytest.raises(ValueError, match="duplicate"):
        parse_tokenbench_list(path)


def _benchmark_config(name=TOKENBENCH_NAME, protocol=TOKENBENCH_PROTOCOL_ID, count=2):
    return {
        "name": name,
        "protocol_id": protocol,
        "expected_samples": count,
        "source_counts": {"group": count},
    }


def _row(path, index):
    return {
        "benchmark_schema_version": 1,
        "benchmark_name": TOKENBENCH_NAME,
        "protocol_id": TOKENBENCH_PROTOCOL_ID,
        "sample_id": f"sample-{index}",
        "official_id": f"official-{index}",
        "official_index": index,
        "source_group": "group",
        "media_type": "video",
        "path": str(path),
        "path_exists": True,
        "decode_valid": True,
        "source_sha256": f"sha-{index}",
        "official_list_sha256": "list-sha",
    }


def test_exhaustive_manifest_rejects_missing_rows_files_and_duplicates(tmp_path):
    files = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    for path in files:
        path.write_bytes(b"video")
    rows = [_row(files[0], 0), _row(files[1], 1)]
    assert [row["sample_id"] for row in validate_benchmark_rows(rows, _benchmark_config())] == [
        "sample-0",
        "sample-1",
    ]
    with pytest.raises(ValueError, match="expected 2"):
        validate_benchmark_rows(rows[:1], _benchmark_config())
    rows[1]["sample_id"] = rows[0]["sample_id"]
    with pytest.raises(ValueError, match="unique"):
        validate_benchmark_rows(rows, _benchmark_config())
    rows[1]["sample_id"] = "sample-1"
    files[1].unlink()
    with pytest.raises(FileNotFoundError):
        validate_benchmark_rows(rows, _benchmark_config())


def test_manifest_digest_is_row_order_independent(tmp_path):
    path = tmp_path / "sample.mp4"
    path.write_bytes(b"video")
    rows = [_row(path, 0), _row(path, 1)]
    assert manifest_rows_digest(rows) == manifest_rows_digest(reversed(rows))


def test_davis_center_sampling_is_exact_24_to_12_fps():
    indices = center_sample_indices(41, native_fps=24.0, target_fps=12.0, num_frames=17)
    assert len(indices) == 17
    assert all(right - left == 2 for left, right in zip(indices, indices[1:]))
    assert indices[0] + indices[-1] == 40
    with pytest.raises(ValueError, match="require at least"):
        center_sample_indices(32, native_fps=24.0, target_fps=12.0, num_frames=17)


def test_davis_manifest_uses_exact_30_sequences_and_reads_jpegs(tmp_path):
    root = tmp_path / "DAVIS"
    split = root / "ImageSets" / "2017" / "val.txt"
    split.parent.mkdir(parents=True)
    sequences = [f"sequence-{index:02d}" for index in range(30)]
    split.write_text("\n".join(sequences) + "\n", encoding="utf-8")
    for sequence in sequences:
        directory = root / "JPEGImages" / "480p" / sequence
        directory.mkdir(parents=True)
        for frame in range(33):
            Image.new("RGB", (12, 8), color=(frame, 2, 3)).save(
                directory / f"{frame:05d}.jpg"
            )
    rows, report = prepare_davis_rows(root)
    assert len(rows) == report["sample_count"] == 30
    assert rows[0]["sampled_frame_indices"] == list(range(0, 33, 2))
    video, metadata = decode_frame_directory(
        rows[0]["path"],
        native_fps=24.0,
        target_fps=12.0,
        num_frames=17,
        height=8,
        width=12,
        sampled_frame_indices=rows[0]["sampled_frame_indices"],
    )
    assert video.shape == (3, 17, 8, 12)
    assert metadata["sampled_frame_indices"].tolist() == list(range(0, 33, 2))
    (root / "JPEGImages" / "480p" / sequences[0] / "00016.jpg").unlink()
    with pytest.raises(ValueError, match="non-contiguous"):
        prepare_davis_rows(root)


@pytest.mark.parametrize("shape", [(3, 240, 1000), (3, 1000, 240)])
def test_resize_to_cover_center_crop_has_exact_geometry_without_black_bars(shape):
    video = torch.ones(2, *shape)
    output = resize_to_cover_center_crop(video, height=480, width=768)
    assert output.shape == (2, 3, 480, 768)
    assert torch.allclose(output, torch.ones_like(output))


def _checkpoint(stage="stage1a", *, complete=True):
    frames, latents = ((33, 9) if stage == "stage2b" else (17, 5))
    return {
        "checkpoint_schema_version": 4,
        "stage": stage,
        "objective_mode": OBJECTIVE_BY_STAGE[stage],
        "optimizer_step": 10,
        "stage_max_steps": 10,
        "stage_complete": complete,
        "state_contract": StateContract().to_dict(),
        "representation_identity": {"layout": "fixed"},
        "config": {
            "training": {
                "stage": stage,
                "objective_mode": OBJECTIVE_BY_STAGE[stage],
            },
            "model": {"state": {"num_frames": latents}},
            "data": {"num_frames": frames},
        },
    }


def test_stage_metadata_rejects_mismatch_and_incomplete_checkpoint():
    with pytest.raises(RuntimeError, match="expected 'stage1b'"):
        validate_stage_checkpoint_metadata(_checkpoint(), expected_stage="stage1b")
    with pytest.raises(RuntimeError, match="stage_complete"):
        validate_stage_checkpoint_metadata(_checkpoint(complete=False), expected_stage="stage1a")


def test_stage2b_training_geometry_replays_on_17_frame_evaluation_geometry():
    metadata = validate_stage_checkpoint_metadata(_checkpoint("stage2b"), expected_stage="stage2b")
    assert metadata["training_geometry"] == {"rgb_frames": 33, "temporal_latents": 9}
    assert metadata["evaluation_geometry"] == {"rgb_frames": 17, "temporal_latents": 5}


def test_projector_result_is_checked_only_after_state_is_available():
    contract = StateContract()
    state = ProgressiveState(
        tokens=torch.zeros(1, 5, 48, 30, 48),
        layout_version=contract.layout_version,
        layout_checksum=contract.layout_checksum,
        contract=contract,
    )
    assert _require_full_projected_state(SimpleNamespace(state=state)) is state


def test_davis_protocol_constants_are_fixed():
    assert DAVIS_NAME == "DAVIS17-Val-PVR-17x480x768"
    assert DAVIS_PROTOCOL_ID == "davis17_val_pvr_17x480x768_12fps_center_v1"
