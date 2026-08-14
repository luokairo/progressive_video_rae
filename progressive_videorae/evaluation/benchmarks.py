from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..data.dataset import VideoSamplingConfig, decode_contiguous_clip


BENCHMARK_MANIFEST_SCHEMA_VERSION = 1
TOKENBENCH_NAME = "TokenBench-PVR-17x480x768"
TOKENBENCH_PROTOCOL_ID = "tokenbench_pvr_17x480x768_12fps_center_v1"
DAVIS_NAME = "DAVIS17-Val-PVR-17x480x768"
DAVIS_PROTOCOL_ID = "davis17_val_pvr_17x480x768_12fps_center_v1"
TOKENBENCH_COUNTS = {
    "bdd100k": 100,
    "bridgedata_v2": 100,
    "panda_70m": 100,
    "egoexo_4d": 200,
}
DAVIS_SEQUENCE_COUNT = 30

_TOKENBENCH_HEADER_ALIASES = {
    "bdd100k": "bdd100k",
    "bdd 100k": "bdd100k",
    "bridgedata v2": "bridgedata_v2",
    "bridge data v2": "bridgedata_v2",
    "bridgev2": "bridgedata_v2",
    "panda-70m": "panda_70m",
    "panda 70m": "panda_70m",
    "egoexo-4d": "egoexo_4d",
    "egoexo 4d": "egoexo_4d",
}


def sha256_path(path: str | Path) -> str:
    resolved = Path(path)
    digest = hashlib.sha256()
    if resolved.is_file():
        with resolved.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    files = sorted(item for item in resolved.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"Frame directory is empty: {resolved}")
    for item in files:
        digest.update(item.relative_to(resolved).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def portable_sample_id(benchmark: str, official_id: str) -> str:
    value = hashlib.sha256(f"{benchmark}\0{official_id}".encode("utf-8")).hexdigest()
    return f"{benchmark.lower()}-{value}"


def manifest_rows_digest(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda value: str(value["sample_id"])):
        canonical = {
            key: row[key]
            for key in (
                "sample_id",
                "official_id",
                "source_group",
                "media_type",
                "source_sha256",
            )
        }
        digest.update(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _normalize_tokenbench_header(line: str) -> str | None:
    value = line.lstrip("#").strip().lower().replace("_", " ")
    value = " ".join(value.split())
    for alias, canonical in _TOKENBENCH_HEADER_ALIASES.items():
        if alias in value:
            return canonical
    return None


def parse_tokenbench_list(path: str | Path) -> list[tuple[str, str]]:
    current_group: str | None = None
    entries: list[tuple[str, str]] = []
    for line_number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            group = _normalize_tokenbench_header(line)
            if group is not None:
                current_group = group
            continue
        if current_group is None:
            raise ValueError(f"TokenBench entry before a recognized source header at line {line_number}")
        entries.append((current_group, line))
    counts = {group: 0 for group in TOKENBENCH_COUNTS}
    for group, _ in entries:
        counts[group] += 1
    if counts != TOKENBENCH_COUNTS:
        raise ValueError(
            f"TokenBench official list counts are {counts}, expected {TOKENBENCH_COUNTS}"
        )
    official_ids = [f"{group}:{entry}" for group, entry in entries]
    if len(official_ids) != len(set(official_ids)):
        raise ValueError("TokenBench official list contains duplicate entries")
    return entries


def _build_file_index(root: Path) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    by_name: dict[str, list[Path]] = {}
    by_suffix: dict[str, list[Path]] = {}
    for item in root.rglob("*"):
        if not (item.is_file() or item.is_dir()):
            continue
        by_name.setdefault(item.name, []).append(item)
        relative = item.relative_to(root).as_posix()
        by_suffix.setdefault(relative, []).append(item)
    return by_name, by_suffix


def _resolve_official_entry(
    root: Path,
    official_entry: str,
    indexes: tuple[dict[str, list[Path]], dict[str, list[Path]]],
    *,
    processed_prefix: str,
) -> Path:
    direct = root / official_entry
    relative = Path(official_entry)
    derived = [
        root / f"{official_entry}.mp4",
        root / relative.with_suffix(".mp4"),
        root / Path(f"{processed_prefix}_{official_entry}").with_suffix(".mp4"),
        root / f"{processed_prefix}_{relative.stem}.mp4",
    ]
    for candidate in (*derived, direct):
        if candidate.is_file():
            return candidate.resolve()
    if direct.is_dir():
        return direct.resolve()
    by_name, by_suffix = indexes
    normalized = official_entry.lstrip("./")
    candidates = list(by_suffix.get(normalized, ()))
    if not candidates:
        names = (Path(normalized).name, f"{Path(normalized).stem}.mp4")
        candidates = [item for name in names for item in by_name.get(name, ())]
    candidates = sorted({item.resolve() for item in candidates})
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one local match for official entry {official_entry!r} under {root}, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def prepare_tokenbench_rows(
    official_list: str | Path,
    source_roots: dict[str, str | Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if set(source_roots) != set(TOKENBENCH_COUNTS):
        raise ValueError(f"TokenBench roots must contain exactly {sorted(TOKENBENCH_COUNTS)}")
    entries = parse_tokenbench_list(official_list)
    roots = {key: Path(value).expanduser().resolve() for key, value in source_roots.items()}
    for group, root in roots.items():
        if not root.is_dir():
            raise FileNotFoundError(f"TokenBench {group} root is unavailable: {root}")
    indexes = {group: _build_file_index(root) for group, root in roots.items()}
    processed_prefixes = {
        "bdd100k": "bdd_100",
        "bridgedata_v2": "bridgev2",
        "panda_70m": "panda",
        "egoexo_4d": "egoexo4D",
    }
    list_sha = sha256_path(official_list)
    rows: list[dict[str, Any]] = []
    for official_index, (group, entry) in enumerate(entries):
        source = _resolve_official_entry(
            roots[group],
            entry,
            indexes[group],
            processed_prefix=processed_prefixes[group],
        )
        if not source.is_file() or source.suffix.lower() != ".mp4":
            raise ValueError(
                f"TokenBench official preprocessing must produce one MP4 file for {entry!r}; "
                f"got {source}"
            )
        media_type = "video"
        official_id = f"{group}:{entry}"
        rows.append(
            {
                "benchmark_schema_version": BENCHMARK_MANIFEST_SCHEMA_VERSION,
                "benchmark_name": TOKENBENCH_NAME,
                "protocol_id": TOKENBENCH_PROTOCOL_ID,
                "sample_id": portable_sample_id("tokenbench", official_id),
                "official_id": official_id,
                "official_index": official_index,
                "source_group": group,
                "category": group,
                "source_tags": [group],
                "media_type": media_type,
                "path": str(source),
                "path_exists": True,
                "decode_valid": True,
                "native_fps": 12.0,
                "source_sha256": sha256_path(source),
                "official_list_sha256": list_sha,
            }
        )
    report = {
        "benchmark_schema_version": BENCHMARK_MANIFEST_SCHEMA_VERSION,
        "benchmark_name": TOKENBENCH_NAME,
        "protocol_id": TOKENBENCH_PROTOCOL_ID,
        "official_list_sha256": list_sha,
        "sample_count": len(rows),
        "source_counts": TOKENBENCH_COUNTS,
        "manifest_rows_digest": manifest_rows_digest(rows),
    }
    return rows, report


def center_sample_indices(
    total_frames: int,
    *,
    native_fps: float,
    target_fps: float = 12.0,
    num_frames: int = 17,
) -> list[int]:
    if total_frames <= 0 or native_fps <= 0 or target_fps <= 0 or num_frames <= 0:
        raise ValueError("Frame counts and frame rates must be positive")
    offsets = [int(round(index * native_fps / target_fps)) for index in range(num_frames)]
    if len(offsets) != len(set(offsets)):
        raise ValueError("Target FPS would select duplicate source frames")
    span = offsets[-1]
    if total_frames <= span:
        raise ValueError(
            f"Only {total_frames} source frames are available; {num_frames} frames at "
            f"{target_fps:g} FPS from {native_fps:g} FPS require at least {span + 1}"
        )
    start = (total_frames - 1 - span) // 2
    return [start + offset for offset in offsets]


def _image_files(directory: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    files = sorted(item for item in directory.iterdir() if item.suffix.lower() in extensions)
    if not files:
        raise ValueError(f"No image frames found in {directory}")
    return files


def _require_contiguous_davis_frames(files: Sequence[Path], sequence: str) -> None:
    try:
        identifiers = [int(path.stem) for path in files]
    except ValueError as exc:
        raise ValueError(f"DAVIS {sequence} frame names must be numeric") from exc
    expected = list(range(identifiers[0], identifiers[0] + len(identifiers)))
    if identifiers != expected:
        raise ValueError(f"DAVIS {sequence} has missing or non-contiguous frame IDs")


def prepare_davis_rows(davis_root: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(davis_root).expanduser().resolve()
    split_path = root / "ImageSets" / "2017" / "val.txt"
    if not split_path.is_file():
        raise FileNotFoundError(f"DAVIS 2017 validation split is unavailable: {split_path}")
    sequences = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(sequences) != DAVIS_SEQUENCE_COUNT or len(sequences) != len(set(sequences)):
        raise ValueError("DAVIS 2017 val.txt must contain exactly 30 unique sequences")
    split_sha = sha256_path(split_path)
    rows: list[dict[str, Any]] = []
    for official_index, sequence in enumerate(sequences):
        directory = root / "JPEGImages" / "480p" / sequence
        if not directory.is_dir():
            raise FileNotFoundError(f"DAVIS frame directory is unavailable: {directory}")
        frames = _image_files(directory)
        _require_contiguous_davis_frames(frames, sequence)
        indices = center_sample_indices(len(frames), native_fps=24.0)
        official_id = f"davis2017-val:{sequence}"
        rows.append(
            {
                "benchmark_schema_version": BENCHMARK_MANIFEST_SCHEMA_VERSION,
                "benchmark_name": DAVIS_NAME,
                "protocol_id": DAVIS_PROTOCOL_ID,
                "sample_id": portable_sample_id("davis17-val", official_id),
                "official_id": official_id,
                "official_index": official_index,
                "source_group": "davis2017_val",
                "category": "davis2017_val",
                "source_tags": ["davis2017_val", sequence],
                "media_type": "frame_directory",
                "path": str(directory.resolve()),
                "path_exists": True,
                "decode_valid": True,
                "native_fps": 24.0,
                "source_frame_count": len(frames),
                "sampled_frame_indices": indices,
                "source_sha256": sha256_path(directory),
                "official_list_sha256": split_sha,
            }
        )
    report = {
        "benchmark_schema_version": BENCHMARK_MANIFEST_SCHEMA_VERSION,
        "benchmark_name": DAVIS_NAME,
        "protocol_id": DAVIS_PROTOCOL_ID,
        "official_list_sha256": split_sha,
        "sample_count": len(rows),
        "source_counts": {"davis2017_val": len(rows)},
        "manifest_rows_digest": manifest_rows_digest(rows),
    }
    return rows, report


def validate_benchmark_rows(
    rows: Sequence[dict[str, Any]], benchmark: dict[str, Any]
) -> list[dict[str, Any]]:
    expected = int(benchmark["expected_samples"])
    if len(rows) != expected:
        raise ValueError(f"Benchmark manifest has {len(rows)} rows, expected {expected}")
    sample_ids = [str(row.get("sample_id", "")) for row in rows]
    official_ids = [str(row.get("official_id", "")) for row in rows]
    if not all(sample_ids) or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Benchmark manifest sample_id values must be non-empty and unique")
    if not all(official_ids) or len(official_ids) != len(set(official_ids)):
        raise ValueError("Benchmark manifest official_id values must be non-empty and unique")
    required = {
        "benchmark_schema_version",
        "benchmark_name",
        "protocol_id",
        "source_group",
        "media_type",
        "path",
        "path_exists",
        "decode_valid",
        "source_sha256",
        "official_list_sha256",
    }
    for row in rows:
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"Benchmark manifest row is missing fields: {missing}")
        if int(row["benchmark_schema_version"]) != BENCHMARK_MANIFEST_SCHEMA_VERSION:
            raise ValueError("Unsupported benchmark manifest schema")
        if row["benchmark_name"] != benchmark["name"]:
            raise ValueError("Benchmark name does not match evaluation protocol")
        if row["protocol_id"] != benchmark["protocol_id"]:
            raise ValueError("Benchmark protocol_id does not match evaluation protocol")
        if row["media_type"] not in ("video", "frame_directory"):
            raise ValueError(f"Unsupported benchmark media_type: {row['media_type']}")
        flags = (row["path_exists"], row["decode_valid"])
        if any(not isinstance(value, (bool, np.bool_)) or not bool(value) for value in flags):
            raise ValueError("Exhaustive benchmark rows must be marked present and valid")
        if not Path(str(row["path"])).expanduser().exists():
            raise FileNotFoundError(f"Benchmark sample is unavailable: {row['path']}")
    official_indices = [int(row.get("official_index", -1)) for row in rows]
    if sorted(official_indices) != list(range(expected)):
        raise ValueError("Benchmark official_index values must be exactly [0, sample_count)")
    expected_groups = benchmark.get("source_counts")
    if expected_groups:
        actual = {str(group): 0 for group in expected_groups}
        for row in rows:
            group = str(row["source_group"])
            if group not in actual:
                raise ValueError(f"Unexpected benchmark source group: {group}")
            actual[group] += 1
        normalized = {str(key): int(value) for key, value in expected_groups.items()}
        if actual != normalized:
            raise ValueError(f"Benchmark source counts are {actual}, expected {normalized}")
    official_hashes = {str(row["official_list_sha256"]) for row in rows}
    if len(official_hashes) != 1:
        raise ValueError("Benchmark manifest has inconsistent official list hashes")
    return sorted((dict(row) for row in rows), key=lambda row: int(row.get("official_index", 0)))


def load_benchmark_records(path: str | Path, benchmark: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("Install pandas and pyarrow to read benchmark manifests") from exc
    rows = pd.read_parquet(path).to_dict("records")
    return validate_benchmark_rows(rows, benchmark)


def resize_to_cover_center_crop(
    video: torch.Tensor, *, height: int = 480, width: int = 768
) -> torch.Tensor:
    if video.ndim != 4:
        raise ValueError("video must be [T,C,H,W]")
    source_h, source_w = int(video.shape[-2]), int(video.shape[-1])
    scale = max(height / source_h, width / source_w)
    resized_h = max(height, math.ceil(source_h * scale))
    resized_w = max(width, math.ceil(source_w * scale))
    resized = F.interpolate(
        video.float(), size=(resized_h, resized_w), mode="bilinear", align_corners=False
    )
    top = (resized_h - height) // 2
    left = (resized_w - width) // 2
    return resized[:, :, top : top + height, left : left + width].contiguous()


def decode_frame_directory(
    path: str | Path,
    *,
    native_fps: float,
    target_fps: float,
    num_frames: int,
    height: int,
    width: int,
    sampled_frame_indices: Sequence[int] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    directory = Path(path)
    frames = _image_files(directory)
    indices = (
        [int(value) for value in sampled_frame_indices]
        if sampled_frame_indices is not None
        else center_sample_indices(
            len(frames), native_fps=native_fps, target_fps=target_fps, num_frames=num_frames
        )
    )
    if len(indices) != num_frames or len(indices) != len(set(indices)):
        raise ValueError("Frame-directory sampling must produce unique requested frames")
    if min(indices) < 0 or max(indices) >= len(frames):
        raise IndexError("Frame-directory sample index is out of range")
    tensors = []
    for index in indices:
        with Image.open(frames[index]) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        tensors.append(torch.from_numpy(array).permute(2, 0, 1))
    video = torch.stack(tensors).float().div(255.0)
    video = resize_to_cover_center_crop(video, height=height, width=width)
    timestamps = torch.tensor(indices, dtype=torch.float64).div(float(native_fps))
    return video.permute(1, 0, 2, 3).contiguous(), {
        "native_fps": float(native_fps),
        "sampled_frame_indices": torch.tensor(indices, dtype=torch.int64),
        "sampled_timestamps": timestamps,
    }


def decode_benchmark_record(
    row: dict[str, Any],
    *,
    num_frames: int,
    target_fps: float,
    height: int,
    width: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if row["media_type"] == "frame_directory":
        stored = row.get("sampled_frame_indices")
        if hasattr(stored, "tolist"):
            stored = stored.tolist()
        if isinstance(stored, float) and np.isnan(stored):
            stored = None
        return decode_frame_directory(
            row["path"],
            native_fps=float(row["native_fps"]),
            target_fps=target_fps,
            num_frames=num_frames,
            height=height,
            width=width,
            sampled_frame_indices=stored,
        )
    config = VideoSamplingConfig(
        num_frames=num_frames,
        target_fps=target_fps,
        height=height,
        width=width,
        split="test",
        horizontal_flip=False,
    )
    return decode_contiguous_clip(str(row["path"]), config)
