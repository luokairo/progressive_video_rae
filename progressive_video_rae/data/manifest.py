from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class CsvSource:
    path: Path
    category: str
    source_tag: str
    description: str


@dataclass
class ManifestBuildResult:
    records: list[dict[str, Any]]
    report: dict[str, Any]


def parse_csv_spec(spec_path: str | Path) -> list[CsvSource]:
    path = Path(spec_path)
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    csv_entries: list[tuple[Path, str]] = []
    for index, line in enumerate(lines):
        if not line.lower().endswith(".csv"):
            continue
        description = ""
        for following in lines[index + 1 :]:
            if following:
                description = following
                break
        csv_entries.append((Path(line).expanduser(), description))
    if len(csv_entries) < 3:
        raise ValueError(f"Expected at least three CSV paths in {path}, found {len(csv_entries)}")

    sources = []
    for index, (csv_path, description) in enumerate(csv_entries):
        if index == 0:
            category, tag = "human", "human"
        elif index == 1:
            category, tag = "non_speech", "environment"
        elif index == 2:
            category, tag = "non_speech", "music"
        else:
            raise ValueError(
                "Additional CSV entries require an explicit source mapping; only human/environment/music are defined"
            )
        sources.append(CsvSource(csv_path, category, tag, description))
    return sources


def normalize_video_path(path: str) -> str:
    return os.path.abspath(os.path.normpath(os.path.expanduser(str(path).strip())))


def stable_sample_id(path: str) -> str:
    return hashlib.sha1(path.encode("utf-8"), usedforsecurity=False).hexdigest()


def assign_split(path: str, seed: int, ratios: tuple[float, float, float]) -> str:
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"split ratios must sum to one, got {ratios}")
    digest = hashlib.sha256(f"{seed}\0{path}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < ratios[0]:
        return "train"
    if value < ratios[0] + ratios[1]:
        return "val"
    return "test"


def _select_caption(current: str, candidate: Any) -> str:
    candidate = "" if candidate is None else str(candidate).strip()
    if candidate.lower() == "nan":
        candidate = ""
    return candidate if len(candidate) > len(current) else current


def merge_source_rows(source_rows: Iterable[tuple[CsvSource, Iterable[dict[str, Any]]]]) -> list[dict[str, Any]]:
    human: dict[str, dict[str, Any]] = {}
    non_speech: dict[str, dict[str, Any]] = {}
    for source, rows in source_rows:
        target = human if source.category == "human" else non_speech
        for row in rows:
            raw_path = row.get("path")
            if raw_path is None or str(raw_path).strip().lower() in {"", "nan", "none"}:
                continue
            path = normalize_video_path(str(raw_path))
            record = target.setdefault(
                path,
                {
                    "path": path,
                    "caption": "",
                    "category": source.category,
                    "source_tags": set(),
                },
            )
            record["caption"] = _select_caption(record["caption"], row.get("caption", ""))
            record["source_tags"].add(source.source_tag)

    for path in human.keys() & non_speech.keys():
        non_speech.pop(path)

    merged = []
    for record in [*human.values(), *non_speech.values()]:
        record["source_tags"] = sorted(record["source_tags"])
        record["sample_id"] = stable_sample_id(record["path"])
        merged.append(record)
    return merged


def probe_video(path: str) -> dict[str, Any]:
    result = {
        "path_exists": Path(path).is_file(),
        "native_fps": None,
        "duration": None,
        "num_frames": None,
        "decode_valid": False,
        "probe_error": None,
    }
    if not result["path_exists"]:
        result["probe_error"] = "file_not_found"
        return result
    try:
        import av

        with av.open(path, metadata_errors="ignore") as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate else None
            duration = None
            if stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                duration = float(container.duration / av.time_base)
            frames = int(stream.frames) if stream.frames else None
            result.update(
                native_fps=fps,
                duration=duration,
                num_frames=frames,
                decode_valid=bool(duration is None or duration >= 15 / 12),
            )
    except Exception as exc:  # corrupt videos must be reported, not abort the whole manifest.
        result["probe_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _fixed_balanced_records(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    humans = [record for record in records if record["category"] == "human"]
    non_speech = [record for record in records if record["category"] == "non_speech"]
    count = min(len(humans), len(non_speech))
    key = lambda record: hashlib.sha256(
        f"balanced\0{seed}\0{record['path']}".encode("utf-8")
    ).digest()
    return sorted(humans, key=key)[:count] + sorted(non_speech, key=key)[:count]


def build_manifest(
    spec_path: str | Path,
    *,
    split_seed: int = 20260807,
    split_ratios: tuple[float, float, float] = (0.95, 0.025, 0.025),
    probe: bool = True,
    probe_workers: int = 16,
) -> ManifestBuildResult:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("Install pandas and pyarrow to build manifests") from exc

    sources = parse_csv_spec(spec_path)
    source_rows = []
    raw_counts = {}
    for source in sources:
        if not source.path.is_file():
            raise FileNotFoundError(f"CSV not found: {source.path}")
        frame = pd.read_csv(source.path, usecols=lambda column: column in {"path", "caption"})
        if "path" not in frame.columns or "caption" not in frame.columns:
            raise ValueError(f"CSV must contain path and caption columns: {source.path}")
        rows = frame[["path", "caption"]].to_dict("records")
        raw_counts[source.source_tag] = len(rows)
        source_rows.append((source, rows))

    records = merge_source_rows(source_rows)
    if probe:
        with ThreadPoolExecutor(max_workers=probe_workers) as pool:
            metadata = list(pool.map(probe_video, (record["path"] for record in records)))
    else:
        metadata = [
            {
                "path_exists": Path(record["path"]).is_file(),
                "native_fps": None,
                "duration": None,
                "num_frames": None,
                "decode_valid": Path(record["path"]).is_file(),
                "probe_error": None,
            }
            for record in records
        ]
    for record, info in zip(records, metadata):
        record.update(info)
        record["split"] = assign_split(record["path"], split_seed, split_ratios)
    records.sort(key=lambda item: item["sample_id"])

    category_counts = {
        category: sum(record["category"] == category for record in records)
        for category in ("human", "non_speech")
    }
    overlap_count = sum(set(record["source_tags"]) == {"environment", "music"} for record in records)
    report = {
        "csv_spec": str(Path(spec_path).resolve()),
        "split_seed": split_seed,
        "split_ratios": split_ratios,
        "raw_counts": raw_counts,
        "category_counts": category_counts,
        "environment_music_overlap": overlap_count,
        "split_counts": {
            split: sum(record["split"] == split for record in records)
            for split in ("train", "val", "test")
        },
        "missing_files": sum(not record["path_exists"] for record in records),
        "decode_invalid": sum(not record["decode_valid"] for record in records),
    }
    return ManifestBuildResult(records=records, report=report)


def write_manifests(result: ManifestBuildResult, output_dir: str | Path, seed: int) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("Install pandas and pyarrow to write manifests") from exc
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(result.records)
    frame.to_parquet(output / "manifest.parquet", index=False)
    for split in ("train", "val", "test"):
        split_records = [record for record in result.records if record["split"] == split]
        pd.DataFrame(split_records).to_parquet(output / f"{split}.parquet", index=False)
        if split in ("val", "test"):
            balanced = _fixed_balanced_records(split_records, seed)
            pd.DataFrame(balanced).to_parquet(output / f"{split}_balanced.parquet", index=False)
    (output / "manifest_report.json").write_text(
        json.dumps(result.report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
