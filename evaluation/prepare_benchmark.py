from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from progressive_videorae.evaluation.benchmarks import (
    prepare_davis_rows,
    prepare_tokenbench_rows,
)


def _write_outputs(output: str | Path, rows: list[dict[str, Any]], report: dict[str, Any]) -> None:
    path = Path(output).expanduser().resolve()
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError(f"Refusing to overwrite benchmark manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    pd.DataFrame(rows).to_parquet(temporary, index=False)
    os.replace(temporary, path)
    sidecar = path.with_suffix(".json")
    temporary_json = sidecar.with_name(f".{sidecar.name}.{os.getpid()}.tmp")
    with temporary_json.open("w", encoding="utf-8") as handle:
        json.dump({**report, "manifest_path": str(path)}, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_json, sidecar)


def _tokenbench(args: argparse.Namespace) -> dict[str, Any]:
    rows, report = prepare_tokenbench_rows(
        args.official_list,
        {
            "bdd100k": args.bdd100k_root,
            "bridgedata_v2": args.bridgedata_v2_root,
            "panda_70m": args.panda_70m_root,
            "egoexo_4d": args.egoexo_4d_root,
        },
    )
    _write_outputs(args.output, rows, report)
    return report


def _davis(args: argparse.Namespace) -> dict[str, Any]:
    rows, report = prepare_davis_rows(args.davis_root)
    _write_outputs(args.output, rows, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate licensed benchmark data and build PVR manifests"
    )
    subparsers = parser.add_subparsers(dest="benchmark", required=True)
    tokenbench = subparsers.add_parser("tokenbench")
    tokenbench.add_argument("--official-list", required=True)
    tokenbench.add_argument("--bdd100k-root", required=True)
    tokenbench.add_argument("--bridgedata-v2-root", required=True)
    tokenbench.add_argument("--panda-70m-root", required=True)
    tokenbench.add_argument("--egoexo-4d-root", required=True)
    tokenbench.add_argument("--output", required=True)
    tokenbench.set_defaults(function=_tokenbench)

    davis = subparsers.add_parser("davis")
    davis.add_argument("--davis-root", required=True)
    davis.add_argument("--output", required=True)
    davis.set_defaults(function=_davis)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = args.function(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
