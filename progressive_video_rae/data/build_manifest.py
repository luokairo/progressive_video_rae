from __future__ import annotations

import argparse
from pathlib import Path

from .manifest import build_manifest, write_manifests


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deduplicated Progressive VideoRAE manifests")
    parser.add_argument("--csv-spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=20260807)
    parser.add_argument("--split-ratios", type=float, nargs=3, default=(0.95, 0.025, 0.025))
    parser.add_argument("--probe-workers", type=int, default=16)
    parser.add_argument("--probe-video", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    result = build_manifest(
        args.csv_spec,
        split_seed=args.split_seed,
        split_ratios=tuple(args.split_ratios),
        probe=args.probe_video,
        probe_workers=args.probe_workers,
    )
    write_manifests(result, args.output_dir, args.split_seed)
    print(f"Wrote {len(result.records)} records to {args.output_dir}")
    print(result.report)


if __name__ == "__main__":
    main()

