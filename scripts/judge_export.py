#!/usr/bin/env python
"""Export blinded judge batches from a completed generation run."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.judge_io import export_judge_batches


def _sample_size(value):
    size = int(value)
    if not 150 <= size <= 200:
        raise argparse.ArgumentTypeError("sample size must be between 150 and 200")
    return size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--sample-manifest", type=Path, required=True,
                        help="Reuse this same sample manifest when exporting every arm.")
    parser.add_argument("--strata-rankings", type=Path,
                        help="Canonical retrieval run used to stratify a new shared sample.")
    parser.add_argument("--batch-size", type=int, choices=[10, 15, 20], default=15)
    parser.add_argument("--sample-size", type=_sample_size, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        paths = export_judge_batches(args.run_dir, args.data_dir, args.sample_manifest,
                                     rankings_path=args.strata_rankings, batch_size=args.batch_size,
                                     sample_size=args.sample_size, seed=args.seed)
        print("\n".join(map(str, paths)))
    except (ValueError, FileNotFoundError, OSError) as exc:
        parser.exit(1, f"Judge export failed: {exc}\n")


if __name__ == "__main__":
    main()
