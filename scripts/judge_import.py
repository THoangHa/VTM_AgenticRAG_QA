#!/usr/bin/env python
"""Validate strict-JSON pasted judgments and archive them with a run."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.judge_io import import_judgements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--responses", type=Path, nargs="+", required=True,
                        help="Files containing strict JSON arrays returned by the judge.")
    parser.add_argument("--judge-model", required=True, help="Exact judge model name used for this session.")
    parser.add_argument("--judge-date", required=True, help="Session date in YYYY-MM-DD format.")
    args = parser.parse_args()
    try:
        manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
        rag = manifest["mode"] != "llm_only"
        output = import_judgements(args.run_dir, args.responses, rag_mode=rag,
                                   sample_manifest_path=args.sample_manifest,
                                   judge_model=args.judge_model, judge_date=args.judge_date)
        print(f"Saved validated judgments: {output}")
    except (ValueError, FileNotFoundError, OSError, KeyError) as exc:
        parser.exit(1, f"Judge import failed: {exc}\n")


if __name__ == "__main__":
    main()
