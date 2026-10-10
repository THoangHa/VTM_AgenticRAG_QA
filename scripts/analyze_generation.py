#!/usr/bin/env python
"""Compare completed generation arms with paired intervals and judge tests."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.benchmark_data import read_jsonl
from src.evaluation.qa_statistics import (cohen_kappa_correctness, metric_judge_alignment,
    mcnemar_correctness, paired_metric_intervals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "qa" / "paired_analysis.json")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--agreement-labels", type=Path,
                        help="Optional hand/second-judge JSONL with the same correctness labels.")
    args = parser.parse_args()
    try:
        runs = [args.run_a.resolve(), args.run_b.resolve()]
        manifests = [json.loads((path / "manifest.json").read_text(encoding="utf-8")) for path in runs]
        statuses = [json.loads((path / "status.json").read_text(encoding="utf-8")) for path in runs]
        if any(status.get("status") != "complete" for status in statuses):
            raise ValueError("Paired test analysis requires complete runs.")
        if any(manifests[0].get(key) != manifests[1].get(key) for key in ("split", "fingerprints")):
            raise ValueError("Runs must use the same split and data fingerprints.")
        summaries = [json.loads((path / "metrics.json").read_text(encoding="utf-8")) for path in runs]
        maps = [summary["per_question"] for summary in summaries]
        intervals = paired_metric_intervals({"a": maps[0], "b": maps[1]}, "a", "b", args.resamples, args.seed)
        result = {"run_a": str(runs[0]), "run_b": str(runs[1]), "split": manifests[0]["split"],
                  "fingerprints": manifests[0]["fingerprints"], "paired_bootstrap": intervals}
        judge_paths = [path / "judge" / "judgments.jsonl" for path in runs]
        if all(path.is_file() for path in judge_paths):
            labels = [{row["question_idx"]: row for row in read_jsonl(path)} for path in judge_paths]
            result["mcnemar_correctness"] = mcnemar_correctness(labels[0], labels[1])
            result["metric_judge_alignment"] = {
                "run_a": metric_judge_alignment(maps[0], labels[0]),
                "run_b": metric_judge_alignment(maps[1], labels[1])}
            if args.agreement_labels:
                agreement = {row["question_idx"]: row for row in read_jsonl(args.agreement_labels)}
                result["judge_validation"] = {
                    "run_a": cohen_kappa_correctness(labels[0], agreement),
                    "run_b": cohen_kappa_correctness(labels[1], agreement)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(args.output)
    except (ValueError, KeyError, FileNotFoundError, OSError) as exc:
        parser.exit(1, f"Generation analysis failed: {exc}\n")


if __name__ == "__main__":
    main()
