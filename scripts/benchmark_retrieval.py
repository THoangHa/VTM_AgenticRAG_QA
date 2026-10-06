#!/usr/bin/env python
"""Shared benchmark CLI. Defaults to validation; comparison never loads test."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.benchmark_data import load_data
from src.evaluation.benchmark import run_benchmark
from src.evaluation.comparison import compare_dense_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("evaluate", "compare-dense"):
        sub = commands.add_parser(name)
        sub.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
        sub.add_argument("--results-root", type=Path, default=ROOT / "results" / "benchmarks")
        sub.add_argument("--index-root", type=Path, default=ROOT / "indexes")
        sub.add_argument("--dense-config", type=Path, default=None)
        sub.add_argument("--top-k", type=int, default=100)
        sub.add_argument("--k-values", type=int, nargs="+", default=[1, 5, 10, 100])
        sub.add_argument("--batch-size", type=int, default=8)
        sub.add_argument("--rebuild", action="store_true")
    single = commands.choices["evaluate"]
    single.add_argument("--model", choices=["bm25", "bge_m3", "vi_bi_encoder", "hybrid"], required=True)
    single.add_argument("--split", choices=["val", "test"], default="val")
    single.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    single.add_argument("--k1", type=float, default=1.5)
    single.add_argument("--b", type=float, default=0.75)
    single.add_argument("--alpha", type=float, default=None,
                        help="Hybrid-only dense weight in [0, 1] (default: 0.5).")
    single.add_argument("--candidate-k", type=int, default=None,
                        help="Hybrid-only candidates per method before fusion (default: 500).")
    compare = commands.choices["compare-dense"]
    compare.add_argument("--runs", type=Path, nargs=2, help="Compare two completed validation run directories without loading models")
    compare.add_argument("--baseline-path", type=Path, default=ROOT / "configs" / "dense_baseline.yaml")
    args = parser.parse_args()
    try:
        common = dict(top_k=args.top_k, k_values=args.k_values, results_root=args.results_root,
                      index_root=args.index_root, dense_config=args.dense_config,
                      batch_size=args.batch_size, rebuild=args.rebuild)
        if args.command == "evaluate":
            if args.model != "hybrid" and (args.alpha is not None or args.candidate_k is not None):
                raise ValueError("--alpha and --candidate-k apply only to --model hybrid.")
            data = load_data(args.data_dir, args.split)
            run_benchmark(data, args.model, device=args.device, k1=args.k1, b=args.b,
                          alpha=0.5 if args.alpha is None else args.alpha,
                          candidate_k=500 if args.candidate_k is None else args.candidate_k, **common)
        else:
            if not {1, 5, 10, 100}.issubset(args.k_values) or args.top_k < 100:
                raise ValueError("Dense comparison requires cutoffs 1, 5, 10, 100 and depth >= 100.")
            paths = args.runs
            if paths is None:
                data = load_data(args.data_dir, "val")
                paths = [run_benchmark(data, key, device="cuda", **common) for key in ("bge_m3", "vi_bi_encoder")]
            compare_dense_runs(paths, args.results_root, args.baseline_path, args.data_dir)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        parser.exit(1, f"Benchmark failed: {exc}\n")


if __name__ == "__main__":
    main()
