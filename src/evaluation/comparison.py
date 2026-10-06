"""Paired dense comparison and export of a reloadable selected configuration."""
import json
import math
import os
from pathlib import Path
import tempfile
import numpy as np
import yaml

from src.evaluation.benchmark import new_directory, write_json, write_jsonl
from src.evaluation.benchmark_data import ROOT, read_jsonl


def paired_bootstrap(a, b, resamples=10000, seed=42):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or not a.size or not np.isfinite(a - b).all():
        raise ValueError("Bootstrap requires matching nonempty finite query scores.")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples <= 0:
        raise ValueError("resamples must be a positive integer.")
    rng = np.random.default_rng(seed)
    differences = a - b
    means = []
    for start in range(0, resamples, 256):
        indices = rng.integers(0, len(a), size=(min(256, resamples - start), len(a)))
        means.extend(differences[indices].mean(axis=1))
    low, high = np.percentile(means, [2.5, 97.5])
    return {"difference_a_minus_b": float(differences.mean()), "ci95": [float(low), float(high)],
            "resamples": resamples, "seed": seed, "method": "paired_percentile",
            "conclusion": "inconclusive" if low <= 0 <= high else "a_higher" if low > 0 else "b_higher"}


def compare_dense_runs(run_paths, results_root=ROOT / "results" / "benchmarks",
                       baseline_path=ROOT / "configs" / "dense_baseline.yaml", data_dir=ROOT / "data" / "processed"):
    runs = {}
    for path in map(Path, run_paths):
        if json.loads((path / "status.json").read_text())["status"] != "complete":
            raise ValueError("Only complete runs can be compared.")
        summary = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        if summary["model"] in runs:
            raise ValueError("Comparison requires two distinct dense models.")
        runs[summary["model"]] = (path, summary)
    if set(runs) != {"bge_m3", "vi_bi_encoder"}:
        raise ValueError("Comparison requires BGE-M3 and the Vietnamese bi-encoder.")
    a_path, a = runs["bge_m3"]
    b_path, b = runs["vi_bi_encoder"]
    for key in ("fingerprints", "query_count", "corpus_count", "top_k", "k_values", "evaluation_policy", "device"):
        if a[key] != b[key]:
            raise ValueError(f"Incompatible comparison runs: {key} differs.")
    if a["split"] != "val" or b["split"] != "val":
        raise ValueError("Dense selection uses validation only.")
    if a["device"] != "cuda" or any(summary["configuration"]["precision"] != "float16" for summary in (a, b)):
        raise ValueError("This selection protocol requires CUDA FP16 runs.")
    if a["configuration"]["max_seq_length"] != 512 or b["configuration"]["max_seq_length"] != 256:
        raise ValueError("This selection protocol fixes BGE-M3/Vi encoder limits at 512/256.")
    if any(k not in a["metrics"] or k not in b["metrics"] for k in ("NDCG@10", "Recall@10", "MRR@10")):
        raise ValueError("Selection requires metrics at cutoff 10.")
    def per_query(path):
        rows = read_jsonl(path / "per_query.jsonl")
        values = {row["question_idx"]: row["metrics"] for row in rows}
        if len(values) != len(rows) or len(values) != a["query_count"]:
            raise ValueError("Invalid per-query coverage.")
        return values
    a_queries, b_queries = per_query(a_path), per_query(b_path)
    if set(a_queries) != set(b_queries):
        raise ValueError("Per-query IDs differ.")
    qids = sorted(a_queries)
    for summary, rows in ((a, a_queries), (b, b_queries)):
        for metric in ("NDCG@10", "Recall@10", "MRR@10"):
            values = [row[metric] for row in rows.values()]
            if not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in values):
                raise ValueError("Invalid per-query selection metrics.")
            if not math.isclose(math.fsum(values) / len(values), summary["metrics"][metric], abs_tol=1e-12, rel_tol=0):
                raise ValueError("Aggregate selection metrics do not match per-query results.")
    interval = paired_bootstrap([a_queries[q]["NDCG@10"] for q in qids], [b_queries[q]["NDCG@10"] for q in qids])
    winner, reason = "vi_bi_encoder", "Exact tie on all selection metrics; prefer Vietnamese bi-encoder."
    for metric in ("NDCG@10", "Recall@10", "MRR@10"):
        if a["metrics"][metric] != b["metrics"][metric]:
            winner = "bge_m3" if a["metrics"][metric] > b["metrics"][metric] else "vi_bi_encoder"
            reason = f"Higher unrounded {metric}; uncertainty is reported separately."
            break
    comparison_dir = new_directory(results_root, "comparison")
    comparison = {"winner": winner, "selection_reason": reason, "bootstrap_ndcg10": interval,
                  "runs": {name: str(path.resolve()) for name, (path, _) in runs.items()},
                  "fingerprints": a["fingerprints"], "split": "val"}
    write_json(comparison_dir / "comparison.json", comparison)
    write_jsonl(comparison_dir / "per_query_differences.jsonl", ({"question_idx": q,
        "bge_m3_ndcg10": a_queries[q]["NDCG@10"], "vi_bi_encoder_ndcg10": b_queries[q]["NDCG@10"],
        "difference": a_queries[q]["NDCG@10"] - b_queries[q]["NDCG@10"]} for q in qids))
    lines = ["# Dense validation comparison", "", "| Model | nDCG@10 | Recall@10 | MRR@10 | Recall@100 | p50 latency (ms) |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, (_, summary) in runs.items():
        m = summary["metrics"]
        lines.append(f"| {name} | {m['NDCG@10']:.6f} | {m['Recall@10']:.6f} | {m['MRR@10']:.6f} | {m.get('Recall@100', float('nan')):.6f} | {summary['timings']['latency_p50_ms']:.2f} |")
    lines.extend(["", "| Model | Index build (s) | Batched queries/s | Peak allocated CUDA (MiB) | Truncated passages | Truncated queries |",
                  "|---|---:|---:|---:|---:|---:|"])
    for name, (_, summary) in runs.items():
        timings, truncation = summary["timings"], summary.get("truncation") or {}
        peaks = [timings[key] for key in ("prepare_peak_cuda_mib", "evaluation_peak_cuda_mib") if timings.get(key) is not None]
        fields = [timings.get("index_build_seconds"), timings.get("queries_per_second"), max(peaks) if peaks else None,
                  truncation.get("corpus", {}).get("truncated_count"), truncation.get("queries", {}).get("truncated_count")]
        lines.append("| " + name + " | " + " | ".join("n/a" if value is None else f"{value:.2f}" for value in fields) + " |")
    lines.extend(["", f"Selected baseline: **{winner}**. {reason}", "",
        f"BGE-M3 minus Vietnamese encoder nDCG@10: {interval['difference_a_minus_b']:.6f}; paired 95% CI {interval['ci95']}. Conclusion: {interval['conclusion']}.",
        "", "Selection uses validation and fixed 512/256 token limits. Labels credit the original source passage only.",
        "Bootstrap resamples questions; questions sharing source passages may be correlated. No test questions were evaluated."])
    (comparison_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    winner_path, selected = runs[winner]
    config = selected["configuration"]
    baseline_path = Path(baseline_path).resolve()
    data_dir = Path(selected.get("data_directory", data_dir))
    relative = lambda path: os.path.relpath(Path(path).resolve(), baseline_path.parent)
    baseline = {"baseline_model": winner, "selection": {"split": "val", "metric": "NDCG@10",
        "source_run": relative(winner_path), "comparison": relative(comparison_dir),
        "fingerprints": selected["fingerprints"], "evaluation_policy": selected["evaluation_policy"]},
        "paths": {"corpus": relative(Path(data_dir) / "corpus.jsonl"), "data_dir": relative(data_dir),
                  "index_root": relative(selected.get("index_root", ROOT / "indexes")), "results_dir": relative(results_root)},
        "defaults": {"device": selected["device"], "fp16": config["precision"] == "float16",
                     "batch_size": selected["effective_batch_size"], "top_k": selected["top_k"]},
        "models": {winner: {"hf_name": config["hf_name"], "revision": config["revision"],
                            "max_seq_length": config["max_seq_length"], "dim": config["dim"],
                            "batch_size": selected["effective_batch_size"]}}, "encoding": config}
    text = yaml.safe_dump(baseline, allow_unicode=True, sort_keys=False)
    # Relocate relative paths so the immutable snapshot is independently loadable.
    from copy import deepcopy
    snapshot = deepcopy(baseline)
    for key, value in snapshot["paths"].items():
        snapshot["paths"][key] = os.path.relpath((baseline_path.parent / value).resolve(), comparison_dir.resolve())
    for key in ("source_run", "comparison"):
        snapshot["selection"][key] = os.path.relpath((baseline_path.parent / baseline["selection"][key]).resolve(), comparison_dir.resolve())
    (comparison_dir / "selected_baseline.yaml").write_text(yaml.safe_dump(snapshot, allow_unicode=True, sort_keys=False), encoding="utf-8")
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=baseline_path.parent, suffix=".yaml", encoding="utf-8", delete=False) as stream:
        stream.write(text)
        temp = Path(stream.name)
    os.replace(temp, baseline_path)
    from src.evaluation.reporting import publish_latest_comparison
    report_path = publish_latest_comparison(comparison_dir, results_root)
    print(f"Selected {winner}; open in a browser: {report_path}", flush=True)
    return comparison_dir
