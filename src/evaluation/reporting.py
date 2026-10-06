"""Readable comparison pages and a stable entry point to completed results."""
from html import escape
import json
import os
from pathlib import Path
import tempfile
from urllib.parse import quote


NAMES = {"bge_m3": "BGE-M3", "vi_bi_encoder": "Vietnamese bi-encoder"}


def _atomic_text(path, text):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     suffix=".tmp", delete=False) as stream:
        stream.write(text)
        staged = Path(stream.name)
    os.replace(staged, path)


def _page(comparison_dir, comparison, runs, base):
    def link(path, label):
        relative = os.path.relpath(Path(path).resolve(), base.resolve()).replace("\\", "/")
        return f'<a href="{quote(relative, safe="/.")}">{escape(label)}</a>'

    def value(number, digits=3):
        return "—" if number is None else f"{number:,.{digits}f}"

    quality, performance, sources = [], [], []
    for model, (path, summary) in runs.items():
        name = escape(NAMES.get(model, model))
        metrics, timings = summary["metrics"], summary["timings"]
        selected = ' class="selected"' if model == comparison["winner"] else ""
        quality.append(f"<tr{selected}><th>{name}</th>" + "".join(
            f"<td>{value(metrics.get(key), 6)}</td>" for key in
            ("NDCG@10", "Recall@10", "MRR@10", "Recall@100")) + "</tr>")
        truncation = summary.get("truncation") or {}
        peaks = [timings[key] for key in ("prepare_peak_cuda_mib", "evaluation_peak_cuda_mib")
                 if timings.get(key) is not None]
        diagnostics = (timings.get("index_build_seconds"), timings.get("queries_per_second"),
                       timings.get("latency_p50_ms"), max(peaks) if peaks else None,
                       truncation.get("corpus", {}).get("truncated_count"))
        performance.append(f"<tr><th>{name}</th>" + "".join(
            f"<td>{value(number, 0 if index == 4 else 2)}</td>"
            for index, number in enumerate(diagnostics)) + "</tr>")
        sources.append(f"<li><strong>{name}</strong>: {escape(path.name)} — "
                       + link(path / "metrics.json", "detailed metrics") + "</li>")
    interval = comparison["bootstrap_ndcg10"]
    conclusion = ("The observed difference is inconclusive: the interval includes zero."
                  if interval["conclusion"] == "inconclusive" else
                  "The interval excludes zero in this paired question-bootstrap analysis.")
    summary = next(iter(runs.values()))[1]
    winner = escape(NAMES.get(comparison["winner"], comparison["winner"]))
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Latest retrieval benchmark</title>
<style>
body{margin:0;background:#f3f6fa;color:#192b3c;font:16px/1.65 system-ui,sans-serif}
main{max-width:1050px;margin:40px auto;padding:0 24px}h1{font-size:32px;line-height:1.25}
h2{font-size:21px;margin-top:0}section{background:white;padding:24px;border-radius:12px;margin:20px 0}
.eyebrow{color:#526679;font-size:14px}.winner{border-left:5px solid #167454}
.table{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:14px}
th,td{padding:12px;text-align:right;border-bottom:1px solid #e2e8ef}th:first-child{text-align:left}
.selected{background:#eaf6ef}a{color:#1761a8}li{margin:12px 0;overflow-wrap:anywhere}
.folder{overflow-wrap:anywhere;font-family:monospace}footer{color:#526679;font-size:14px;margin:24px 0}
</style></head><body><main>
<p class="eyebrow">Completed validation comparison</p><h1>Retrieval benchmark results</h1>
""" + f"""
<p>{summary['query_count']:,} validation questions · {summary['corpus_count']:,} passages · depth {summary['top_k']}</p>
<section class="winner"><h2>Selected dense baseline: {winner}</h2>
<p>{escape(comparison['selection_reason'])}</p>
<p>Current comparison folder:<br><span class="folder">{escape(comparison_dir.name)}</span></p></section>
<section><h2>Retrieval quality</h2><div class="table"><table>
<thead><tr><th>Model</th><th>nDCG@10</th><th>Recall@10</th><th>MRR@10</th><th>Recall@100</th></tr></thead>
<tbody>{''.join(quality)}</tbody></table></div>
<p>The highlighted model is selected using unrounded nDCG@10, then Recall@10 and MRR@10 for exact ties.</p></section>
<section><h2>Uncertainty</h2><p>BGE-M3 minus Vietnamese encoder nDCG@10:
<strong>{value(interval['difference_a_minus_b'], 6)}</strong>.
Paired 95% confidence interval: [{value(interval['ci95'][0], 6)}, {value(interval['ci95'][1], 6)}].</p>
<p>{escape(conclusion)} {interval['resamples']:,} resamples; seed {interval['seed']}.</p></section>
<section><h2>Performance diagnostics</h2><div class="table"><table>
<thead><tr><th>Model</th><th>Index build (s)</th><th>Queries/s</th><th>p50 latency (ms)</th><th>Peak CUDA (MiB)</th><th>Truncated passages</th></tr></thead>
<tbody>{''.join(performance)}</tbody></table></div>
<p>Speed and memory do not affect selection. Cached index build times are historical; CUDA memory measures PyTorch allocations.</p></section>
<section><h2>Source runs and files</h2><ul>{''.join(sources)}
<li>{link(comparison_dir / 'comparison.json', 'Comparison data (JSON)')}</li>
<li>{link(comparison_dir / 'selected_baseline.yaml', 'Frozen model configuration (YAML)')}</li></ul></section>
<footer>This comparison uses validation only and fixed 512/256 token limits. Labels credit the original source passage.
Questions sharing passages may be correlated. No test questions were evaluated.</footer>
</main></body></html>
"""


def publish_latest_comparison(comparison_dir, results_root=None):
    """Publish an existing successful comparison without rerunning any models."""
    comparison_dir = Path(comparison_dir).resolve()
    root = Path(results_root or comparison_dir.parent).resolve()
    comparison = json.loads((comparison_dir / "comparison.json").read_text(encoding="utf-8"))
    runs = {}
    for model, path in comparison["runs"].items():
        path = Path(path)
        status = json.loads((path / "status.json").read_text(encoding="utf-8"))
        if status["status"] != "complete":
            raise ValueError("Latest report must reference completed runs.")
        runs[model] = (path, json.loads((path / "metrics.json").read_text(encoding="utf-8")))
    # Old immutable reports remain unchanged; add a browser-readable version.
    report = comparison_dir / "REPORT.html"
    if not report.exists():
        _atomic_text(report, _page(comparison_dir, comparison, runs, comparison_dir))
    root.mkdir(parents=True, exist_ok=True)
    _atomic_text(root / "latest.html", _page(comparison_dir, comparison, runs, root))
    lines = ["LATEST COMPLETED DENSE VALIDATION COMPARISON", "",
             f"Comparison folder: {comparison_dir.name}",
             f"Selected baseline: {NAMES.get(comparison['winner'], comparison['winner'])}",
             comparison["selection_reason"], "",
             "Model                       nDCG@10   Recall@10   MRR@10"]
    for model, (path, summary) in runs.items():
        metrics = summary["metrics"]
        lines.append(f"{NAMES.get(model, model):<27} {metrics['NDCG@10']:.6f}  {metrics['Recall@10']:.6f}    {metrics['MRR@10']:.6f}")
    interval = comparison["bootstrap_ndcg10"]
    lines.extend(["", f"BGE-M3 minus Vietnamese encoder nDCG@10: {interval['difference_a_minus_b']:.6f}",
                  f"Paired 95% confidence interval: {interval['ci95']}",
                  f"Uncertainty result: {interval['conclusion']}", "", "SOURCE RUN FOLDERS"])
    lines.extend(f"{NAMES.get(model, model)}: {path.name}" for model, (path, _) in runs.items())
    lines.extend(["", "For formatted tables, double-click latest.html in File Explorer.",
                  "No test questions were evaluated."])
    _atomic_text(root / "LATEST.txt", "\n".join(lines) + "\n")
    pointer = {"comparison": os.path.relpath(comparison_dir, root).replace("\\", "/"),
               "report": "latest.html", "winner": comparison["winner"]}
    _atomic_text(root / "latest.json", json.dumps(pointer, indent=2) + "\n")
    return root / "latest.html"
