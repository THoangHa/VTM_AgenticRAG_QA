import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml
from src.evaluation.adapters import BM25Adapter, DenseAdapter
from src.evaluation.benchmark import run_benchmark, write_json, write_jsonl
from src.evaluation.benchmark_data import BenchmarkData, ROOT, load_data
from src.evaluation.comparison import compare_dense_runs, paired_bootstrap
from src.evaluation.evaluator import evaluate


class FakeRetriever:
    def retrieve(self, text, top_k):
        return [{"doc_id": "a", "score": 2}, {"doc_id": "b", "score": 1}][:top_k]

    def retrieve_many(self, texts, top_k):
        return [self.retrieve(text, top_k) for text in texts]


class FutureAdapter:
    def search(self, queries, top_k):
        return {qid: dict(list({"a": 2, "b": 1}.items())[:top_k]) for qid in queries}


def sample_data(tmp_path):
    return BenchmarkData([{"doc_id": "a", "text": "A"}, {"doc_id": "b", "text": "B"}],
                         {"q1": "A?", "q2": "B?"}, {"q1": {"a": 1}, "q2": {"b": 1}}, "val", tmp_path)


def test_adapter_interchangeability():
    queries = {"q1": "A?"}
    bm25, dense = object.__new__(BM25Adapter), object.__new__(DenseAdapter)
    bm25.retriever = dense.retriever = FakeRetriever()
    reports = [evaluate({"q1": {"a": 1}}, adapter.search(queries, 2), [1, 2])
               for adapter in (bm25, dense, FutureAdapter())]
    assert reports[0] == reports[1] == reports[2]


def test_runner_unique_outputs_and_failure_status(tmp_path):
    data = sample_data(tmp_path)
    args = dict(results_root=tmp_path / "runs", adapter_factory=FutureAdapter, top_k=2, k_values=[1, 2])
    first = run_benchmark(data, "future", **args)
    before = (first / "metrics.json").read_bytes()
    second = run_benchmark(data, "future", **args)
    assert first != second and (first / "metrics.json").read_bytes() == before
    assert json.loads(before)["query_count"] == 2
    assert json.loads(before)["timings"]["latency_query_count"] == 2
    class Invalid(FutureAdapter):
        def search(self, queries, top_k):
            return {qid: {"unknown": 1} for qid in queries}
    with pytest.raises(ValueError, match="missing"):
        run_benchmark(data, "bad", **{**args, "adapter_factory": Invalid})
    failed = next((tmp_path / "runs").glob("bad_*"))
    assert json.loads((failed / "status.json").read_text())["status"] == "failed"


def test_cutoff_depth_validation(tmp_path):
    with pytest.raises(ValueError, match="depth"):
        run_benchmark(sample_data(tmp_path), "future", top_k=5, k_values=[10])


def test_runner_rejects_missing_labelled_document(tmp_path):
    data = sample_data(tmp_path)
    data.qrels["q1"] = {"missing": 1}
    with pytest.raises(ValueError, match="labelled document"):
        run_benchmark(data, "future", adapter_factory=FutureAdapter)


def test_dense_retry_metadata_is_saved(tmp_path):
    class Recorded(FakeRetriever):
        device = "cpu"
        initial_batch_size, batch_size = 8, 2
        oom_retries = [{"from_batch": 8, "to_batch": 4}, {"from_batch": 4, "to_batch": 2}]
        index_metadata = {"build_seconds": 1}
        configuration = {"model": "fake", "precision": "float32"}
        def audit_lengths(self, texts): return {"count": len(list(texts)), "truncated_count": 0}
    class Adapter(FutureAdapter):
        def __init__(self): self.retriever = Recorded()
    path = run_benchmark(sample_data(tmp_path), "fake", results_root=tmp_path / "runs",
                         adapter_factory=Adapter, top_k=2, k_values=[1, 2])
    summary = json.loads((path / "metrics.json").read_text())
    assert summary["effective_batch_size"] == 2
    assert summary["oom_retries"] == Recorded.oom_retries


def completed_run(path, model, ndcg, fingerprints=None, split="val"):
    path.mkdir()
    write_json(path / "status.json", {"status": "complete"})
    metrics = {"NDCG@10": ndcg, "Recall@10": ndcg, "MRR@10": ndcg, "Recall@100": 1.0}
    summary = {"model": model, "split": split, "fingerprints": fingerprints or {"corpus": "c", "queries": "q", "qrels": "r"},
               "query_count": 2, "corpus_count": 2, "top_k": 100, "k_values": [1, 5, 10, 100],
               "evaluation_policy": {"tie": "ascending"}, "device": "cuda", "metrics": metrics,
               "configuration": {"hf_name": "fake", "revision": "revision", "dim": 2,
                                 "max_seq_length": 512 if model == "bge_m3" else 256, "precision": "float16"},
               "effective_batch_size": 8, "timings": {"latency_p50_ms": 1}}
    write_json(path / "metrics.json", summary)
    write_jsonl(path / "per_query.jsonl", [{"question_idx": q, "metrics": metrics} for q in ("q1", "q2")])
    return path


def test_comparison_export_and_bootstrap(tmp_path):
    a = completed_run(tmp_path / "a", "bge_m3", .7)
    b = completed_run(tmp_path / "b", "vi_bi_encoder", .6)
    config = tmp_path / "configs/baseline.yaml"
    report = compare_dense_runs([a, b], tmp_path / "comparisons", config, tmp_path)
    assert yaml.safe_load(config.read_text())["baseline_model"] == "bge_m3"
    assert yaml.safe_load(config.read_text())["models"]["bge_m3"]["revision"] == "revision"
    snapshot = yaml.safe_load((report / "selected_baseline.yaml").read_text())
    exported = yaml.safe_load(config.read_text())
    assert snapshot["models"] == exported["models"]
    for key, value in exported["paths"].items():
        assert (config.parent / value).resolve() == (report / snapshot["paths"][key]).resolve()
    ci = json.loads((report / "comparison.json").read_text())["bootstrap_ndcg10"]
    pointer = json.loads((tmp_path / "comparisons/latest.json").read_text())
    assert pointer["comparison"] == report.name and pointer["winner"] == "bge_m3"
    assert report.name in (tmp_path / "comparisons/LATEST.txt").read_text()
    assert ci["ci95"] == pytest.approx([.1, .1])
    assert paired_bootstrap([.1, .5, .2], [.2, .4, .3]) == paired_bootstrap([.1, .5, .2], [.2, .4, .3])
    assert paired_bootstrap([.5, .5], [.5, .5])["conclusion"] == "inconclusive"


def test_exact_tie_and_incompatible_runs(tmp_path):
    a = completed_run(tmp_path / "a", "bge_m3", .7)
    b = completed_run(tmp_path / "b", "vi_bi_encoder", .7)
    report = compare_dense_runs([a, b], tmp_path / "comparisons", tmp_path / "baseline.yaml")
    assert json.loads((report / "comparison.json").read_text())["winner"] == "vi_bi_encoder"
    summary = json.loads((b / "metrics.json").read_text())
    summary["fingerprints"]["queries"] = "different"
    write_json(b / "metrics.json", summary)
    latest_before = (tmp_path / "comparisons/latest.json").read_bytes()
    with pytest.raises(ValueError, match="fingerprints"):
        compare_dense_runs([a, b], tmp_path / "comparisons", tmp_path / "baseline.yaml")
    assert (tmp_path / "comparisons/latest.json").read_bytes() == latest_before
    summary["fingerprints"]["queries"] = "q"
    summary["split"] = "test"
    write_json(b / "metrics.json", summary)
    with pytest.raises(ValueError, match="validation"):
        compare_dense_runs([a, b], tmp_path / "comparisons", tmp_path / "baseline.yaml")


def test_cli_from_other_directory(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    write_jsonl(data / "corpus.jsonl", [{"doc_id": "a", "text": "Cam thảo là vị thuốc."},
                                       {"doc_id": "b", "text": "Chế độ ăn và sức khỏe."}])
    write_jsonl(data / "val_qas.jsonl", [{"question_idx": "q", "question": "Cam thảo?"}])
    write_json(data / "qrels_val.json", {"q": "a"})
    command = [sys.executable, str(ROOT / "scripts/benchmark_retrieval.py"), "evaluate", "--model", "bm25",
               "--data-dir", str(data), "--results-root", str(tmp_path / "runs"),
               "--index-root", str(tmp_path / "indexes")]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    run = next((tmp_path / "runs").iterdir())
    assert json.loads((run / "metrics.json").read_text())["query_count"] == 1
    assert not (tmp_path / "results").exists()
    assert len(load_data(data).queries) == 1
