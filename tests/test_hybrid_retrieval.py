import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from src.evaluation import adapters
from src.evaluation.adapters import HybridAdapter, _min_max_normalise, load_hybrid_baseline
from src.evaluation.benchmark import run_benchmark
from src.evaluation.benchmark_data import BenchmarkData, ROOT
from src.evaluation.evaluator import ranked_items


def dataset(tmp_path):
    return BenchmarkData([{"doc_id": did, "text": did} for did in "abcd"],
                         {"q": "Question?"}, {"q": {"a": 1}}, "val", tmp_path)


def baseline(tmp_path, model="bge_m3"):
    path = tmp_path / "selected.yaml"
    path.write_text(yaml.safe_dump({"baseline_model": model,
        "models": {model: {"hf_name": "fake", "revision": "a" * 40, "max_seq_length": 512, "dim": 2}},
        "defaults": {"device": "cuda", "fp16": True, "batch_size": 8, "top_k": 100},
        "selection": {"source_run": "saved-validation-run", "split": "val"}}), encoding="utf-8")
    return path


class FakeComponent:
    def __init__(self, scores, model, cache_reused=False):
        self.scores = scores
        self.calls = []
        self.cache_reused = cache_reused
        self.index_build_seconds = None if cache_reused else 0.1
        self.retriever = SimpleNamespace(configuration={"model": model, "revision": "a" * 40},
            index_metadata={"build_seconds": 0.2}, device="cpu", initial_batch_size=8,
            batch_size=8, oom_retries=[])
        if model != "bm25":
            self.retriever.audit_lengths = lambda texts: {"count": len(list(texts)), "truncated_count": 0}

    def search(self, queries, top_k):
        self.calls.append(top_k)
        return {qid: dict(ranked_items(self.scores)[:top_k]) for qid in queries}


@pytest.fixture
def components(monkeypatch):
    state = {"initializations": []}

    def bm25(data, index_root, rebuild, k1, b):
        state["initializations"].append("bm25")
        state["bm25"] = FakeComponent({"a": 8.0, "b": 4.0, "c": 0.0}, "bm25")
        return state["bm25"]

    def dense(model, data, index_root, config_path, rebuild, device, batch_size):
        state["initializations"].append("dense")
        state["dense_args"] = (model, Path(config_path), device, batch_size)
        state["dense"] = FakeComponent({"b": 0.9, "d": 0.6, "c": 0.3}, model)
        return state["dense"]

    monkeypatch.setattr(adapters, "BM25Adapter", bm25)
    monkeypatch.setattr(adapters, "DenseAdapter", dense)
    return state


def hybrid(tmp_path, **kwargs):
    return HybridAdapter(dataset(tmp_path), tmp_path / "indexes", baseline(tmp_path),
                         device="cpu", candidate_k=500, **kwargs)


def test_hand_calculated_fusion_and_candidate_depth(tmp_path, components):
    adapter = hybrid(tmp_path)
    results = adapter.search({"q": "Question?"}, 3)
    assert list(results["q"]) == ["b", "a", "d"]
    assert results["q"] == pytest.approx({"b": 0.75, "a": 0.5, "d": 0.25})
    assert components["bm25"].calls == [500]
    assert components["dense"].calls == [500]
    assert adapter.search({}, 3) == {}


@pytest.mark.parametrize("alpha,active", [(0.0, "bm25"), (1.0, "dense")])
def test_endpoints_preserve_original_scores_and_do_not_initialize_inactive(tmp_path, components, alpha, active):
    adapter = hybrid(tmp_path, alpha=alpha)
    assert components["initializations"] == [active]
    # Include a zero score whose ID could otherwise lose a union tie at the cutoff.
    components[active].scores = {"z": 5.0, "y": 0.0}
    result = adapter.search({"q": "Question?"}, 2)
    assert result == {"q": {"z": 5.0, "y": 0.0}}
    assert components[active].calls == [2]
    config = adapter.configuration
    assert config["components"][active]["active"] is True
    inactive = "dense" if active == "bm25" else "bm25"
    assert config["components"][inactive] == {"active": False, "configuration": None}
    assert adapter.component_timings[inactive]["prepare_seconds"] is None


def test_disjoint_constant_empty_and_tied_lists(tmp_path, components):
    adapter = hybrid(tmp_path)
    components["bm25"].scores = {"b": 2.0, "a": 2.0}
    components["dense"].scores = {"d": 3.0, "c": 1.0}
    assert adapter.search({"q": "Question?"}, 4) == {"q": {"d": 0.5, "a": 0.0, "b": 0.0, "c": 0.0}}
    components["dense"].scores = {}
    assert list(adapter.search({"q": "Question?"}, 2)["q"]) == ["a", "b"]
    components["bm25"].scores = {}
    assert adapter.search({"q": "Question?"}, 4) == {"q": {}}


def test_normalization_empty_constant_negative_and_extreme_scores():
    assert _min_max_normalise({}) == {}
    assert _min_max_normalise({"a": 4.0}) == {"a": 0.0}
    assert _min_max_normalise({"a": 0.0, "b": 0.0}) == {"a": 0.0, "b": 0.0}
    assert _min_max_normalise({"a": -3.0, "b": -1.0}) == {"a": 0.0, "b": 1.0}
    assert _min_max_normalise({"a": -1e308, "b": 0.0, "c": 1e308}) == {"a": 0.0, "b": 0.5, "c": 1.0}


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf"), True, "1"])
def test_nonfinite_or_invalid_component_scores_rejected(tmp_path, components, score):
    adapter = hybrid(tmp_path)
    # Deliver malformed output directly so the component's fake sorter does not
    # reject it before the hybrid adapter can validate the score.
    components["bm25"].search = lambda queries, top_k: {qid: {"a": score} for qid in queries}
    with pytest.raises(ValueError, match="finite real"):
        adapter.search({"q": "Question?"}, 1)


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan"), float("inf"), True, "0.5"])
def test_invalid_alpha_before_initialization(tmp_path, components, alpha):
    with pytest.raises(ValueError, match="alpha"):
        hybrid(tmp_path, alpha=alpha)
    assert components["initializations"] == []


@pytest.mark.parametrize("depth", [0, -1, True, 2.5])
def test_invalid_candidate_depth_before_initialization(tmp_path, components, depth):
    with pytest.raises(ValueError, match="candidate_k"):
        HybridAdapter(dataset(tmp_path), tmp_path, baseline(tmp_path), candidate_k=depth)
    assert components["initializations"] == []


def test_output_depth_and_component_coverage_validation(tmp_path, components):
    adapter = HybridAdapter(dataset(tmp_path), tmp_path, baseline(tmp_path), candidate_k=2)
    for depth in (0, -1, True, 1.5, 3):
        with pytest.raises(ValueError, match="top_k"):
            adapter.search({"q": "Question?"}, depth)
    components["dense"].search = lambda queries, top_k: {}
    with pytest.raises(ValueError, match="every query"):
        adapter.search({"q": "Question?"}, 1)


def test_default_baseline_and_explicit_selected_model(tmp_path, components, monkeypatch):
    path = baseline(tmp_path, "vi_bi_encoder")
    monkeypatch.setattr(adapters, "ROOT", tmp_path)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    default_path = config_dir / "dense_baseline.yaml"
    default_path.write_bytes(path.read_bytes())
    adapter = HybridAdapter(dataset(tmp_path), tmp_path, device="cpu")
    assert components["dense_args"][:2] == ("vi_bi_encoder", default_path.resolve())
    assert adapter.configuration["dense_baseline"]["selection"]["split"] == "val"
    override = baseline(tmp_path, "bge_m3")
    HybridAdapter(dataset(tmp_path), tmp_path, override, device="cpu")
    assert components["dense_args"][:2] == ("bge_m3", override.resolve())


@pytest.mark.parametrize("content", ["[]", "[broken", "models: {}", "baseline_model: unsupported",
    "baseline_model: bge_m3\nmodels: {}", "baseline_model: bge_m3\nmodels:\n  bge_m3: {}"])
def test_invalid_baseline_fails_before_initialization(tmp_path, components, content):
    path = tmp_path / "invalid.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="[Hh]ybrid"):
        HybridAdapter(dataset(tmp_path), tmp_path, path)
    assert components["initializations"] == []


def test_missing_baseline_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_hybrid_baseline(tmp_path / "missing.yaml")


def test_complete_run_records_hybrid_config_and_component_diagnostics(tmp_path, components):
    path = run_benchmark(dataset(tmp_path), "hybrid", results_root=tmp_path / "runs",
        index_root=tmp_path / "indexes", dense_config=baseline(tmp_path), device="cpu",
        top_k=3, k_values=[1, 3], alpha=0.3, candidate_k=500)
    summary = json.loads((path / "metrics.json").read_text())
    assert json.loads((path / "status.json").read_text()) == {"status": "complete"}
    config = summary["configuration"]
    assert (config["model"], config["alpha"], config["candidate_k"]) == ("hybrid", 0.3, 500)
    assert config["components"]["bm25"]["configuration"]["model"] == "bm25"
    assert config["components"]["dense"]["configuration"]["model"] == "bge_m3"
    assert summary["timings"]["index_build_seconds"] is None
    assert summary["timings"]["components"]["bm25"]["index_build_seconds"] == 0.1
    assert summary["timings"]["components"]["dense"]["index_build_historical"] is False
    assert summary["truncation"]["queries"]["count"] == 1
    rows = [json.loads(line) for line in (path / "rankings.jsonl").read_text().splitlines()]
    assert len(rows) == summary["query_count"] == 1
    assert len(rows[0]["results"]) == 3
    assert (path / "per_query.jsonl").exists()


def test_cache_reuse_tracks_each_active_component(tmp_path, components):
    original_bm25, original_dense = adapters.BM25Adapter, adapters.DenseAdapter

    def cached_bm25(*args):
        component = original_bm25(*args)
        component.cache_reused, component.index_build_seconds = True, None
        return component

    def cached_dense(*args):
        component = original_dense(*args)
        component.cache_reused = True
        return component

    from unittest.mock import patch
    with patch.object(adapters, "BM25Adapter", cached_bm25):
        adapter = hybrid(tmp_path)
        assert adapter.cache_reused is False
        assert adapter.component_timings["bm25"]["cache_reused"] is True
        assert adapter.component_timings["bm25"]["index_build_seconds"] is None
        with patch.object(adapters, "DenseAdapter", cached_dense):
            adapter = hybrid(tmp_path)
            assert adapter.cache_reused is True
            assert adapter.component_timings["dense"]["index_build_historical"] is True
        adapter = hybrid(tmp_path, alpha=0)
        assert adapter.cache_reused is True


def test_failed_run_records_invalid_parameters_and_bad_component_scores(tmp_path, components):
    args = dict(results_root=tmp_path / "runs", dense_config=baseline(tmp_path), top_k=3,
                k_values=[1, 3], device="cpu")
    with pytest.raises(ValueError, match="candidate_k"):
        run_benchmark(dataset(tmp_path), "hybrid", candidate_k=2, **args)
    assert components["initializations"] == []
    failed = next((tmp_path / "runs").iterdir())
    assert json.loads((failed / "status.json").read_text())["status"] == "failed"
    assert not (failed / "metrics.json").exists()
    factory = lambda: hybrid(tmp_path)
    adapter = factory()
    components["bm25"].scores = {"a": float("nan")}
    with pytest.raises(ValueError, match="finite real"):
        run_benchmark(dataset(tmp_path), "hybrid", adapter_factory=lambda: adapter, **args)
    assert all(json.loads((path / "status.json").read_text())["status"] == "failed"
               for path in (tmp_path / "runs").iterdir())


def test_cuda_bm25_endpoint_does_not_check_cuda(tmp_path, components, monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: pytest.fail("Inactive CUDA reset"))
    path = run_benchmark(dataset(tmp_path), "hybrid", results_root=tmp_path / "runs",
        dense_config=baseline(tmp_path), device="cuda", alpha=0, top_k=3, k_values=[1, 3])
    assert components["initializations"] == ["bm25"]
    summary = json.loads((path / "metrics.json").read_text())
    assert summary["device"] == "cpu"
    assert summary["truncation"] is None
    assert summary["initial_batch_size"] is None


def test_active_dense_cuda_check_and_reset_precede_initialization(tmp_path, components, monkeypatch):
    import torch
    available = [False]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available[0])
    events = []
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: events.append("reset"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    args = dict(results_root=tmp_path / "runs", dense_config=baseline(tmp_path),
                device="cuda", top_k=3, k_values=[1, 3])
    with pytest.raises(RuntimeError, match="CUDA is required"):
        run_benchmark(dataset(tmp_path), "hybrid", **args)
    assert components["initializations"] == []
    available[0] = True
    original = adapters.DenseAdapter

    def checked_dense(*params):
        assert events == ["reset"]
        return original(*params)

    monkeypatch.setattr(adapters, "DenseAdapter", checked_dense)
    run_benchmark(dataset(tmp_path), "hybrid", **args)
    assert components["initializations"] == ["bm25", "dense"]


def test_cli_hybrid_bm25_endpoint_from_other_directory(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "corpus.jsonl").write_text(
        json.dumps({"doc_id": "a", "text": "Cam thảo là vị thuốc."}) + "\n" +
        json.dumps({"doc_id": "b", "text": "Chế độ ăn và sức khỏe."}) + "\n", encoding="utf-8")
    (data / "val_qas.jsonl").write_text(json.dumps({"question_idx": "q", "question": "Cam thảo?"}) + "\n", encoding="utf-8")
    (data / "qrels_val.json").write_text(json.dumps({"q": "a"}), encoding="utf-8")
    command = [sys.executable, str(ROOT / "scripts/benchmark_retrieval.py"), "evaluate",
        "--model", "hybrid", "--alpha", "0", "--candidate-k", "500", "--dense-config", str(baseline(tmp_path)),
        "--data-dir", str(data), "--results-root", str(tmp_path / "runs"), "--index-root", str(tmp_path / "indexes")]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    run = next((tmp_path / "runs").iterdir())
    summary = json.loads((run / "metrics.json").read_text())
    assert summary["configuration"]["alpha"] == 0
    assert summary["configuration"]["components"]["dense"]["active"] is False
    assert summary["metrics"]["Recall@1"] == 1.0
    assert not (tmp_path / "indexes/dense_bge_m3").exists()


@pytest.mark.parametrize("flag,value", [("--alpha", "0.3"), ("--candidate-k", "500")])
def test_cli_rejects_hybrid_flags_for_other_models(tmp_path, flag, value):
    result = subprocess.run([sys.executable, str(ROOT / "scripts/benchmark_retrieval.py"),
        "evaluate", "--model", "bm25", flag, value], cwd=tmp_path,
        capture_output=True, text=True, encoding="utf-8")
    assert result.returncode != 0
    assert "apply only to --model hybrid" in result.stderr
