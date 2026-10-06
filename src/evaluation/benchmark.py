"""Run any retrieval adapter, persist independent runs, and measure performance."""
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import platform
import time
from uuid import uuid4

import numpy as np
from tqdm import tqdm
from src.evaluation.benchmark_data import ROOT, validate_corpus
from src.evaluation.evaluator import evaluate, ranked_items, validate_cutoffs

POLICY = {"version": 1, "tie_break": "score_desc_doc_id_asc", "ndcg_gain": "linear",
          "missing_queries": "zero", "equal_query_document_ids": "retain"}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def new_directory(root, prefix):
    run_id = f"{prefix}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:12]}"
    path = Path(root) / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def environment():
    names = ("numpy", "torch", "faiss-cpu", "sentence-transformers", "transformers", "pyvi",
             "rank-bm25", "underthesea", "underthesea_core", "datasets", "huggingface-hub")
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return {"python": platform.python_version(), "platform": platform.platform(), "dependencies": versions}


def run_benchmark(data, model_key, *, top_k=100, k_values=(1, 5, 10, 100),
                  results_root=ROOT / "results" / "benchmarks", index_root=ROOT / "indexes",
                  dense_config=None, device="cuda", batch_size=8, rebuild=False,
                  k1=1.5, b=0.75, alpha=0.5, candidate_k=500, adapter_factory=None):
    ks = validate_cutoffs(k_values)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < max(ks):
        raise ValueError("Retrieval depth must be at least the largest metric cutoff.")
    validate_corpus(data.corpus)
    evaluate(data.qrels, {}, (1,))
    if set(data.queries) != set(data.qrels):
        raise ValueError("Question/qrels IDs must match.")
    corpus_ids = {doc["doc_id"] for doc in data.corpus}
    if any(did not in corpus_ids for labels in data.qrels.values() for did in labels):
        raise ValueError("A labelled document is missing from the corpus.")
    path = new_directory(results_root, model_key + "_" + data.split)
    write_json(path / "status.json", {"status": "running", "model": model_key})
    adapter, r = None, None
    try:
        from src.evaluation.adapters import BM25Adapter, DenseAdapter, HybridAdapter, validate_hybrid_parameters
        if model_key == "hybrid":
            validate_hybrid_parameters(alpha, candidate_k, top_k)
        uses_dense = model_key in {"bge_m3", "vi_bi_encoder"} or (model_key == "hybrid" and alpha > 0)
        if uses_dense and device == "cuda" and adapter_factory is None:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for this benchmark.")
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        if adapter_factory:
            adapter = adapter_factory()
        elif model_key == "bm25":
            adapter = BM25Adapter(data, Path(index_root), rebuild, k1, b)
        elif model_key == "hybrid":
            adapter = HybridAdapter(data, Path(index_root), dense_config, rebuild, device,
                                    batch_size, alpha, candidate_k, k1, b)
        else:
            from src.retrieval.dense.dense_retriever import DEFAULT_CONFIG_PATH
            adapter = DenseAdapter(model_key, data, Path(index_root), dense_config or DEFAULT_CONFIG_PATH,
                                   rebuild, device, batch_size)
        prepare_seconds = time.perf_counter() - started
        r = getattr(adapter, "retriever", None)
        is_dense = r is not None and hasattr(r, "audit_lengths")
        cuda = is_dense and r.device == "cuda"
        if cuda:
            import torch
            sync = torch.cuda.synchronize
            prepare_peak = torch.cuda.max_memory_allocated() / 2**20
            torch.cuda.reset_peak_memory_stats()
        else:
            sync = lambda: None
            prepare_peak = None
        audits = ({"corpus": r.audit_lengths(d["text"] for d in data.corpus),
                   "queries": r.audit_lengths(data.queries.values())} if is_dense else None)
        sync()
        started = time.perf_counter()
        results = adapter.search(data.queries, top_k)
        sync()
        search_seconds = time.perf_counter() - started
        if set(results) != set(data.queries):
            raise ValueError("Adapter must return one ranking (possibly empty) for every query.")
        if any(did not in corpus_ids for scores in results.values() for did in scores):
            raise ValueError("Retrieved document is missing from the corpus.")
        if any(len(scores) > top_k for scores in results.values()):
            raise ValueError("Adapter returned more documents than requested.")
        report = evaluate(data.qrels, results, ks)
        for qid in list(data.queries)[:10]:
            adapter.search({qid: data.queries[qid]}, top_k)
        sync()
        latencies = []
        for qid, text in tqdm(data.queries.items(), desc=f"Latency {model_key}", unit="q"):
            sync()
            started = time.perf_counter()
            adapter.search({qid: text}, top_k)
            sync()
            latencies.append(time.perf_counter() - started)
        config = getattr(adapter, "configuration", None)
        if config is None:
            config = r.configuration if r is not None else {"model": model_key}
        cache_meta = getattr(r, "index_metadata", {})
        timings = {"prepare_seconds": prepare_seconds, "cache_reused": getattr(adapter, "cache_reused", False),
                   "index_build_seconds": cache_meta.get("build_seconds", None if getattr(adapter, "cache_reused", False) else prepare_seconds),
                   "search_seconds": search_seconds, "queries_per_second": len(data.queries) / search_seconds,
                   "latency_warmup_queries": min(10, len(data.queries)),
                   "latency_query_count": len(latencies), "latency_mean_ms": float(np.mean(latencies) * 1000),
                   "latency_p50_ms": float(np.percentile(latencies, 50) * 1000),
                   "latency_p95_ms": float(np.percentile(latencies, 95) * 1000),
                   "prepare_peak_cuda_mib": prepare_peak,
                   "evaluation_peak_cuda_mib": torch.cuda.max_memory_allocated() / 2**20 if cuda else None}
        if model_key == "hybrid":
            # Dense historical build time is not the build time of both hybrid components.
            timings["index_build_seconds"] = None
            timings["components"] = getattr(adapter, "component_timings", {})
        summary = {"schema_version": 1, "model": model_key, "split": data.split,
                   "data_directory": str(data.directory.resolve()), "index_root": str(Path(index_root).resolve()),
                   "query_count": report.query_count, "corpus_count": len(data.corpus),
                   "top_k": top_k, "k_values": ks, "metrics": report.metrics, "configuration": config,
                   "fingerprints": data.fingerprints, "evaluation_policy": POLICY, "environment": environment(),
                   "truncation": audits, "timings": timings,
                   "device": r.device if is_dense else "cpu", "faiss_threads": 1 if is_dense else None,
                   "initial_batch_size": r.initial_batch_size if is_dense else None,
                   "effective_batch_size": r.batch_size if is_dense else None,
                   "oom_retries": r.oom_retries if is_dense else []}
        if cuda:
            summary["environment"]["gpu"] = torch.cuda.get_device_name()
            summary["environment"]["cuda_runtime"] = torch.version.cuda
            summary["environment"]["gpu_total_memory_mib"] = torch.cuda.get_device_properties(0).total_memory / 2**20
        write_jsonl(path / "rankings.jsonl", ({"question_idx": qid, "results": [
            {"doc_id": did, "score": score} for did, score in ranked_items(scores)]} for qid, scores in results.items()))
        write_jsonl(path / "per_query.jsonl", ({"question_idx": qid, "metrics": metrics} for qid, metrics in report.per_query.items()))
        write_json(path / "metrics.json", summary)
        write_json(path / "status.json", {"status": "complete"})
        print(f"Saved {model_key} benchmark: {path}", flush=True)
        return path
    except Exception as exc:
        write_json(path / "status.json", {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        # Sequential runs never retain the previous model in VRAM.
        if adapter is not None:
            adapter, r = None, None
            import gc
            gc.collect()
            if "torch" in __import__("sys").modules:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
