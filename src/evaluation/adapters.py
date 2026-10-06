"""Adapters isolate model lifecycle and search from the shared evaluator."""
import math
from numbers import Real
from pathlib import Path
import time
from typing import Protocol

import yaml
from src.evaluation.benchmark_data import ROOT, records_to_scores
from src.evaluation.evaluator import ranked_items


class RetrieverAdapter(Protocol):
    def search(self, queries: dict[str, str], top_k: int) -> dict[str, dict[str, float]]: ...


class BM25Adapter:
    def __init__(self, data, index_root, rebuild=False, k1=1.5, b=0.75):
        from src.retrieval.bm25 import BM25Retriever, CacheMismatchError, corpus_fingerprint
        path = index_root / "bm25" / "index.pkl"
        self.retriever = BM25Retriever(k1=k1, b=b)
        self.index_build_seconds = None
        self.cache_reused = False
        if path.exists() and not rebuild:
            try:
                self.retriever = BM25Retriever.load(path,
                    expected_corpus_fingerprint=corpus_fingerprint(data.corpus),
                    expected_config=self.retriever.configuration)
                self.cache_reused = True
            except CacheMismatchError:
                pass
        if not self.cache_reused:
            started = time.perf_counter()
            self.retriever.fit(data.corpus)
            self.index_build_seconds = time.perf_counter() - started
            self.retriever.save(path)

    def search(self, queries, top_k):
        return {qid: records_to_scores(self.retriever.retrieve(text, top_k)) for qid, text in queries.items()}


class DenseAdapter:
    def __init__(self, model_key, data, index_root, config_path, rebuild=False,
                 device="cuda", batch_size=8):
        import faiss
        from src.retrieval.dense.registry import RETRIEVERS
        from src.retrieval.dense.dense_retriever import DenseCacheMismatchError
        faiss.omp_set_num_threads(1)
        self.retriever = RETRIEVERS[model_key](config_path=config_path, device=device,
                                              batch_size=batch_size, require_cuda=device == "cuda")
        r = self.retriever
        r.corpus_path = data.directory / "corpus.jsonl"
        r.index_root, r.index_dir = index_root, index_root / f"dense_{model_key}"
        self.cache_reused = False
        if not rebuild:
            try:
                r.load_index()
                self.cache_reused = True
            except (FileNotFoundError, DenseCacheMismatchError):
                pass
        if not self.cache_reused:
            r.build_index()
        if r._fingerprint(r.corpus) != data.fingerprints["corpus"]:
            raise ValueError("Corpus changed after benchmark inputs were loaded.")

    def search(self, queries, top_k):
        rows = self.retriever.retrieve_many(list(queries.values()), top_k)
        return {qid: records_to_scores(records) for qid, records in zip(queries, rows)}



def validate_hybrid_parameters(alpha, candidate_k, top_k=None):
    """Validate before loading models, and again against each search depth."""
    if (isinstance(alpha, bool) or not isinstance(alpha, Real)
            or not math.isfinite(alpha) or not 0 <= alpha <= 1):
        raise ValueError("alpha must be finite and in [0, 1].")
    if isinstance(candidate_k, bool) or not isinstance(candidate_k, int) or candidate_k <= 0:
        raise ValueError("candidate_k must be a positive integer.")
    if top_k is not None:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        if candidate_k < top_k:
            raise ValueError("candidate_k must be at least top_k.")


def load_hybrid_baseline(config_path=None):
    """Load a selected model without importing or initializing an encoder."""
    path = Path(config_path or ROOT / "configs" / "dense_baseline.yaml").resolve()
    with path.open(encoding="utf-8") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid hybrid baseline YAML: {path}") from exc
    if not isinstance(config, dict):
        raise ValueError("Hybrid dense configuration must be a mapping with baseline_model.")
    model = config.get("baseline_model")
    if not isinstance(model, str) or model not in {"bge_m3", "vi_bi_encoder"}:
        raise ValueError("Hybrid baseline_model must be bge_m3 or vi_bi_encoder.")
    models = config.get("models")
    spec = models.get(model) if isinstance(models, dict) else None
    if not isinstance(spec, dict) or not {"hf_name", "revision", "max_seq_length", "dim"} <= spec.keys():
        raise ValueError(f"Hybrid configuration must include the selected model settings for {model}.")
    return path, config, model


def _min_max_normalise(scores):
    """Constant scores provide no ranking signal; missing scores stay absent."""
    if any(isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(score)
           for score in scores.values()):
        raise ValueError("Hybrid component scores must be finite real numbers.")
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        return {did: 0.0 for did in scores}
    # Scale first to avoid overflow when finite values span almost the full float range.
    scale = max(abs(lo), abs(hi))
    low, high = lo / scale, hi / scale
    return {did: float((score / scale - low) / (high - low)) for did, score in scores.items()}


class HybridAdapter:
    """Candidate-based BM25/dense fusion using the frozen selected baseline."""

    def __init__(self, data, index_root, dense_config_path=None, rebuild=False,
                 device="cuda", batch_size=8, alpha=0.5, candidate_k=500, k1=1.5, b=0.75):
        validate_hybrid_parameters(alpha, candidate_k)
        self.alpha, self.candidate_k = float(alpha), candidate_k
        self.baseline_path, self.baseline, self.dense_model_key = load_hybrid_baseline(dense_config_path)
        self.bm25 = self.dense = None
        self.bm25_parameters = {"k1": k1, "b": b}
        self.component_timings = {}
        for name in ("bm25", "dense"):
            active = self.alpha < 1 if name == "bm25" else self.alpha > 0
            details = {"active": active, "cache_reused": None, "prepare_seconds": None,
                       "index_build_seconds": None, "index_build_historical": None}
            if active:
                started = time.perf_counter()
                if name == "bm25":
                    component = BM25Adapter(data, Path(index_root), rebuild, k1, b)
                    build_seconds = component.index_build_seconds
                else:
                    component = DenseAdapter(self.dense_model_key, data, Path(index_root),
                                             self.baseline_path, rebuild, device, batch_size)
                    build_seconds = component.retriever.index_metadata.get("build_seconds")
                details.update(prepare_seconds=time.perf_counter() - started,
                               cache_reused=component.cache_reused, index_build_seconds=build_seconds,
                               index_build_historical=bool(component.cache_reused) if build_seconds is not None else None)
                setattr(self, name, component)
            self.component_timings[name] = details
        self.cache_reused = all(details["cache_reused"] for details in self.component_timings.values() if details["active"])
        # The runner uses this reference for active dense audits/device/OOM instrumentation.
        self.retriever = (self.dense or self.bm25).retriever

    @property
    def configuration(self):
        return {"model": "hybrid", "alpha": self.alpha, "candidate_k": self.candidate_k,
                "fusion": "min_max_linear", "normalization": "per_query_per_component_candidates",
                "missing_score": 0.0, "constant_scores": 0.0,
                "tie_break": "score_desc_doc_id_asc",
                "endpoint_policy": "original_active_component_scores",
                "bm25_parameters": self.bm25_parameters,
                "dense_baseline": {"path": str(self.baseline_path), "baseline_model": self.dense_model_key,
                                   "selection": self.baseline.get("selection"),
                                   "requested_model": self.baseline["models"][self.dense_model_key],
                                   "requested_defaults": self.baseline.get("defaults"),
                                   "encoding": self.baseline.get("encoding")},
                "components": {name: {"active": component is not None,
                                      "configuration": component.retriever.configuration if component else None}
                               for name, component in (("bm25", self.bm25), ("dense", self.dense))}}

    def search(self, queries, top_k):
        validate_hybrid_parameters(self.alpha, self.candidate_k, top_k)
        if self.alpha == 0:
            return self.bm25.search(queries, top_k)
        if self.alpha == 1:
            return self.dense.search(queries, top_k)
        bm25_raw = self.bm25.search(queries, self.candidate_k)
        dense_raw = self.dense.search(queries, self.candidate_k)
        if set(bm25_raw) != set(queries) or set(dense_raw) != set(queries):
            raise ValueError("Hybrid components must return a ranking for every query.")
        merged = {}
        for qid in queries:
            bm25_scores = _min_max_normalise(bm25_raw[qid])
            dense_scores = _min_max_normalise(dense_raw[qid])
            scores = {did: self.alpha * dense_scores.get(did, 0.0)
                           + (1 - self.alpha) * bm25_scores.get(did, 0.0)
                      for did in bm25_scores.keys() | dense_scores.keys()}
            merged[qid] = dict(ranked_items(scores)[:top_k])
        return merged
