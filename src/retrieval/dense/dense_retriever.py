"""Dense encoding, exact search, and validated model-specific index caches."""
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
from importlib.metadata import version
import json
import os
import re
from pathlib import Path
import tempfile
import time

import faiss
import numpy as np
import yaml
from loguru import logger
from src.evaluation.benchmark_data import ROOT, corpus_fingerprint, read_jsonl, validate_corpus

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"
CACHE_SCHEMA = 2


class DenseCacheMismatchError(ValueError):
    """Readable index belongs to different inputs or encoding settings."""


class DenseCacheReadError(ValueError):
    """Corrupt/incomplete index; explicit --rebuild is required."""


def load_config(path=None):
    with Path(path or DEFAULT_CONFIG_PATH).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class BaseDenseRetriever(ABC):
    MODEL_KEY = None
    PREPROCESSING = "raw_text_v1"

    def __init__(self, config_path=DEFAULT_CONFIG_PATH, device=None, batch_size=None,
                 require_cuda=False, model=None, resolved_revision=None):
        """Injected encoder/revision permit offline tests without model downloads."""
        import torch
        config_path = Path(config_path or DEFAULT_CONFIG_PATH).resolve()
        self.config = load_config(config_path)
        models = self.config.get("models", {})
        if self.MODEL_KEY not in models:
            raise ValueError(f"Unknown model {self.MODEL_KEY!r}; configured keys: {list(models)}")
        self.spec = models[self.MODEL_KEY]
        defaults = self.config["defaults"]
        self.hf_name, self.dim = self.spec["hf_name"], self.spec["dim"]
        self.max_seq_length = self.spec["max_seq_length"]
        self.batch_size = batch_size if batch_size is not None else self.spec.get("batch_size", defaults["batch_size"])
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.initial_batch_size = self.batch_size
        self.top_k, self.device = defaults["top_k"], device or defaults["device"]
        if self.device == "cuda" and not torch.cuda.is_available():
            if require_cuda:
                raise RuntimeError("CUDA is required. Install CUDA-enabled PyTorch.")
            logger.warning("CUDA unavailable; using CPU for this non-strict retriever")
            self.device = "cpu"
        self.fp16 = bool(defaults["fp16"] and self.device == "cuda")
        self.oom_retries = []
        for attr, key in (("corpus_path", "corpus"), ("data_dir", "data_dir"),
                          ("index_root", "index_root"), ("results_root", "results_dir")):
            setattr(self, attr, (config_path.parent / self.config["paths"][key]).resolve())
        self.index_dir = self.index_root / f"dense_{self.MODEL_KEY}"
        self.results_dir = self.results_root / f"dense_{self.MODEL_KEY}"
        if model is None:
            from huggingface_hub import HfApi
            from sentence_transformers import SentenceTransformer
            revision = resolved_revision or self.spec.get("revision") or "main"
            # A frozen commit can be loaded from the local snapshot without a Hub lookup.
            self.resolved_revision = (revision if re.fullmatch(r"[0-9a-f]{40}", revision)
                                      else HfApi().model_info(self.hf_name, revision=revision).sha)
            logger.info(f"Loading {self.hf_name}@{self.resolved_revision} on {self.device}")
            self.model = SentenceTransformer(self.hf_name, revision=self.resolved_revision,
                device=self.device, cache_folder=str(ROOT / ".cache" / "models"))
        else:
            if not resolved_revision:
                raise ValueError("Injected encoders require a resolved_revision.")
            self.model, self.resolved_revision = model, resolved_revision
        self.model.max_seq_length = self.max_seq_length
        if self.fp16:
            self.model.half()
        self.model.eval()
        if self.model.get_sentence_embedding_dimension() != self.dim:
            raise ValueError("Configured embedding dimension differs from encoder dimension.")
        self.index, self.corpus, self.doc_ids, self.index_metadata = None, [], [], {}

    @abstractmethod
    def _prep(self, text: str) -> str:
        raise NotImplementedError

    @property
    def configuration(self):
        return {"model": self.MODEL_KEY, "hf_name": self.hf_name, "revision": self.resolved_revision,
                "dim": self.dim, "max_seq_length": self.max_seq_length, "preprocessing": self.PREPROCESSING,
                "precision": "float16" if self.fp16 else "float32", "normalization": "l2_float32",
                "index_type": "IndexFlatIP", "score_function": "cosine", "text_fields": ["text"]}

    def dependency_versions(self):
        names = ["numpy", "faiss-cpu", "torch", "sentence-transformers", "transformers", "huggingface-hub"]
        if self.PREPROCESSING != "raw_text_v1":
            names.append("pyvi")
        return {name: version(name) for name in names}

    def _encode(self, texts, show_progress=False):
        import torch
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        if any(not isinstance(t, str) for t in texts):
            raise TypeError("Encoding inputs must be strings.")
        prepared = [self._prep(t) for t in texts]
        while True:
            try:
                embeddings = self.model.encode(prepared, batch_size=self.batch_size,
                    show_progress_bar=show_progress, convert_to_numpy=True,
                    convert_to_tensor=False, normalize_embeddings=False, device=self.device)
                break
            except RuntimeError as exc:
                is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()
                if self.device != "cuda" or not is_oom:
                    raise
                if self.batch_size == 1:
                    raise RuntimeError("CUDA out of memory at batch size 1; encoding cannot continue.") from exc
                next_batch = max(1, self.batch_size // 2)
                self.oom_retries.append({"input_count": len(texts), "from_batch": self.batch_size, "to_batch": next_batch})
                logger.warning(f"CUDA OOM: restarting encoding with batch size {next_batch}")
                self.batch_size = next_batch
            # Exit the handler to release exception frames and partial outputs.
            gc.collect()
            torch.cuda.empty_cache()
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.shape != (len(texts), self.dim) or not np.isfinite(embeddings).all():
            raise ValueError("Encoder returned invalid shape or nonfinite embeddings.")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or np.any(norms == 0):
            raise ValueError("Encoder returned zero-norm or invalid embeddings.")
        embeddings = np.ascontiguousarray(embeddings / norms, dtype=np.float32)
        return embeddings[0] if single else embeddings

    def audit_lengths(self, texts):
        """Actual tokenizer lengths, without truncation, after preprocessing."""
        texts, lengths = list(texts), []
        for start in range(0, len(texts), 128):
            encoded = self.model.tokenizer([self._prep(t) for t in texts[start:start + 128]],
                add_special_tokens=True, padding=False, truncation=False)
            lengths.extend(len(ids) for ids in encoded["input_ids"])
        array = np.asarray(lengths)
        return {"count": len(lengths), "limit": self.max_seq_length, "includes_special_tokens": True,
                "truncated_count": int((array > self.max_seq_length).sum()),
                "truncated_fraction": float((array > self.max_seq_length).mean()) if len(array) else 0.0,
                "median_tokens": float(np.median(array)) if len(array) else 0.0,
                "p95_tokens": float(np.percentile(array, 95)) if len(array) else 0.0,
                "max_tokens": int(array.max()) if len(array) else 0}

    _fingerprint = staticmethod(corpus_fingerprint)

    def _load_corpus(self):
        corpus = read_jsonl(self.corpus_path)
        validate_corpus(corpus)
        return corpus

    def build_index(self):
        started = time.perf_counter()
        corpus = self._load_corpus()
        vectors = self._encode([doc["text"] for doc in corpus], show_progress=True)
        index = faiss.IndexFlatIP(self.dim)
        index.add(vectors)
        self.index, self.corpus = index, corpus
        self.doc_ids = [doc["doc_id"] for doc in corpus]
        self._save(time.perf_counter() - started)

    def _save(self, build_seconds=None):
        self.index_dir.mkdir(parents=True, exist_ok=True)
        # Stage files; checksum checks reject a crash between replacements.
        with tempfile.TemporaryDirectory(dir=self.index_dir) as staging:
            stage = Path(staging)
            faiss.write_index(self.index, str(stage / "index.faiss"))
            with (stage / "corpus_snapshot.jsonl").open("w", encoding="utf-8") as stream:
                for doc in self.corpus:
                    stream.write(json.dumps(doc, ensure_ascii=False) + "\n")
            meta = {"schema_version": CACHE_SCHEMA, "configuration": self.configuration,
                    "dependency_versions": self.dependency_versions(), "n_docs": len(self.corpus),
                    "corpus_fingerprint": self._fingerprint(self.corpus),
                    "index_sha256": file_digest(stage / "index.faiss"),
                    "built_at": datetime.now(timezone.utc).isoformat(), "build_seconds": build_seconds,
                    "effective_batch_size": self.batch_size}
            (stage / "meta.json").write_text(json.dumps(meta, indent=2, allow_nan=False), encoding="utf-8")
            for name in ("index.faiss", "corpus_snapshot.jsonl", "meta.json"):
                os.replace(stage / name, self.index_dir / name)
        self.index_metadata = meta

    def load_index(self):
        if not self.index_dir.exists():
            raise FileNotFoundError(f"No index in {self.index_dir}")
        try:
            meta = json.loads((self.index_dir / "meta.json").read_text(encoding="utf-8"))
            if not isinstance(meta, dict):
                raise ValueError("Metadata must be a dictionary.")
        except Exception as exc:
            raise DenseCacheReadError("Unreadable index metadata; use --rebuild.") from exc
        if meta.get("schema_version") != CACHE_SCHEMA:
            raise DenseCacheMismatchError("Index schema changed; rebuild required.")
        try:
            snapshot = read_jsonl(self.index_dir / "corpus_snapshot.jsonl")
            validate_corpus(snapshot)
            index = faiss.read_index(str(self.index_dir / "index.faiss"))
            saved_dim = meta["configuration"]["dim"]
            if (type(index).__name__ != "IndexFlatIP" or index.d != saved_dim
                    or index.ntotal != len(snapshot) or len(snapshot) != meta["n_docs"]
                    or self._fingerprint(snapshot) != meta["corpus_fingerprint"]
                    or file_digest(self.index_dir / "index.faiss") != meta["index_sha256"]):
                raise ValueError("Index dimensions, counts, snapshot, or checksum are inconsistent.")
            vectors = index.reconstruct_n(0, index.ntotal)
            if (not np.isfinite(vectors).all()
                    or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)):
                raise ValueError("Index vectors must be finite and normalized.")
        except Exception as exc:
            raise DenseCacheReadError("Corrupt/incomplete dense index; use --rebuild.") from exc
        if (meta.get("configuration") != self.configuration
                or meta.get("dependency_versions") != self.dependency_versions()
                or meta["corpus_fingerprint"] != self._fingerprint(self._load_corpus())):
            raise DenseCacheMismatchError("Corpus or encoding configuration changed; rebuild required.")
        self.index, self.corpus = index, snapshot
        self.doc_ids = [doc["doc_id"] for doc in snapshot]
        self.index_metadata = meta

    def retrieve_many(self, queries, top_k=10):
        if self.index is None:
            raise RuntimeError("Index not ready: build_index() or load_index() first.")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        queries = list(queries)
        if any(not isinstance(q, str) for q in queries):
            raise TypeError("query must be a string")
        outputs = [[] for _ in queries]
        active = [i for i, q in enumerate(queries) if q.strip()]
        if not active:
            return outputs
        vectors = self._encode([queries[i] for i in active])
        ids = np.asarray(self.doc_ids)
        # Bound score buffers; exhaustive search also gives stable cutoff ties.
        for start in range(0, len(active), 32):
            scores, indices = self.index.search(vectors[start:start + 32], self.index.ntotal)
            for offset, (row_scores, row_indices) in enumerate(zip(scores, indices)):
                if not np.isfinite(row_scores).all():
                    raise ValueError("Nonfinite retrieval scores.")
                order = np.lexsort((ids[row_indices], -row_scores))[:top_k]
                outputs[active[start + offset]] = [
                    dict(deepcopy(self.corpus[row_indices[j]]), score=float(row_scores[j])) for j in order]
        return outputs

    def retrieve(self, query, top_k=10):
        return self.retrieve_many([query], top_k)[0]
