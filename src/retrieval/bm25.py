"""Vietnamese BM25 retrieval with reproducible index persistence.

``fit`` computes corpus statistics; it does not learn question-answer pairs.
Only passage text is indexed. Queries use exactly the same tokenizer.
"""

from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
import math
from numbers import Real
import pickle
from pathlib import Path

from loguru import logger
import numpy as np
from rank_bm25 import BM25Okapi
from underthesea import word_tokenize

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX_PATH = ROOT / "data" / "processed" / "bm25_index.pkl"
CACHE_SCHEMA_VERSION = 1
TOKENIZER_SETTINGS = {
    "name": "underthesea.word_tokenize", "format": "text", "lowercase": True,
    "remove_punctuation": False, "remove_stopwords": False,
}


class CacheMismatchError(ValueError):
    """The cache is readable, but must be rebuilt for the current inputs."""


class CacheReadError(ValueError):
    """The cache is unreadable or structurally inconsistent."""


def dependency_versions() -> dict[str, str]:
    """Record the actual runtime versions, without changing dependencies."""
    return {name: version(name) for name in ("rank-bm25", "underthesea", "underthesea_core", "numpy")}


def corpus_fingerprint(corpus: list[dict]) -> str:
    """Hash ordered ID/text pairs: content and row order both matter."""
    digest = hashlib.sha256()
    for doc in corpus:
        pair = json.dumps([doc["doc_id"], doc["text"]], ensure_ascii=False)
        digest.update((pair + "\n").encode("utf-8"))
    return digest.hexdigest()


def validate_corpus(corpus: list[dict]) -> None:
    """Check records before computing a fingerprint or fitting an index."""
    if not isinstance(corpus, list) or not corpus:
        raise ValueError("corpus must be a non-empty list of document dicts.")
    ids = set()
    for row, doc in enumerate(corpus):
        if not isinstance(doc, dict):
            raise ValueError(f"Corpus row {row} must be a document dict.")
        doc_id, text = doc.get("doc_id"), doc.get("text")
        if not isinstance(doc_id, str) or not doc_id.strip() or doc_id in ids:
            raise ValueError(f"Corpus row {row} needs a unique nonblank string doc_id.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Document {doc_id} needs nonblank string text.")
        ids.add(doc_id)


def _tokenize(text: str) -> list[str]:
    return word_tokenize(text.lower(), format="text").split()


class BM25Retriever:
    """Return original passage dictionaries plus their BM25 scores."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        for name, value in (("k1", k1), ("b", b)):
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number.")
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("Require k1 > 0 and 0 <= b <= 1.")
        self.k1 = float(k1)
        self.b = float(b)
        self._index: BM25Okapi | None = None
        self._corpus: list[dict] = []
        self._doc_ids: list[str] = []
        self._metadata: dict = {}

    @property
    def configuration(self) -> dict:
        return {"k1": self.k1, "b": self.b, "tokenizer": deepcopy(TOKENIZER_SETTINGS)}

    @property
    def metadata(self) -> dict:
        return deepcopy(self._metadata)

    def fit(self, corpus: list[dict]) -> "BM25Retriever":
        validate_corpus(corpus)
        snapshot = deepcopy(corpus)
        logger.info(f"Tokenizing {len(snapshot):,} passages ...")
        tokens = [_tokenize(doc["text"]) for doc in snapshot]
        if not any(tokens):
            raise ValueError("Corpus has no tokenized vocabulary.")
        index = BM25Okapi(tokens, k1=self.k1, b=self.b)
        self._index = index
        self._corpus = snapshot
        self._doc_ids = [doc["doc_id"] for doc in snapshot]
        self._metadata = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "corpus_fingerprint": corpus_fingerprint(snapshot),
            "configuration": self.configuration,
            "dependency_versions": dependency_versions(),
        }
        logger.success(f"BM25 built: {len(snapshot):,} docs | k1={self.k1} | b={self.b}")
        return self

    def save(self, path: Path | str = DEFAULT_INDEX_PATH) -> None:
        if self._index is None:
            raise RuntimeError("Cannot save: call fit() before save().")
        if self.configuration != self._metadata["configuration"]:
            raise ValueError("Parameters changed after fit(); rebuild before saving.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "index": self._index, "corpus": self._corpus,
            "doc_ids": self._doc_ids, "metadata": self._metadata,
            "k1": self.k1, "b": self.b,
        }
        with path.open("wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        logger.success(f"BM25 index saved -> {path}")

    @classmethod
    def load(
        cls, path: Path | str = DEFAULT_INDEX_PATH, *,
        expected_corpus_fingerprint: str | None = None,
        expected_config: dict | None = None,
    ) -> "BM25Retriever":
        """Load a locally created pickle; optionally check current corpus/config."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No BM25 index found at: {path}")
        try:
            with path.open("rb") as stream:
                payload = pickle.load(stream)
        except Exception as exc:
            raise CacheReadError(f"Cannot read BM25 cache {path}; rerun with --rebuild.") from exc
        if not isinstance(payload, dict):
            raise CacheReadError(f"Invalid BM25 cache {path}; rerun with --rebuild.")
        metadata = payload.get("metadata")
        if metadata is None:
            raise CacheMismatchError("Legacy cache has no metadata; rebuild required.")
        try:
            if metadata["schema_version"] != CACHE_SCHEMA_VERSION:
                raise CacheMismatchError("Cache schema changed; rebuild required.")
            if metadata["dependency_versions"] != dependency_versions():
                raise CacheMismatchError("Dependency/tokenizer versions changed; rebuild required.")
            config = metadata["configuration"]
            if config["tokenizer"] != TOKENIZER_SETTINGS:
                raise CacheMismatchError("Tokenizer settings changed; rebuild required.")
            if expected_config is not None and config != expected_config:
                raise CacheMismatchError("BM25 configuration changed; rebuild required.")
            if (expected_corpus_fingerprint is not None
                    and metadata["corpus_fingerprint"] != expected_corpus_fingerprint):
                raise CacheMismatchError("Corpus content/order changed; rebuild required.")
            retriever = cls(k1=payload["k1"], b=payload["b"])
            index, corpus, doc_ids = payload["index"], payload["corpus"], payload["doc_ids"]
            valid = (
                isinstance(index, BM25Okapi) and isinstance(corpus, list)
                and isinstance(doc_ids, list) and bool(corpus)
                and len(corpus) == len(doc_ids) == index.corpus_size
                and len(index.doc_len) == len(index.doc_freqs) == len(corpus)
                and all(isinstance(d, str) and d.strip() for d in doc_ids)
                and len(set(doc_ids)) == len(doc_ids)
                and doc_ids == [doc["doc_id"] for doc in corpus]
                and all(isinstance(doc["text"], str) and doc["text"].strip() for doc in corpus)
                and corpus_fingerprint(corpus) == metadata["corpus_fingerprint"]
                and retriever.configuration == config
                and index.k1 == retriever.k1 and index.b == retriever.b
            )
            if not valid:
                raise ValueError("Index, metadata, and corpus are inconsistent.")
        except CacheMismatchError:
            raise
        except Exception as exc:
            raise CacheReadError(f"Corrupted BM25 cache {path}; rerun with --rebuild.") from exc
        retriever._index, retriever._corpus, retriever._doc_ids = index, corpus, doc_ids
        retriever._metadata = metadata
        logger.success(f"BM25 index loaded: {len(corpus):,} docs")
        return retriever

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        if self._index is None:
            raise RuntimeError("Call fit() or load() before retrieval.")
        if not isinstance(query, str):
            raise TypeError("query must be a string.")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._index.get_scores(tokens)
        # lexsort's last key is primary; IDs provide reproducible score ties.
        indices = np.lexsort((np.asarray(self._doc_ids), -scores))[:top_k]
        return [dict(deepcopy(self._corpus[i]), score=float(scores[i])) for i in indices]
