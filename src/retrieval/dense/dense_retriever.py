'''
Encode all corpus documents and test queries into embeddings
'''

import os
import argparse
import hashlib
import json
import math
import time
import statistics
import datetime
import re
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path

import yaml
import jsonlines
from loguru import logger

from sentence_transformers import SentenceTransformer
import torch
import faiss
import numpy as np

from pyvi import ViTokenizer

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config(path=None):
    '''
    Load configuration
    '''
    path = path or DEFAULT_CONFIG_PATH
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Config not found: {path}")


class BaseDenseRetriever(ABC):
    # Class attribute, match MODEL_KEY of a retriever subclass
    MODEL_KEY = None

    def __init__(self, config_path=DEFAULT_CONFIG_PATH, device=None):
        '''
        Load Configuration

        Read model settings, merge defaults, resolve paths,
        load sentence transformer, set max_seq_length, call .half()
        '''
        config_path = Path(config_path or DEFAULT_CONFIG_PATH)

        # Load config
        try:
            self.config = load_config(config_path)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to load config: {e}")

        # Validate MODEL_KEY
        models = self.config.get("models", {})
        model_cfg = models.get(self.MODEL_KEY)
        if not model_cfg:
            raise ValueError(
                f"Model config not found for {self.MODEL_KEY!r}. "
                f"Valid keys: {list(models)}"
            )

        # Model settings 
        defaults = self.config["defaults"]
        self.spec = model_cfg
        self.hf_name = self.spec["hf_name"]
        self.max_seq_length = self.spec["max_seq_length"]
        self.dim = self.spec["dim"]

        self.batch_size = self.spec.get("batch_size", defaults["batch_size"])
        self.top_k = defaults["top_k"]
        self.fp16 = defaults["fp16"]

        # Device
        requested = device or defaults["device"]
        if requested == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            requested = "cpu"
        self.device = requested

        # Resolve paths 
        paths = self.config["paths"]
        config_dir = config_path.parent
        self.corpus_path = (config_dir / paths["corpus"]).resolve()
        self.data_dir = (config_dir / paths["data_dir"]).resolve()
        self.index_root = (config_dir / paths["index_root"]).resolve()
        self.results_root = (config_dir / paths["results_dir"]).resolve()

        self.index_dir = self.index_root / f"dense_{self.MODEL_KEY}"
        self.results_dir = self.results_root / f"dense_{self.MODEL_KEY}"

        # Load model
        logger.info(f"Loading {self.hf_name} on {self.device} ...")
        self.model = SentenceTransformer(self.hf_name, device=self.device)
        self.model.max_seq_length = self.max_seq_length
        if self.fp16 and self.device == "cuda":
            self.model.half()

        # Validate embedding dimension
        actual_dim = self.model.get_sentence_embedding_dimension()
        if actual_dim != self.dim:
            raise ValueError(
                f"Dimension mismatch for {self.MODEL_KEY}: "
                f"config says {self.dim}, model gives {actual_dim}"
            )

        # Index state 
        self.index = None
        self.doc_ids = []
        self.corpus = []  # full corpus records, in index-row order

    @abstractmethod
    def _prep(self, text: str) -> str:
        '''
        Model-specific text preprocessing, applied to passages and queries.
        '''
        raise NotImplementedError

    def _encode(self, texts, show_progress=False):
        # Wrap texts into a list
        if isinstance(texts, str):
            texts = [texts]
            wrap = True
        else:
            wrap = False

        # Check empty lists
        texts = list(texts)
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)  # empty float32 array

        # Preprocess
        texts = [self._prep(t) for t in texts]

        # Batch encode with progress
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,  # must be numpy for faiss
            convert_to_tensor=False,
            normalize_embeddings=False,  # normalized manually below
            device=self.device if self.device != "cuda" else None  # None = use model.device
        )

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        # L2 normalize all embeddings
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norm = np.where(norm == 0, 1.0, norm)
        embeddings = embeddings / norm

        # Return to user
        if wrap:
            return embeddings[0]
        return embeddings

    @staticmethod
    def _fingerprint(records):
        '''
        SHA-256 over the ordered [doc_id, text] pairs. Detects changed text,
        ids or order between the index and the corpus.
        '''
        payload = json.dumps(
            [[r["doc_id"], r["text"]] for r in records], ensure_ascii=False
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _load_corpus(self):
        '''
        Read corpus.jsonl and return the full records (list of dicts).
        Validates: file exists, not empty, valid doc_id, non-blank text,
        unique doc_ids.
        '''
        if not Path(self.corpus_path).exists():
            raise FileNotFoundError(
                f"Corpus not found: {self.corpus_path}. "
                "Run `python -m src.data.data_processing` first."
            )

        with jsonlines.open(self.corpus_path, 'r') as f:
            records = list(f)

        if not records:
            raise ValueError(f"Corpus is empty: {self.corpus_path}")

        seen = set()
        for n, r in enumerate(records, 1):
            doc_id, text = r.get("doc_id"), r.get("text")
            if not isinstance(doc_id, str) or not doc_id:
                raise ValueError(f"Corpus line {n}: missing or invalid 'doc_id'")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Corpus line {n} (doc_id={doc_id}): blank or missing 'text'")
            if doc_id in seen:
                raise ValueError(f"Corpus line {n}: duplicate doc_id {doc_id}")
            seen.add(doc_id)

        return records

    def build_index(self):
        start = time.perf_counter()
        corpus = self._load_corpus()
        texts = [d['text'] for d in corpus]

        embs = self._encode(texts, show_progress=True)

        if (embs.shape == (len(corpus), self.dim)):
            logger.info("Embeddings shape matches the number of documents and dimension")
        else:
            raise ValueError("Embeddings shape does not match the number of documents and dimension")

        index = faiss.IndexFlatIP(self.dim)
        index.add(embs)

        if (index.ntotal == len(corpus)):
            logger.info("Index built successfully")
        else:
            raise ValueError("Index not built successfully")

        self.index = index
        self.corpus = corpus
        self.doc_ids = [d['doc_id'] for d in corpus]

        self._save(build_seconds=time.perf_counter() - start)

    def _save(self, build_seconds=None):
        # Create directory if not exists 
        os.makedirs(self.index_dir, exist_ok=True)

        faiss.write_index(self.index, str(self.index_dir / "index.faiss"))

        # The corpus snapshot, in index-row order, so row i in FAISS maps to record i.
        with open(self.index_dir / "corpus_snapshot.jsonl", "w", encoding="utf-8") as f:
            for record in self.corpus:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # meta.json is written last: a crash before this leaves an index that
        # load_index() refuses to load.
        meta = {
            "model": self.MODEL_KEY,
            "hf_name": self.hf_name,
            "dim": self.dim,
            "batch_size": self.batch_size,  # informational only
            "max_seq_length": self.max_seq_length,
            "fp16": self.fp16,
            "ntotal": int(self.index.ntotal),
            "n_docs": len(self.corpus),
            "index_type": type(self.index).__name__,
            "corpus_fingerprint": self._fingerprint(self.corpus),
            "built_at": datetime.datetime.now().isoformat(),
            "build_seconds": build_seconds,
        }
        with open(self.index_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info(f"Index saved to {self.index_dir}")

    def load_index(self):
        required = ["index.faiss", "corpus_snapshot.jsonl", "meta.json"]

        # Check if all required files exist
        missing = [f for f in required if not (self.index_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing index files in {self.index_dir}: {missing}. "
                "Build the index first."
            )

        # Compare metadata with current configuration
        with open(self.index_dir / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        expected = {
            "model": self.MODEL_KEY,
            "hf_name": self.hf_name,
            "dim": self.dim,
            "max_seq_length": self.max_seq_length,
        }
        for key, value in expected.items():
            if meta.get(key) != value:
                raise ValueError(
                    f"Index/config mismatch on '{key}': index has {meta.get(key)!r}, "
                    f"config has {value!r}. Rebuild the index."
                )
        # fp16 slightly changes embeddings but does not invalidate the index
        if meta.get("fp16") != self.fp16:
            logger.warning(
                f"fp16 differs: index built with {meta.get('fp16')}, config has {self.fp16}"
            )

        # Load index and corpus snapshot
        try:
            index = faiss.read_index(str(self.index_dir / "index.faiss"))
            with open(self.index_dir / "corpus_snapshot.jsonl", "r", encoding="utf-8") as f:
                snapshot = [json.loads(line) for line in f if line.strip()]
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise

        if meta.get("index_type") != type(index).__name__:
            raise ValueError(
                f"Index type mismatch: meta says {meta.get('index_type')}, "
                f"loaded {type(index).__name__}"
            )
        if not (index.ntotal == len(snapshot) == meta["ntotal"] == meta["n_docs"]):
            raise ValueError(
                f"Count mismatch: index={index.ntotal}, snapshot={len(snapshot)}, "
                f"meta={meta['ntotal']}/{meta['n_docs']}. Rebuild the index."
            )
        if self._fingerprint(snapshot) != meta["corpus_fingerprint"]:
            raise ValueError("Corpus snapshot is corrupted (fingerprint mismatch). Rebuild the index.")

        self.index = index
        self.corpus = snapshot
        self.doc_ids = [d["doc_id"] for d in snapshot]
        logger.info(f"Index loaded from {self.index_dir} ({index.ntotal:,} vectors)")

        # Warn (do not fail) if corpus.jsonl changed since the index was built
        try:
            if self._fingerprint(self._load_corpus()) != meta["corpus_fingerprint"]:
                logger.warning(
                    "corpus.jsonl has changed since this index was built; "
                    "rebuild the index to use the new corpus."
                )
        except Exception as e:
            logger.warning(f"Could not compare the index with the current corpus file: {e}")

    def retrieve(self, query, top_k=10):
        '''
        Return up to top_k corpus records (dicts) with an added "score",
        best first. Ties are broken by ascending doc_id.
        '''
        if self.index is None:
            raise RuntimeError("Index not ready: call build_index() or load_index() first")

        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        if not query.strip():
            return []

        query_emb = self._encode(query)
        query_emb = query_emb.reshape(1, -1)

        # Score the whole corpus (flat search is exhaustive anyway) so ties are
        # ordered exactly like BM25Retriever: descending score, then ascending doc_id.
        scores, idx = self.index.search(query_emb, self.index.ntotal)
        scores, idx = scores[0], idx[0]
        valid = idx != -1
        scores, idx = scores[valid], idx[valid]

        doc_ids = np.asarray(self.doc_ids)[idx]
        order = np.lexsort((doc_ids, -scores))[:top_k]

        # Independent deep copies, so callers cannot mutate the index's corpus
        return [
            dict(deepcopy(self.corpus[idx[j]]), score=float(scores[j]))
            for j in order
        ]
    
    