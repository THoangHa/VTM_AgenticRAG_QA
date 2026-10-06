"""Adapters isolate model lifecycle and search from the shared evaluator."""
from typing import Protocol
from src.evaluation.benchmark_data import records_to_scores


class RetrieverAdapter(Protocol):
    def search(self, queries: dict[str, str], top_k: int) -> dict[str, dict[str, float]]: ...


class BM25Adapter:
    def __init__(self, data, index_root, rebuild=False, k1=1.5, b=0.75):
        from src.retrieval.bm25 import BM25Retriever, CacheMismatchError, corpus_fingerprint
        path = index_root / "bm25" / "index.pkl"
        self.retriever = BM25Retriever(k1=k1, b=b)
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
            self.retriever.fit(data.corpus).save(path)

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
