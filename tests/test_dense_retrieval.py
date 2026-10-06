import json
import numpy as np
import pytest
import yaml
import faiss
from src.retrieval.dense.BGEM3Retriever import BGEM3Retriever
from src.retrieval.dense.dense_retriever import DenseCacheMismatchError, DenseCacheReadError


class FakeEncoder:
    max_seq_length = 8
    def get_sentence_embedding_dimension(self): return 2
    def eval(self): return self
    def half(self): return self
    def encode(self, texts, **kwargs):
        return np.asarray([{"A": [3, 0], "B": [0, 4], "C": [3, 0]}.get(t, [1, 1]) for t in texts], dtype=np.float32)
    def tokenizer(self, texts, **kwargs):
        assert kwargs == {"add_special_tokens": True, "padding": False, "truncation": False}
        return {"input_ids": [[0] * (len(t.split()) + 2) for t in texts]}


@pytest.fixture
def retriever(tmp_path, monkeypatch):
    from src.evaluation.benchmark import write_jsonl
    write_jsonl(tmp_path / "corpus.jsonl", [{"doc_id": "c", "text": "C"}, {"doc_id": "a", "text": "A"}, {"doc_id": "b", "text": "B"}])
    config = {"models": {"bge_m3": {"hf_name": "fake", "dim": 2, "max_seq_length": 4}},
              "defaults": {"batch_size": 8, "top_k": 100, "device": "cpu", "fp16": True},
              "paths": {"corpus": "corpus.jsonl", "data_dir": ".", "index_root": "indexes", "results_dir": "results"}}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    r = BGEM3Retriever(config_path, model=FakeEncoder(), resolved_revision="test-revision")
    monkeypatch.setattr(r, "dependency_versions", lambda: {"fake": "1"})
    return r


def test_batched_search_cosine_roundtrip_and_mutation(retriever):
    r = retriever
    r.build_index()
    results = r.retrieve_many(["A", "B", "", "A"], 10)
    assert [d["doc_id"] for d in results[0]] == ["a", "c", "b"]
    assert results[0] == r.retrieve("A", 10)
    assert results[1] == r.retrieve("B", 10)
    assert results[0][0]["score"] == pytest.approx(1)
    assert results[2] == []
    results[0][0]["text"] = "mutated"
    assert r.retrieve("A")[0]["text"] == "A"
    r.load_index()
    assert r.retrieve("A")[0]["doc_id"] == "a"


def test_audit_and_invalid_embeddings(retriever, monkeypatch):
    r = retriever
    audit = r.audit_lengths(["A", "one two three"])
    assert audit["truncated_count"] == 1 and audit["max_tokens"] == 5
    for bad in ([[0, 0]], [[float("nan"), 1]], [[float("inf"), 1]], [[1, 2, 3]]):
        monkeypatch.setattr(r.model, "encode", lambda *a, bad=bad, **k: np.array(bad))
        with pytest.raises(ValueError): r._encode(["A"])


@pytest.mark.parametrize("setting", ["revision", "max_seq_length", "preprocessing", "precision", "dependencies", "corpus", "corpus_order"])
def test_cache_mismatch(retriever, setting, monkeypatch):
    r = retriever
    r.build_index()
    if setting == "revision": r.resolved_revision = "different"
    elif setting == "max_seq_length": r.max_seq_length += 1
    elif setting == "preprocessing": r.PREPROCESSING = "different"
    elif setting == "precision": r.fp16 = True
    elif setting == "dependencies": monkeypatch.setattr(r, "dependency_versions", lambda: {"fake": "2"})
    else:
        records = [json.loads(line) for line in r.corpus_path.read_text().splitlines()]
        if setting == "corpus_order": records.reverse()
        else: records[0]["text"] = "Changed"
        r.corpus_path.write_text("\n".join(json.dumps(row) for row in records))
    with pytest.raises(DenseCacheMismatchError): r.load_index()


@pytest.mark.parametrize("damage", ["missing", "checksum", "dimension", "count", "snapshot", "metadata"])
def test_corrupt_cache_requires_rebuild(retriever, damage):
    r = retriever
    r.build_index()
    meta_path = r.index_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    if damage == "missing": (r.index_dir / "index.faiss").unlink()
    elif damage == "checksum": (r.index_dir / "index.faiss").write_bytes(b"broken")
    elif damage == "dimension":
        index = faiss.IndexFlatIP(3)
        index.add(np.ones((3, 3), dtype=np.float32))
        faiss.write_index(index, str(r.index_dir / "index.faiss"))
    elif damage == "count": meta["n_docs"] += 1; meta_path.write_text(json.dumps(meta))
    elif damage == "snapshot": (r.index_dir / "corpus_snapshot.jsonl").write_text('{}\n')
    else: meta_path.write_text('not json')
    with pytest.raises(DenseCacheReadError): r.load_index()


def test_cuda_oom_retries_and_terminal_failure(retriever, monkeypatch):
    import torch
    r = retriever
    r.device = "cuda"
    attempted = []
    def encode(texts, batch_size, **kwargs):
        attempted.append(batch_size)
        if batch_size > 2: raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return np.array([[1., 0.] for _ in texts], dtype=np.float32)
    monkeypatch.setattr(r.model, "encode", encode)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    r._encode(["A"])
    assert attempted == [8, 4, 2] and r.batch_size == 2
    assert [event["to_batch"] for event in r.oom_retries] == [4, 2]
    monkeypatch.setattr(r.model, "encode", lambda *a, **k: (_ for _ in ()).throw(torch.cuda.OutOfMemoryError("CUDA out of memory")))
    with pytest.raises(RuntimeError, match="batch size 1"): r._encode(["A"])
    assert r.batch_size == 1


def test_strict_cuda_and_input_validation(retriever, monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        BGEM3Retriever(retriever.corpus_path.parent / "config.yaml", device="cuda", require_cuda=True,
                       model=FakeEncoder(), resolved_revision="fake")
    retriever.build_index()
    with pytest.raises(TypeError): retriever.retrieve(None)
    with pytest.raises(ValueError): retriever.retrieve("A", 0)
    assert retriever._encode([]).shape == (0, 2)
