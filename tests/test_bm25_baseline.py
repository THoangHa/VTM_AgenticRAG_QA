"""Contract tests for the shared corpus, BM25 cache, and evaluation CLI."""

from copy import deepcopy
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys

import jsonlines
import pytest

from src.data import data_processing as processing
from src.evaluation.metrics import recall_at_k, precision_at_k, mrr
from src.retrieval import bm25
from scripts import evaluate_bm25 as evaluation


def row(qid, context, question=None, topic=3):
    return {
        "question_idx": qid, "question": question or f"Question {qid}",
        "answer": "reference", "context": context, "title": "title",
        "keyword": "keyword", "article_url": "https://example.org", "topic": topic,
    }


@pytest.fixture
def dataset():
    return {
        "train": [row("train", "apple fruit"), row("shared", "banana yellow")],
        "validation": [row("val", "banana yellow")],
        "test": [row("test", "cherry red")],
    }


@pytest.fixture
def corpus():
    return [
        {"doc_id": "c", "text": "apple fruit", "metadata": {"source": "original"}},
        {"doc_id": "b", "text": "banana yellow"},
        {"doc_id": "a", "text": "cherry red"},
    ]


@pytest.fixture
def retriever(monkeypatch, corpus):
    # Controlled tokens isolate ranking/cache behavior from linguistic choices.
    monkeypatch.setattr(bm25, "_tokenize", lambda text: text.lower().split())
    return bm25.BM25Retriever().fit(corpus)


def test_fixed_corpus_and_outputs(dataset, tmp_path):
    result = processing.process(dataset, str(tmp_path))
    assert len(result["corpus"]) == 3
    ids = {doc["doc_id"] for doc in result["corpus"]}
    assert result["qrels_test"]["test"] == processing._context_hash("cherry red")
    assert result["qrels_test"]["test"] in ids
    assert result["qrels_val"]["val"] in ids
    assert all("question" not in doc and "answer" not in doc for doc in result["corpus"])
    assert {p.name for p in tmp_path.iterdir()} == {
        "corpus.jsonl", "val_qas.jsonl", "test_qas.jsonl", "qrels_val.json", "qrels_test.json", "data_quality_report.json",
    }
    assert processing._context_hash("cherry red") == processing._context_hash("cherry red")


@pytest.mark.parametrize("problem", ["empty_val", "empty_test", "empty_corpus", "blank", "duplicate_id", "missing_gold"])
def test_processing_fails_before_writing(dataset, tmp_path, problem):
    kwargs = {}
    if problem == "empty_val":
        dataset["validation"] = []
    elif problem == "empty_test":
        dataset["test"] = []
    elif problem == "empty_corpus":
        kwargs["corpus_topics"] = set()
    elif problem == "blank":
        dataset["test"][0]["context"] = "  "
    elif problem == "duplicate_id":
        dataset["test"].append(row("test", "another context"))
    else:
        kwargs["corpus_topics"] = {2}
        dataset["train"].append(row("drug", "drug passage", topic=2))
    with pytest.raises(ValueError):
        processing.process(dataset, tmp_path, **kwargs)
    assert not list(tmp_path.iterdir())


def test_hash_collision(monkeypatch):
    monkeypatch.setattr(processing, "_context_hash", lambda text: "collision")
    with pytest.raises(ValueError, match="collision"):
        processing._deduplicate_corpus([row("a", "first"), row("b", "second")])


def test_overlap_warns_and_keeps_questions(dataset, tmp_path, monkeypatch):
    dataset["test"][0]["question"] = dataset["validation"][0]["question"]
    messages = []
    monkeypatch.setattr(processing.logger, "warning", messages.append)
    result = processing.process(dataset, tmp_path)
    assert len(result["val_qas"]) == len(result["test_qas"]) == 1
    assert any("identical question" in message for message in messages)


def test_coverage_error_names_split_and_question():
    with pytest.raises(ValueError, match="val.*q-missing"):
        processing._sanity_check([{"doc_id": "a"}], {"q-missing": "b"}, "val")


@pytest.mark.parametrize("parameters", [
    {"k1": 0}, {"k1": -1}, {"k1": float("nan")}, {"k1": float("inf")},
    {"k1": True}, {"k1": "1.5"}, {"b": -0.1}, {"b": 1.1}, {"b": float("nan")},
])
def test_invalid_parameters(parameters):
    with pytest.raises(ValueError):
        bm25.BM25Retriever(**parameters)


@pytest.mark.parametrize("cutoff", [0, -1, True, False, 1.5, "2"])
def test_invalid_cutoffs(retriever, cutoff):
    with pytest.raises(ValueError, match="positive integer"):
        retriever.retrieve("apple", cutoff)


def test_retrieval_contract(retriever):
    results = retriever.retrieve("APPLE", 10)
    assert len(results) == 3
    assert results[0]["doc_id"] == "c" and results[0]["score"] > 0
    assert [d["doc_id"] for d in retriever.retrieve("unknown", 3)] == ["a", "b", "c"]
    assert all(d["score"] == 0 for d in retriever.retrieve("unknown", 3))
    assert retriever.retrieve("  ") == []
    with pytest.raises(TypeError):
        retriever.retrieve(None)


def test_unfitted():
    retriever = bm25.BM25Retriever()
    with pytest.raises(RuntimeError):
        retriever.retrieve("query")
    with pytest.raises(RuntimeError):
        retriever.save()


@pytest.mark.parametrize("corpus", [[], [{"doc_id": "a", "text": " "}],
                                      [{"doc_id": 1, "text": "text"}],
                                      [{"doc_id": "a", "text": "one"}, {"doc_id": "a", "text": "two"}]])
def test_invalid_corpus(corpus):
    with pytest.raises(ValueError):
        bm25.BM25Retriever().fit(corpus)


def test_empty_vocabulary(monkeypatch, corpus):
    monkeypatch.setattr(bm25, "_tokenize", lambda text: [])
    with pytest.raises(ValueError, match="vocabulary"):
        bm25.BM25Retriever().fit(corpus)


def test_snapshot_and_result_mutation(retriever, corpus):
    corpus.reverse()
    corpus[-1]["text"] = "changed"
    corpus[-1]["metadata"]["source"] = "changed"
    result = retriever.retrieve("apple", 1)[0]
    assert result["text"] == "apple fruit"
    assert result["metadata"]["source"] == "original"
    result["metadata"]["source"] = "mutated output"
    assert retriever.retrieve("apple", 1)[0]["metadata"]["source"] == "original"


def test_cache_roundtrip(retriever, tmp_path):
    path = tmp_path / "index.pkl"
    retriever.save(path)
    loaded = bm25.BM25Retriever.load(path)
    assert loaded.retrieve("apple") == retriever.retrieve("apple")
    assert loaded.metadata == retriever.metadata


def test_fingerprint_and_positive_ties(monkeypatch):
    monkeypatch.setattr(bm25, "_tokenize", lambda text: text.split())
    docs = [{"doc_id": "b", "text": "match"}, {"doc_id": "a", "text": "match"}]
    docs.extend({"doc_id": str(i), "text": f"unrelated{i}"} for i in range(4))
    retriever = bm25.BM25Retriever().fit(docs)
    assert [d["doc_id"] for d in retriever.retrieve("match", 2)] == ["a", "b"]
    assert retriever.retrieve("match", 1)[0]["score"] > 0
    assert bm25.corpus_fingerprint(docs) != bm25.corpus_fingerprint(list(reversed(docs)))


def test_changed_parameters_after_fit(retriever, tmp_path):
    retriever.k1 = 2.0
    with pytest.raises(ValueError, match="changed after fit"):
        retriever.save(tmp_path / "index.pkl")


def test_cache_input_changes(retriever, corpus, tmp_path):
    path = tmp_path / "index.pkl"
    retriever.save(path)
    changed = deepcopy(corpus)
    changed[0]["text"] += " change"
    for different in (changed, list(reversed(corpus))):
        with pytest.raises(bm25.CacheMismatchError, match="Corpus"):
            bm25.BM25Retriever.load(path, expected_corpus_fingerprint=bm25.corpus_fingerprint(different))
    with pytest.raises(bm25.CacheMismatchError, match="configuration"):
        bm25.BM25Retriever.load(path, expected_config=bm25.BM25Retriever(k1=2).configuration)


@pytest.mark.parametrize("change", ["version", "tokenizer", "schema", "legacy", "alignment", "length"])
def test_cache_metadata_and_alignment(retriever, tmp_path, change):
    path = tmp_path / "index.pkl"
    retriever.save(path)
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    error = bm25.CacheMismatchError
    if change == "version":
        payload["metadata"]["dependency_versions"]["underthesea"] = "old"
    elif change == "tokenizer":
        payload["metadata"]["configuration"]["tokenizer"]["lowercase"] = False
    elif change == "schema":
        payload["metadata"]["schema_version"] = -1
    elif change == "legacy":
        del payload["metadata"]
    elif change == "alignment":
        payload["doc_ids"].reverse()
        error = bm25.CacheReadError
    else:
        payload["index"].doc_len.pop()
        error = bm25.CacheReadError
    with path.open("wb") as stream:
        pickle.dump(payload, stream)
    with pytest.raises(error):
        bm25.BM25Retriever.load(path)


def test_corrupted_cache(tmp_path):
    path = tmp_path / "bad.pkl"
    path.write_bytes(b"not a pickle")
    with pytest.raises(bm25.CacheReadError, match="--rebuild"):
        bm25.BM25Retriever.load(path)


def test_evaluator_rebuilds_mismatch(retriever, corpus, tmp_path, monkeypatch):
    path = tmp_path / "index.pkl"
    monkeypatch.setattr(evaluation, "DEFAULT_INDEX_PATH", path)
    retriever.save(path)
    changed = deepcopy(corpus)
    changed[0]["text"] += " changed"
    rebuilt = evaluation._load_or_build(changed, 2.0, 0.5, False)
    assert rebuilt.metadata["corpus_fingerprint"] == bm25.corpus_fingerprint(changed)
    assert rebuilt.k1 == 2 and rebuilt.b == 0.5
    with path.open("wb") as stream:
        pickle.dump({"legacy": True}, stream)
    evaluation._load_or_build(corpus, 1.5, 0.75, False)
    path.write_bytes(b"broken")
    with pytest.raises(bm25.CacheReadError):
        evaluation._load_or_build(corpus, 1.5, 0.75, False)
    evaluation._load_or_build(corpus, 1.5, 0.75, True)


def test_hand_calculated_metrics():
    run = {
        "q1": [{"doc_id": "gold1"}],
        "q2": [{"doc_id": "other"}, {"doc_id": "gold2"}],
        "q3": [{"doc_id": "other"}, {"doc_id": "other2"}, {"doc_id": "gold3"}],
    }
    qrels = {f"q{i}": f"gold{i}" for i in range(1, 5)}
    truncated = {qid: docs[:2] for qid, docs in run.items()}
    assert recall_at_k(truncated, qrels, 1) == 0.25
    assert recall_at_k(truncated, qrels, 2) == 0.5
    assert precision_at_k(truncated, qrels, 2) == 0.25
    assert mrr(truncated, qrels) == 0.375
    with pytest.raises(ValueError):
        precision_at_k(run, qrels, 0)


def test_eval_id_mismatch_and_coverage(corpus):
    questions = [row("q1", "ignored")]
    with pytest.raises(ValueError, match="IDs differ"):
        evaluation._validate_eval_data(questions, {"wrong": "a"}, corpus, "val")
    with pytest.raises(ValueError, match="q1"):
        evaluation._validate_eval_data(questions, {"q1": "missing"}, corpus, "val")


def test_real_vietnamese_tokenizer():
    tokens = bm25._tokenize("Cam thảo có tác dụng gì trong y học cổ truyền?")
    assert "cam_thảo" in tokens and "y_học" in tokens
    assert bm25._tokenize("CAM THẢO") == bm25._tokenize("cam thảo")


def test_cli_from_other_directory(tmp_path, dataset):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = Path(__file__).resolve().parents[1]
    shutil.copytree(source / "src", repo / "src", ignore=shutil.ignore_patterns("__pycache__"))
    (repo / "scripts").mkdir()
    shutil.copy2(source / "scripts" / "evaluate_bm25.py", repo / "scripts" / "evaluate_bm25.py")
    processing.process(dataset, repo / "data" / "processed")
    command = [sys.executable, str(repo / "scripts" / "evaluate_bm25.py")]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    completed = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8")
    assert completed.returncode == 0, completed.stderr
    assert "gold coverage 100%" in completed.stderr
    summary = json.loads((repo / "results" / "bm25_val_metrics.json").read_text(encoding="utf-8"))
    assert summary["split"] == "val" and summary["query_count"] == 1
    assert "MRR@10" in summary["metrics"] and "MRR" not in summary["metrics"]
    assert summary["configuration"]["k1"] == 1.5
    with jsonlines.open(repo / "results" / "bm25_val_results.jsonl") as reader:
        results = list(reader)
    assert results[0]["question_idx"] == "val"
    assert set(results[0]["results"][0]) == {"doc_id", "score"}
    assert (repo / "data" / "processed" / "bm25_index.pkl").exists()
    assert not (tmp_path / "data").exists()
    completed = subprocess.run(command + ["--split", "test", "--k1", "2", "--b", "0.5"],
                               cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8")
    assert completed.returncode == 0, completed.stderr
    assert "Rebuilding BM25 cache" in completed.stderr
    test_summary = json.loads((repo / "results" / "bm25_test_metrics.json").read_text(encoding="utf-8"))
    assert test_summary["configuration"]["k1"] == 2.0
    bad = subprocess.run(command + ["--top-k", "0"], cwd=tmp_path, env=env,
                         capture_output=True, text=True, encoding="utf-8")
    assert bad.returncode != 0 and "positive integer" in bad.stderr


@pytest.mark.parametrize("context", ["", "  ", None])
def test_invalid_context_exclusion_is_consistent_and_audited(dataset, tmp_path, context):
    dataset["train"].append(row("bad-train", context, topic=2))
    dataset["validation"].append(row("bad-val", context))
    dataset["test"].append(row("bad-test", context))
    raw_snapshot = deepcopy(dataset)
    result = processing.process(dataset, tmp_path)
    assert dataset == raw_snapshot
    assert len(result["corpus"]) == 3
    assert [r["question_idx"] for r in result["val_qas"]] == ["val"]
    assert [r["question_idx"] for r in result["test_qas"]] == ["test"]
    assert set(result["qrels_val"]) == {"val"}
    assert set(result["qrels_test"]) == {"test"}
    report = json.loads((tmp_path / "data_quality_report.json").read_text(encoding="utf-8"))
    assert report == result["data_quality_report"]
    assert report["excluded_count"] == 3
    assert report["splits"]["validation"]["excluded_eval_questions"] == 1
    assert {r["question_idx"] for r in report["excluded_rows"]} == {"bad-train", "bad-val", "bad-test"}


def test_irrelevant_blank_context_not_counted(dataset, tmp_path):
    dataset["train"].append(row("irrelevant", "", topic=0))
    result = processing.process(dataset, tmp_path)
    assert result["data_quality_report"]["excluded_count"] == 0
