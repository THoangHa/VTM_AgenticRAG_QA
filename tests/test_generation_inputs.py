import json
import subprocess
import sys

import pytest

from src.evaluation.benchmark_data import load_data
from src.generation.input_loader import load_generation_data


def make_data(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "corpus.jsonl").write_text('{"doc_id":"d1","text":"gold context"}\n{"doc_id":"d2","text":"other"}\n', encoding="utf-8")
    (tmp_path / "val_qas.jsonl").write_text(
        '{"question_idx":"q1","question":"question","answer":"secret answer","context":"gold context"}\n'
        '{"question_idx":"q2","question":"question two","answer":"another answer","context":"other"}\n', encoding="utf-8")
    (tmp_path / "qrels_val.json").write_text('{"q1":"d1","q2":"d2"}', encoding="utf-8")
    return tmp_path


def make_rankings(data_dir, path, rows=None, model="bm25"):
    benchmark = load_data(data_dir, "val")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in (rows if rows is not None else [
        {"question_idx": "q1", "results": [{"doc_id": "d2", "score": 0.9}]},
        {"question_idx": "q2", "results": [{"doc_id": "d1", "score": 0.8}]}])), encoding="utf-8")
    (path.parent / "metrics.json").write_text(json.dumps({"model": model, "split": "val",
        "fingerprints": benchmark.fingerprints}), encoding="utf-8")
    (path.parent / "status.json").write_text('{"status":"complete"}', encoding="utf-8")
    return path


def test_oracle_and_normal_requests_are_separated(tmp_path):
    data_dir = make_data(tmp_path / "data")
    ranking = make_rankings(data_dir, tmp_path / "run" / "rankings.jsonl")
    rag = load_generation_data(data_dir, "val", "rag_bm25", ranking)
    assert [p.doc_id for p in rag.request("q1", "rag_bm25").passages] == ["d2"]
    assert "secret answer" not in [p.text for p in rag.request("q1", "rag_bm25").passages]
    oracle = load_generation_data(data_dir, "val", "rag_oracle")
    assert oracle.request("q1", "rag_oracle").passages[0].doc_id == "d1"
    assert oracle.references["q1"] == "secret answer"


@pytest.mark.parametrize("rows", [
    [{"question_idx": "q1", "results": [{"doc_id": "missing", "score": 0.2}]}],
    [{"question_idx": "q1", "results": [{"doc_id": "d1", "score": float("nan")}]}],
    [{"question_idx": "q1", "results": []}, {"question_idx": "q1", "results": []}],
    [],
])
def test_rejects_unknown_nonfinite_duplicate_or_missing_rankings(tmp_path, rows):
    data_dir = make_data(tmp_path / "data")
    ranking = make_rankings(data_dir, tmp_path / "run" / "rankings.jsonl", rows, model="bge_m3")
    with pytest.raises(ValueError):
        load_generation_data(data_dir, "val", "rag_dense", ranking)


def test_ranking_metadata_fingerprints_and_split_are_required(tmp_path):
    data_dir = make_data(tmp_path / "data")
    ranking = make_rankings(data_dir, tmp_path / "run" / "rankings.jsonl")
    (ranking.parent / "metrics.json").write_text('{"split":"test","fingerprints":{}}', encoding="utf-8")
    with pytest.raises(ValueError, match="split"):
        load_generation_data(data_dir, "val", "rag_bm25", ranking)


def test_generation_package_import_does_not_load_torch():
    result = subprocess.run([sys.executable, "-c",
        "import sys; import src.generation; assert 'torch' not in sys.modules"],
        check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
