import json
import math
from pathlib import Path
import pytest
from src.evaluation.evaluator import evaluate
from src.evaluation.benchmark_data import ROOT, read_jsonl, records_to_scores


def test_hand_calculated_and_missing_queries():
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}, "q3": {"c": 1}}
    results = {"q1": {"x": 9, "a": 8}, "q2": {}}
    report = evaluate(qrels, results, [1, 2, 5])
    assert report.query_count == 3
    assert report.metrics["Recall@2"] == pytest.approx(1/3)
    assert report.metrics["Precision@5"] == pytest.approx(1/15)
    assert report.metrics["MRR@2"] == pytest.approx(1/6)
    assert report.metrics["MAP@2"] == pytest.approx(1/6)
    assert report.metrics["NDCG@2"] == pytest.approx(1 / math.log2(3) / 3)
    assert all(v == 0 for v in report.per_query["q3"].values())


def test_multiple_graded_labels_and_ap_denominator():
    report = evaluate({"q": {"a": 3, "b": 1, "c": 1}}, {"q": {"b": 3, "a": 2}}, [1, 2, 10])
    assert report.metrics["MAP@1"] == pytest.approx(1/3)
    assert report.metrics["MAP@2"] == pytest.approx(2/3)
    assert report.metrics["NDCG@2"] == pytest.approx((1 + 3/math.log2(3)) / (3 + 1/math.log2(3)))


def test_ties_and_equal_query_document_ids():
    assert evaluate({"q": {"a": 1}}, {"q": {"z": 2, "a": 2}}, [1]).metrics["Recall@1"] == 1
    assert evaluate({"q": {"q": 1}}, {"q": {"q": 2}}, [1]).metrics["Recall@1"] == 1


@pytest.mark.parametrize("qrels,results,ks", [({}, {}, [1]), ({"q": {"a": 0}}, {}, [1]),
    ({"q": {"a": -1}}, {}, [1]), ({"q": {"a": True}}, {}, [1]),
    ({"q": {"a": 1}}, {"other": {}}, [1]), ({"q": {"a": 1}}, {"q": {"a": float("nan")}}, [1]),
    ({"q": {"a": 1}}, {"q": {"a": float("inf")}}, [1]), ({"q": {"a": 1}}, {}, [0]),
    ({"q": {"a": 1}}, {}, [1, 1]), ({"q": {"a": 1}}, {}, [True])])
def test_invalid_inputs(qrels, results, ks):
    with pytest.raises(ValueError):
        evaluate(qrels, results, ks)


def test_reference_fixture():
    fixture = json.loads((Path(__file__).parent / "fixtures" / "trec_metrics.json").read_text())
    report = evaluate(fixture["qrels"], fixture["results"], fixture["k_values"])
    for name, expected in fixture["metrics"].items():
        assert report.metrics[name] == pytest.approx(expected, abs=5e-6)


def test_saved_bm25_regression():
    path = ROOT / "results" / "bm25_val_results.jsonl"
    if not path.exists():
        pytest.skip("Local BM25 rankings are not committed.")
    labels = json.loads((ROOT / "data/processed/qrels_val.json").read_text())
    run = {row["question_idx"]: records_to_scores(row["results"]) for row in read_jsonl(path)}
    expected = json.loads((ROOT / "results/bm25_val_metrics.json").read_text())["metrics"]
    actual = evaluate({qid: {gold: 1} for qid, gold in labels.items()}, run, [1, 5, 10])
    for name, value in expected.items():
        assert actual.metrics[name] == pytest.approx(value, abs=1e-12)


def test_duplicate_adapter_records():
    with pytest.raises(ValueError, match="Duplicate"):
        records_to_scores([{"doc_id": "a", "score": 2}, {"doc_id": "a", "score": 1}])
