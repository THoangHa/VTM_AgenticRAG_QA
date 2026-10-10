import pytest

from src.evaluation.qa_statistics import (metric_judge_alignment, mcnemar_correctness,
    cohen_kappa_correctness, paired_metric_intervals, retrieval_hit_slices)


def test_paired_bootstrap_and_hit_miss_slices():
    result = paired_metric_intervals({"a": {"q": {"f1": 1.0}}, "b": {"q": {"f1": 0.0}}}, "a", "b", 100, 42)
    assert result["f1"]["difference_a_minus_b"] == 1
    slices = retrieval_hit_slices(["a", "b"], {"a": ["d1"], "b": ["d2"]},
        {"a": [{"doc_id": "d1"}], "b": []}, {"a": "yes", "b": "no"},
        {"a": "yes", "b": "yes"})
    assert slices["hit"]["question_count"] == 1
    assert slices["miss"]["exact_match"] == 0


def test_exact_mcnemar_counts_only_discordant_correctness():
    result = mcnemar_correctness({"a": {"correctness": "correct"}, "b": {"correctness": "incorrect"}},
                                 {"a": {"correctness": "incorrect"}, "b": {"correctness": "correct"}})
    assert result["discordant_pairs"] == 2
    assert result["p_value_exact"] == pytest.approx(1.0)


def test_metric_judge_alignment_reports_correlations():
    result = metric_judge_alignment({"q": {"f1_token": 0.5, "rouge_l": 0.6}},
                                    {"q": {"correctness": "partial"}})
    assert result["f1_token"]["question_count"] == 1


def test_cohen_kappa_for_manual_judge_validation_subset():
    labels = {"q1": {"correctness": "correct"}, "q2": {"correctness": "partial"}}
    assert cohen_kappa_correctness(labels, labels)["cohen_kappa"] == 1
