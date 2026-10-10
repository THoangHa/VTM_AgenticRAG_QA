import pytest

from src.evaluation.qa_metrics import exact_match, f1_token, normalize_answer, rouge_l


def test_vietnamese_normalization_keeps_accents_and_canonical_composition():
    assert normalize_answer("  Đau, ĐẦU!\n") == "đau đầu"
    assert exact_match("a\u0301", "á") == 1
    assert exact_match("thuốc", "thuoc") == 0


def test_repeated_token_f1_uses_multiset_overlap():
    assert f1_token("đau đau đầu", "đau đầu đầu") == pytest.approx(2 / 3)


def test_rouge_l_known_examples_and_empty_prediction():
    assert rouge_l("a b c", "a c") == pytest.approx(0.8)
    assert rouge_l("", "nonempty") == 0
