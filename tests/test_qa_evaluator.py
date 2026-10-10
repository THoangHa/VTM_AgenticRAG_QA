import pytest

from src.evaluation.qa_evaluator import evaluate_qa


def test_aggregates_and_per_question_scores():
    report = evaluate_qa({"q1": "đau đầu", "q2": ""}, {"q1": "đau đầu", "q2": "sốt"})
    assert report["query_count"] == 2
    assert report["aggregate"]["exact_match"] == 0.5
    assert report["per_question"]["q1"]["f1_token"] == 1


@pytest.mark.parametrize("predictions,references", [({}, {"q": "a"}), ({"q": "a"}, {}),
    ({"q": "a"}, {"other": "a"}), ({"q": "a"}, {"q": "  "})])
def test_rejects_misaligned_or_empty_inputs(predictions, references):
    with pytest.raises(ValueError):
        evaluate_qa(predictions, references)
