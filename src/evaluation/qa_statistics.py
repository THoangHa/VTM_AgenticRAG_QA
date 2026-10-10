"""Analysis helpers for retrieval slices and manual-judge outcomes."""

from collections import defaultdict
from scipy.stats import binomtest
from scipy.stats import spearmanr

from src.evaluation.comparison import paired_bootstrap


def retrieval_hit_slices(qids, gold_doc_ids, rankings, predictions, references, k=5):
    if k <= 0:
        raise ValueError("k must be positive.")
    groups = defaultdict(list)
    for qid in qids:
        hit = bool(set(gold_doc_ids[qid]) & {row["doc_id"] for row in rankings.get(qid, [])[:k]})
        groups["hit" if hit else "miss"].append(qid)
    from src.evaluation.qa_evaluator import evaluate_qa
    return {name: {"question_count": len(ids), **evaluate_qa(
        {qid: predictions[qid] for qid in ids}, {qid: references[qid] for qid in ids})["aggregate"]}
        for name, ids in groups.items()}


def paired_metric_intervals(per_question_by_arm, arm_a, arm_b, resamples=10000, seed=42):
    a, b = per_question_by_arm[arm_a], per_question_by_arm[arm_b]
    if set(a) != set(b) or not a:
        raise ValueError("Paired arms must contain the same nonempty question set.")
    output = {}
    for metric in a[next(iter(a))]:
        output[metric] = paired_bootstrap([a[q][metric] for q in sorted(a)],
                                          [b[q][metric] for q in sorted(b)], resamples, seed)
    return output


def mcnemar_correctness(labels_a, labels_b):
    if set(labels_a) != set(labels_b) or not labels_a:
        raise ValueError("McNemar inputs must contain matching nonempty question IDs.")
    correct_a = {qid: labels_a[qid]["correctness"] == "correct" for qid in labels_a}
    correct_b = {qid: labels_b[qid]["correctness"] == "correct" for qid in labels_b}
    b = sum(correct_a[q] and not correct_b[q] for q in correct_a)
    c = sum(correct_b[q] and not correct_a[q] for q in correct_a)
    total = b + c
    p = 1.0 if total == 0 else float(binomtest(min(b, c), total, 0.5, alternative="two-sided").pvalue)
    return {"a_correct_b_incorrect": b, "b_correct_a_incorrect": c, "discordant_pairs": total,
            "p_value_exact": p, "method": "exact_binomial_mcnemar"}


def metric_judge_alignment(per_question, judgements):
    if not per_question or not judgements or not set(judgements) <= set(per_question):
        raise ValueError("Judge question IDs must be a nonempty subset of the metric results.")
    correctness = {"incorrect": 0, "partial": 1, "correct": 2}
    qids = sorted(judgements)
    judged = [correctness[judgements[q]["correctness"]] for q in qids]
    output = {}
    for metric in ("f1_token", "rouge_l"):
        scores = [per_question[q][metric] for q in qids]
        coefficient, p_value = spearmanr(scores, judged)
        output[metric] = {"spearman_rho": None if coefficient != coefficient else float(coefficient),
                          "p_value": None if p_value != p_value else float(p_value),
                          "question_count": len(scores),
                          "judge_scale": "incorrect=0, partial=1, correct=2"}
    return output


def cohen_kappa_correctness(labels_a, labels_b):
    if set(labels_a) != set(labels_b) or not labels_a:
        raise ValueError("Agreement labels must contain matching nonempty question IDs.")
    categories = ("correct", "partial", "incorrect")
    n = len(labels_a)
    observed = sum(labels_a[q]["correctness"] == labels_b[q]["correctness"] for q in labels_a) / n
    expected = sum((sum(labels_a[q]["correctness"] == category for q in labels_a) / n)
                   * (sum(labels_b[q]["correctness"] == category for q in labels_b) / n)
                   for category in categories)
    kappa = 1.0 if expected == 1.0 and observed == 1.0 else (observed - expected) / (1 - expected)
    return {"cohen_kappa": float(kappa), "observed_agreement": observed, "question_count": n,
            "labels": list(categories)}
