"""Portable BEIR-style metrics. No model, dataset loader, or GPU is required."""
from dataclasses import dataclass
import math
from numbers import Real


@dataclass
class EvaluationReport:
    metrics: dict[str, float]
    per_query: dict[str, dict[str, float]]
    query_count: int


def _identifier(value):
    return isinstance(value, str) and bool(value.strip())


def validate_cutoffs(k_values):
    values = list(k_values)
    if not values or any(isinstance(k, bool) or not isinstance(k, int) or k <= 0 for k in values):
        raise ValueError("Cutoffs must be positive integers.")
    if len(set(values)) != len(values):
        raise ValueError("Cutoffs must not contain duplicates.")
    return sorted(values)


def ranked_items(scores):
    """Higher score wins; equal scores use ascending document ID."""
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def evaluate(qrels, results, k_values=(1, 5, 10, 100)) -> EvaluationReport:
    """Macro-average over *all* labelled queries; absent rankings score zero.

    nDCG uses linear gains, and AP@k divides by all positive judgements,
    matching trec_eval. Query/document ID equality does not remove a result.
    """
    ks = validate_cutoffs(k_values)
    if not isinstance(qrels, dict) or not qrels or not isinstance(results, dict):
        raise ValueError("Nonempty qrels and dictionary results are required.")
    for qid, labels in qrels.items():
        if not _identifier(qid) or not isinstance(labels, dict) or not labels:
            raise ValueError("Each query needs a nonblank ID and relevance labels.")
        if any(not _identifier(did) or isinstance(g, bool) or not isinstance(g, int) or g < 0
               for did, g in labels.items()):
            raise ValueError(f"{qid}: grades must be nonnegative integers with valid document IDs.")
        if not any(g > 0 for g in labels.values()):
            raise ValueError(f"{qid}: at least one positive relevance label is required.")
    if set(results) - set(qrels):
        raise ValueError("Unexpected query IDs in results.")
    for qid, scores in results.items():
        if not isinstance(scores, dict):
            raise ValueError(f"{qid}: scores must be a document-to-score dictionary.")
        for did, score in scores.items():
            if (not _identifier(did) or isinstance(score, bool) or not isinstance(score, Real)
                    or not math.isfinite(score)):
                raise ValueError(f"{qid}: document IDs and scores must be valid and finite.")
    per_query = {}
    for qid, labels in qrels.items():
        ranking = ranked_items(results.get(qid, {}))[:max(ks)]
        positive_count = sum(g > 0 for g in labels.values())
        ideal = sorted(labels.values(), reverse=True)
        row = {}
        for k in ks:
            hits, ap, dcg, rr = 0, 0.0, 0.0, 0.0
            for rank, (did, _) in enumerate(ranking[:k], 1):
                grade = labels.get(did, 0)
                if grade > 0:
                    hits += 1
                    ap += hits / rank
                    dcg += grade / math.log2(rank + 1)
                    if rr == 0:
                        rr = 1 / rank
            idcg = math.fsum(g / math.log2(rank + 1) for rank, g in enumerate(ideal[:k], 1))
            row.update({f"NDCG@{k}": dcg / idcg, f"MAP@{k}": ap / positive_count,
                        f"Recall@{k}": hits / positive_count, f"Precision@{k}": hits / k,
                        f"MRR@{k}": rr})
        per_query[qid] = row
    metrics = {name: math.fsum(row[name] for row in per_query.values()) / len(qrels)
               for name in next(iter(per_query.values()))}
    return EvaluationReport(metrics, per_query, len(qrels))
