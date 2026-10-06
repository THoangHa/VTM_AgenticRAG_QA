"""
Retrieval metric utilities for the ViMedAQA RAG pipeline.

All functions are pure (no I/O, no side effects) and work on the
output format produced by any retriever's retrieve() method.

Functions
---------
recall_at_k       : fraction of questions where gold doc appears in top-k
precision_at_k    : fraction of top-k results that are relevant (single-relevant)
mrr               : Mean Reciprocal Rank over all questions
"""

from __future__ import annotations


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")


def recall_at_k(
    run: dict[str, list[dict]],
    qrels: dict[str, str],
    k: int,
) -> float:
    """
    Compute Recall@k.

    Since each question has exactly one relevant document in ViMedAQA,
    Recall@k == 1 iff the gold doc_id appears in the top-k results,
    0 otherwise.  We average this binary signal over all questions.

    Parameters
    ----------
    run   : {question_idx: [{"doc_id": ..., "score": ...}, ...]}
            Retrieval results, sorted descending by score.
    qrels : {question_idx: gold_doc_id}
            Ground-truth relevance labels from qrels_val/test.json.
    k     : int
            Cut-off rank.

    Returns
    -------
    float  (in [0, 1])
    """
    _validate_k(k)
    hits = 0
    total = 0

    for qid, gold_doc_id in qrels.items():
        if qid not in run:
            total += 1
            continue
        retrieved_ids = [r["doc_id"] for r in run[qid][:k]]
        if gold_doc_id in retrieved_ids:
            hits += 1
        total += 1

    return hits / total if total > 0 else 0.0


def precision_at_k(
    run: dict[str, list[dict]],
    qrels: dict[str, str],
    k: int,
) -> float:
    """
    Compute Precision@k.

    With a single relevant document per question, P@k == 1/k if the gold
    doc appears in the top-k, else 0.  Averaged over all questions.

    Parameters
    ----------
    run   : same format as recall_at_k
    qrels : same format as recall_at_k
    k     : cut-off rank

    Returns
    -------
    float  (in [0, 1/k])
    """
    _validate_k(k)
    total_precision = 0.0
    total = 0

    for qid, gold_doc_id in qrels.items():
        if qid not in run:
            total += 1
            continue
        retrieved_ids = [r["doc_id"] for r in run[qid][:k]]
        if gold_doc_id in retrieved_ids:
            total_precision += 1.0 / k
        total += 1

    return total_precision / total if total > 0 else 0.0


def mrr(
    run: dict[str, list[dict]],
    qrels: dict[str, str],
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).

    For each question, the reciprocal rank is 1 / rank_of_first_relevant_doc
    (or 0 if the gold doc never appears).  MRR is the mean over all questions.

    Parameters
    ----------
    run   : same format as recall_at_k
    qrels : same format as recall_at_k

    Returns
    -------
    float  (in [0, 1])
    """
    reciprocal_ranks = []

    for qid, gold_doc_id in qrels.items():
        if qid not in run:
            reciprocal_ranks.append(0.0)
            continue

        rr = 0.0
        for rank, result in enumerate(run[qid], start=1):
            if result["doc_id"] == gold_doc_id:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
