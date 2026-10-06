#!/usr/bin/env python
"""
Evaluate the BM25 retriever on the ViMedAQA val/test splits.

Usage
-----
    # Run on the validation set (default; use this while choosing settings)
    python scripts/evaluate_bm25.py

    # Final test evaluation after settings are fixed
    python scripts/evaluate_bm25.py --split test --k1 1.5 --b 0.75

    # Force rebuild the index even if a cache file exists
    python scripts/evaluate_bm25.py --rebuild

    # Show retrieved docs for the first 5 queries (debug)
    python scripts/evaluate_bm25.py --show 5

Prerequisites
-------------
    data/processed/corpus.jsonl   — built by src/data/data_processing.py
    data/processed/qrels_val.json
    data/processed/qrels_test.json
    data/processed/val_qas.jsonl and test_qas.jsonl

The fixed corpus includes eligible train/validation/test source passages.
Questions and answers are not indexed. Corpus/config/version changes trigger
an automatic rebuild. Corrupted caches require an explicit --rebuild.
Paths are anchored to the repository, independent of the working directory.
Ranked IDs/scores and metric/config summaries are saved under results/.
"""

import argparse
import json
import sys
from pathlib import Path

import jsonlines
from loguru import logger
from tqdm import tqdm

# ── Make sure the repo root is importable ─────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.retrieval.bm25 import (
    BM25Retriever, DEFAULT_INDEX_PATH, CacheMismatchError,
    corpus_fingerprint, validate_corpus,
)
from src.evaluation.metrics import recall_at_k, precision_at_k, mrr

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR   = ROOT / "data" / "processed"
CORPUS_PATH     = PROCESSED_DIR / "corpus.jsonl"
QRELS_VAL_PATH  = PROCESSED_DIR / "qrels_val.json"
QRELS_TEST_PATH = PROCESSED_DIR / "qrels_test.json"
RESULTS_DIR = ROOT / "results"

# ── Evaluation config ─────────────────────────────────────────────────────────
K_VALUES = [1, 5, 10]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_corpus(path: Path) -> list[dict]:
    with jsonlines.open(path) as reader:
        return list(reader)


def _load_qrels(path: Path) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_questions(split: str) -> list[dict]:
    """Load val_qas.jsonl or test_qas.jsonl."""
    qa_path = PROCESSED_DIR / f"{split}_qas.jsonl"
    if not qa_path.exists():
        raise FileNotFoundError(f"QA file not found: {qa_path}")
    with jsonlines.open(qa_path) as reader:
        return list(reader)


def _validate_eval_data(questions: list[dict], qrels: dict[str, str], corpus: list[dict], split: str) -> None:
    if not questions or not isinstance(qrels, dict) or not qrels:
        raise ValueError(f"{split}: questions and qrels must be nonempty.")
    ids = set()
    for row, qa in enumerate(questions):
        if not isinstance(qa, dict):
            raise ValueError(f"{split}: question row {row} must be a dict.")
        qid = qa.get("question_idx")
        if not isinstance(qid, str) or not qid.strip() or qid in ids:
            raise ValueError(f"{split}: invalid/duplicate question ID in row {row}: {qid!r}.")
        if not isinstance(qa.get("question"), str) or not qa["question"].strip():
            raise ValueError(f"{split}: question {qid} needs nonblank string question text.")
        ids.add(qid)
    if ids != set(qrels):
        raise ValueError(
            f"{split}: question/qrels IDs differ; missing labels={sorted(ids - set(qrels))}, "
            f"extra labels={sorted(set(qrels) - ids)}."
        )
    corpus_ids = {doc["doc_id"] for doc in corpus}
    missing = [qid for qid, gold in qrels.items() if not isinstance(gold, str) or gold not in corpus_ids]
    if missing:
        raise ValueError(f"{split}: gold passages missing/invalid for question IDs: {', '.join(missing)}.")
    logger.success(f"{split}: gold coverage 100% ({len(qrels):,}/{len(qrels):,})")


def _load_or_build(corpus: list[dict], k1: float, b: float, rebuild: bool) -> BM25Retriever:
    retriever = BM25Retriever(k1=k1, b=b)
    if DEFAULT_INDEX_PATH.exists() and not rebuild:
        try:
            return BM25Retriever.load(
                DEFAULT_INDEX_PATH, expected_corpus_fingerprint=corpus_fingerprint(corpus),
                expected_config=retriever.configuration,
            )
        except CacheMismatchError as exc:
            logger.warning(f"Rebuilding BM25 cache: {exc}")
    retriever.fit(corpus)
    retriever.save(DEFAULT_INDEX_PATH)
    return retriever


def _save_results(run: dict[str, list[dict]], metrics: dict[str, float], retriever: BM25Retriever,
                  split: str, top_k: int) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS_DIR / f"bm25_{split}_results.jsonl"
    with jsonlines.open(result_path, mode="w") as writer:
        for qid, results in run.items():
            writer.write({"question_idx": qid, "results": [
                {"doc_id": doc["doc_id"], "score": doc["score"]} for doc in results
            ]})
    summary = {
        "split": split, "top_k": top_k, "query_count": len(run), "metrics": metrics,
        **retriever.metadata,
    }
    summary_path = RESULTS_DIR / f"bm25_{split}_metrics.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
    logger.success(f"Saved results -> {result_path}; summary -> {summary_path}")


def _build_run(
    retriever: BM25Retriever,
    questions: list[dict],
    top_k: int,
) -> dict[str, list[dict]]:
    """
    Run retrieval for every question; return a run dict.

    Returns
    -------
    {question_idx: [{"doc_id": ..., "score": ...}, ...]}
    """
    run: dict[str, list[dict]] = {}
    for qa in tqdm(questions, desc="Retrieving", unit="q"):
        qid     = qa["question_idx"]
        results = retriever.retrieve(qa["question"], top_k=top_k)
        run[qid] = results
    return run


def _print_table(metrics: dict[str, float]) -> None:
    """Pretty-print the metric table."""
    border = "─" * 40
    logger.info(border)
    logger.info(f"  {'Metric':<20}  {'Value':>8}")
    logger.info(border)
    for name, value in metrics.items():
        logger.info(f"  {name:<20}  {value:>8.4f}")
    logger.info(border)


def _show_examples(
    retriever: BM25Retriever,
    questions: list[dict],
    qrels: dict[str, str],
    n: int,
    top_k: int,
) -> None:
    """Print the top-k retrieved docs for the first n queries."""
    for qa in questions[:n]:
        qid      = qa["question_idx"]
        gold_id  = qrels.get(qid, "?")
        results  = retriever.retrieve(qa["question"], top_k=top_k)

        print(f"\n{'='*70}")
        print(f"Q [{qid}]: {qa['question']}")
        print(f"Gold doc_id : {gold_id}")
        print(f"{'─'*70}")
        for rank, r in enumerate(results, start=1):
            hit_marker = " ✓" if r["doc_id"] == gold_id else ""
            print(f"  #{rank:>2}  [{r['doc_id']}]  score={r['score']:.3f}{hit_marker}")
            print(f"       {r['text'][:120].replace(chr(10), ' ')} …")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate BM25 retriever on ViMedAQA val / test splits."
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="Which split to evaluate on (default: val; reserve test for final reporting).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Maximum number of docs to retrieve per query (default: 10).",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild the BM25 index even if a cache file exists.",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=0,
        metavar="N",
        help="Print top retrieved docs for the first N queries (0 = disabled).",
    )
    parser.add_argument("--k1", type=float, default=1.5, help="Term-frequency saturation (default: 1.5).")
    parser.add_argument("--b", type=float, default=0.75, help="Length normalization (default: 0.75).")
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k must be a positive integer.")
    if args.show < 0:
        parser.error("--show must be nonnegative.")
    try:
        BM25Retriever(k1=args.k1, b=args.b)
    except ValueError as exc:
        parser.error(str(exc))

    # Validate current inputs even when an index already exists.
    if not CORPUS_PATH.exists():
        parser.error(f"Corpus not found: {CORPUS_PATH}. Run python -m src.data.data_processing first.")
    corpus = _load_corpus(CORPUS_PATH)
    qrels_path = QRELS_VAL_PATH if args.split == "val" else QRELS_TEST_PATH
    if not qrels_path.exists():
        logger.error(
            f"Qrels file not found: {qrels_path}. "
            "Run python -m src.data.data_processing first."
        )
        sys.exit(1)

    qrels     = _load_qrels(qrels_path)
    questions = _load_questions(args.split)
    try:
        validate_corpus(corpus)
        _validate_eval_data(questions, qrels, corpus, args.split)
        retriever = _load_or_build(corpus, args.k1, args.b, args.rebuild)
    except ValueError as exc:
        parser.error(str(exc))

    logger.info(
        f"Evaluating on '{args.split}' split — "
        f"{len(questions):,} questions | {len(qrels):,} qrels"
    )

    # ── 3. Optional: show example retrievals ──────────────────────────────────
    if args.show > 0:
        _show_examples(retriever, questions, qrels, n=args.show, top_k=args.top_k)

    # ── 4. Build run ──────────────────────────────────────────────────────────
    run = _build_run(retriever, questions, top_k=args.top_k)

    # ── 5. Compute & print metrics ────────────────────────────────────────────
    metrics: dict[str, float] = {}
    for k in K_VALUES:
        if k <= args.top_k:
            metrics[f"Recall@{k}"]    = recall_at_k(run, qrels, k)
            metrics[f"Precision@{k}"] = precision_at_k(run, qrels, k)
    metrics[f"MRR@{args.top_k}"] = mrr(run, qrels)

    logger.info(f"\nBM25 Evaluation Results  [{args.split} split]")
    _print_table(metrics)
    _save_results(run, metrics, retriever, args.split, args.top_k)


if __name__ == "__main__":
    main()
