"""
Process raw ViMedAQA data into retrieval-ready artifacts.

Responsibilities
----------------
- Filter rows by topic to build the corpus and evaluation sets.
- Deduplicate corpus passages.
- Build ground-truth qrels (retrieval relevance labels).
- Run diagnostics (length stats, sanity checks).
- Save all artifacts to data/processed/.

Input  : raw DatasetDict from load_vimedaqa.load_vimedaqa()
Output : corpus.jsonl, val_qas.jsonl, test_qas.jsonl,
         qrels_val.json, qrels_test.json

Topic filtering strategy
------------------------
ViMedAQA is a general Vietnamese medical dataset — not VTM-specific.
We use it as a proxy baseline with the following filter:

  Corpus   →  drug (2) + medicine (3)  from train + val
               Broadest VTM-adjacent coverage; no test leakage.
  Eval     →  medicine (3)             from test split only
               Most controlled and most VTM-relevant question set.

Usage:
    from src.data.load_vimedaqa import load_vimedaqa
    from src.data.data_processing import process

    ds     = load_vimedaqa()
    result = process(ds)
"""

import hashlib
import json
import statistics
from pathlib import Path

import jsonlines
from datasets import DatasetDict
from loguru import logger

OUTPUT_DIR    = Path("data/processed")
CORPUS_TOPICS = {2, 3}  # drug + medicine  →  knowledge base
EVAL_TOPIC    = 3       # medicine only     →  evaluation questions


def _context_hash(context: str) -> str:
    """
    Derive a stable 12-char doc_id from passage text (MD5 prefix).

    Because every ViMedAQA QA pair was generated from its 'context' field,
    hash(context) == the ground-truth relevant doc_id with zero extra annotation.
    """
    return hashlib.md5(context.encode("utf-8")).hexdigest()[:12]


def _deduplicate_corpus(rows: list[dict]) -> list[dict]:
    """
    Remove duplicate passages from a list of raw rows.

    One YouMed article can generate many QA pairs all sharing the same
    context text. Keeping duplicates inflates the index and skews Recall@k.
    Deduplication key: MD5 hash of the context string.
    """
    seen:   set[str]   = set()
    corpus: list[dict] = []

    for row in rows:
        doc_id = _context_hash(row["context"])
        if doc_id not in seen:
            seen.add(doc_id)
            corpus.append({
                "doc_id":      doc_id,
                "title":       row["title"],
                "text":        row["context"],
                "keyword":     row["keyword"],
                "article_url": row["article_url"],
                "topic":       row["topic"],
            })

    return corpus


def _build_qrels(rows: list[dict]) -> dict[str, str]:
    """
    Build ground-truth retrieval labels.

    Returns {question_idx: doc_id} where doc_id = hash(row["context"]).
    The relevant passage for any question is the one it was derived from.
    """
    return {
        row["question_idx"]: _context_hash(row["context"])
        for row in rows
    }


def _save_jsonl(data: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, mode="w") as writer:
        writer.write_all(data)
    logger.info(f"  Saved {len(data):>7,} rows     →  {path}")


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"  Saved {len(data):>7,} entries  →  {path}")


def _length_stats(corpus: list[dict]) -> None:
    """
    Report word-count statistics over corpus passages.
    Median < 400 words → no chunking required before indexing.
    """
    lengths = sorted(len(doc["text"].split()) for doc in corpus)
    p95     = lengths[int(0.95 * len(lengths))]

    logger.info("── Passage length statistics (words) ───────────────────")
    logger.info(f"  min    : {lengths[0]}")
    logger.info(f"  median : {statistics.median(lengths):.0f}")
    logger.info(f"  mean   : {statistics.mean(lengths):.0f}")
    logger.info(f"  p95    : {p95}")
    logger.info(f"  max    : {lengths[-1]}")

    if statistics.median(lengths) > 400:
        logger.warning("  → median > 400 words: consider chunking before indexing")
    else:
        logger.success("  → passages are short; chunking is NOT required")


def _sanity_check(corpus: list[dict], qrels_test: dict[str, str]) -> None:
    """
    Verify every test question's gold passage exists in the corpus.

    If a test context's topic was excluded from CORPUS_TOPICS, its doc_id
    will be absent from the corpus — Recall@k will be 0 for those questions.
    """
    corpus_ids = {doc["doc_id"] for doc in corpus}
    missing    = [
        qid for qid, doc_id in qrels_test.items()
        if doc_id not in corpus_ids
    ]

    if missing:
        logger.warning(
            f"  {len(missing)} test questions have a gold passage NOT in the corpus.\n"
            f"  Their context topic may be excluded from CORPUS_TOPICS={CORPUS_TOPICS}."
        )
    else:
        logger.success(
            f"  Sanity check passed: all {len(qrels_test):,} test gold "
            f"passages are in the corpus."
        )


# ── Public API ────────────────────────────────────────────────────────────────

def process(
    ds:            DatasetDict,
    output_dir:    Path     = OUTPUT_DIR,
    corpus_topics: set[int] = CORPUS_TOPICS,
    eval_topic:    int      = EVAL_TOPIC,
) -> dict:
    """
    Filter, deduplicate, build qrels, and save all processed artifacts.

    Parameters
    ----------
    ds            : raw DatasetDict from load_vimedaqa()
    output_dir    : destination directory for all output files
    corpus_topics : topic IDs to include in the retrieval knowledge base
    eval_topic    : topic ID for evaluation questions (test split)

    Returns
    -------
    dict with keys:
        'corpus'      : list of unique passage dicts
        'val_qas'     : list of validation QA dicts
        'test_qas'    : list of test QA dicts
        'qrels_val'   : {question_idx: doc_id}
        'qrels_test'  : {question_idx: doc_id}
    """
    # ── 1. Filter rows ────────────────────────────────────────────────────────
    corpus_rows = (
        [r for r in ds["train"]      if r["topic"] in corpus_topics] +
        [r for r in ds["validation"] if r["topic"] in corpus_topics]
    )
    val_qas  = [r for r in ds["validation"] if r["topic"] == eval_topic]
    test_qas = [r for r in ds["test"]       if r["topic"] == eval_topic]

    logger.info(f"Corpus source rows  (topics={corpus_topics}): {len(corpus_rows):,}")
    logger.info(f"Val  questions      (topic={eval_topic}):  {len(val_qas):,}")
    logger.info(f"Test questions      (topic={eval_topic}):  {len(test_qas):,}")

    # ── 2. Deduplicate corpus ─────────────────────────────────────────────────
    corpus = _deduplicate_corpus(corpus_rows)
    logger.info(f"Unique passages after deduplication: {len(corpus):,}")

    # ── 3. Build qrels ────────────────────────────────────────────────────────
    qrels_val  = _build_qrels(val_qas)
    qrels_test = _build_qrels(test_qas)

    # ── 4. Save artifacts ─────────────────────────────────────────────────────
    logger.info("Saving artifacts …")
    _save_jsonl(corpus,   output_dir / "corpus.jsonl")
    _save_jsonl(val_qas,  output_dir / "val_qas.jsonl")
    _save_jsonl(test_qas, output_dir / "test_qas.jsonl")
    _save_json(qrels_val,  output_dir / "qrels_val.json")
    _save_json(qrels_test, output_dir / "qrels_test.json")

    # ── 5. Diagnostics ────────────────────────────────────────────────────────
    _length_stats(corpus)
    _sanity_check(corpus, qrels_test)

    return {
        "corpus":     corpus,
        "val_qas":    val_qas,
        "test_qas":   test_qas,
        "qrels_val":  qrels_val,
        "qrels_test": qrels_test,
    }


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.data.load_vimedaqa import load_vimedaqa

    ds     = load_vimedaqa()
    result = process(ds)

    logger.success(
        f"Done — "
        f"corpus: {len(result['corpus']):,} passages | "
        f"test: {len(result['test_qas']):,} questions | "
        f"qrels: {len(result['qrels_test']):,} entries"
    )
