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
         qrels_val.json, qrels_test.json, data_quality_report.json

Topic filtering strategy
------------------------
ViMedAQA is a general Vietnamese medical dataset — not VTM-specific.
We use it as a proxy baseline with the following filter:

  Corpus   → drug (2) + medicine (3) from train + validation + test contexts.
  Eval     → medicine (3) questions from validation and test separately.

This is a fixed-corpus retrieval benchmark. Test source passages are searchable,
but test questions and answers are never indexed or used to tune retrieval.

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

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR    = ROOT / "data" / "processed"
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
    seen:   dict[str, str] = {}
    corpus: list[dict] = []

    for position, row in enumerate(rows):
        if not isinstance(row.get("context"), str) or not row["context"].strip():
            raise ValueError(
                f"Corpus source row {position} (question {row.get('question_idx', '?')}) "
                "has blank/non-string context. Repair the source data before processing."
            )
        required = {"title", "keyword", "article_url", "topic"}
        if not required.issubset(row):
            raise ValueError(f"Corpus source row {position} is missing metadata: {required - row.keys()}")
        doc_id = _context_hash(row["context"])
        if doc_id in seen and seen[doc_id] != row["context"]:
            raise ValueError(f"Document hash collision: {doc_id} identifies different contexts.")
        if doc_id not in seen:
            seen[doc_id] = row["context"]
            corpus.append({
                "doc_id":      doc_id,
                "title":       row["title"],
                "text":        row["context"],
                "keyword":     row["keyword"],
                "article_url": row["article_url"],
                "topic":       row["topic"],
            })

    return corpus


def _build_qrels(rows: list[dict], split: str = "evaluation") -> dict[str, str]:
    """
    Build ground-truth retrieval labels.

    Returns {question_idx: doc_id} where doc_id = hash(row["context"]).
    The relevant passage for any question is the one it was derived from.
    """
    if not rows:
        raise ValueError(f"{split}: evaluation set is empty.")
    qrels = {}
    for position, row in enumerate(rows):
        qid = row.get("question_idx")
        if not isinstance(qid, str) or not qid.strip() or qid in qrels:
            raise ValueError(f"{split}: row {position} has invalid/duplicate question ID {qid!r}.")
        for field in ("context", "question"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"{split}: question {qid} has blank/non-string {field}.")
        qrels[qid] = _context_hash(row["context"])
    return qrels


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
    Whitespace counts are descriptive, not a rule that chunking is required.
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
        logger.info("  → median <= 400 words; inspect the long tail before chunking")


def _sanity_check(corpus: list[dict], qrels: dict[str, str], split: str = "test") -> None:
    """
    Require complete gold-passage coverage before writing or evaluating data.
    """
    if not corpus:
        raise ValueError("Corpus is empty after topic filtering.")
    corpus_ids = {doc["doc_id"] for doc in corpus}
    if len(corpus_ids) != len(corpus):
        raise ValueError("Corpus document IDs must be unique.")
    missing    = [
        qid for qid, doc_id in qrels.items()
        if doc_id not in corpus_ids
    ]

    if missing:
        raise ValueError(
            f"{split}: {len(missing)} gold passages missing from corpus; "
            f"question IDs: {', '.join(missing)}. Check topic filtering."
        )
    logger.success(f"{split}: gold coverage 100% ({len(qrels):,}/{len(qrels):,})")


# ── Public API ────────────────────────────────────────────────────────────────

def process(
    ds:            DatasetDict,
    output_dir:    Path | str = OUTPUT_DIR,
    corpus_topics: set[int] = CORPUS_TOPICS,
    eval_topic:    int      = EVAL_TOPIC,
) -> dict:
    """
    Exclude invalid contexts with an audit, filter, deduplicate, and build qrels.

    Parameters
    ----------
    ds            : raw DatasetDict from load_vimedaqa()
    output_dir    : destination directory for all output files
    corpus_topics : topic IDs to include in the retrieval knowledge base
    eval_topic    : topic ID for evaluation questions (test split)

    Returns
    -------
    dict with keys:
        'data_quality_report': exclusion policy, audited rows, and counts
        'corpus'      : list of unique passage dicts
        'val_qas'     : list of validation QA dicts
        'test_qas'    : list of test QA dicts
        'qrels_val'   : {question_idx: doc_id}
        'qrels_test'  : {question_idx: doc_id}
    """
    # ── 1. Filter rows ────────────────────────────────────────────────────────
    output_dir = Path(output_dir)
    # Apply the same eligibility policy to corpus sources and evaluation rows.
    # Keep raw data intact; audit only rows relevant to this benchmark.
    cleaned = {}
    excluded = []
    for split in ("train", "validation", "test"):
        cleaned[split] = []
        for position, row in enumerate(ds[split]):
            relevant = row["topic"] in corpus_topics or (
                split != "train" and row["topic"] == eval_topic
            )
            context = row.get("context")
            if relevant and (not isinstance(context, str) or not context.strip()):
                excluded.append({
                    "split": split, "row_position": position,
                    "question_idx": row.get("question_idx"), "topic": row["topic"],
                    "article_url": row.get("article_url"),
                    "reason": "blank_context" if isinstance(context, str) else "non_string_context",
                    "excluded_from_corpus": row["topic"] in corpus_topics,
                    "excluded_from_evaluation": split != "train" and row["topic"] == eval_topic,
                })
                continue
            cleaned[split].append(row)
    if excluded:
        logger.warning(f"Excluded {len(excluded)} invalid-context source rows; see data_quality_report.json.")
    corpus_rows = [
        row for split in ("train", "validation", "test")
        for row in cleaned[split] if row["topic"] in corpus_topics
    ]
    val_qas  = [r for r in cleaned["validation"] if r["topic"] == eval_topic]
    test_qas = [r for r in cleaned["test"]       if r["topic"] == eval_topic]
    quality_report = {
        "schema_version": 1, "policy": "exclude_invalid_context_v1",
        "corpus_topics": sorted(corpus_topics), "eval_topic": eval_topic,
        "excluded_count": len(excluded), "excluded_rows": excluded,
        "splits": {
            split: {
                "raw_rows": len(ds[split]),
                "excluded_context_rows": sum(r["split"] == split for r in excluded),
                "excluded_eval_questions": sum(
                    r["split"] == split and r["excluded_from_evaluation"] for r in excluded
                ),
            } for split in ("train", "validation", "test")
        },
    }

    logger.info(f"Corpus source rows  (topics={corpus_topics}): {len(corpus_rows):,}")
    logger.info(f"Val  questions      (topic={eval_topic}):  {len(val_qas):,}")
    logger.info(f"Test questions      (topic={eval_topic}):  {len(test_qas):,}")

    # ── 2. Deduplicate corpus ─────────────────────────────────────────────────
    corpus = _deduplicate_corpus(corpus_rows)
    logger.info(f"Unique passages after deduplication: {len(corpus):,}")
    logger.info(f"Duplicate source passages removed: {len(corpus_rows) - len(corpus):,}")

    # ── 3. Build qrels ────────────────────────────────────────────────────────
    qrels_val  = _build_qrels(val_qas, "val")
    qrels_test = _build_qrels(test_qas, "test")
    _sanity_check(corpus, qrels_val, "val")
    _sanity_check(corpus, qrels_test, "test")
    overlap = {row["question"] for row in val_qas} & {row["question"] for row in test_qas}
    if overlap:
        logger.warning(f"{len(overlap)} identical question texts occur across val/test; retained unchanged.")

    # ── 4. Save artifacts ─────────────────────────────────────────────────────
    logger.info("Saving artifacts …")
    _save_jsonl(corpus,   output_dir / "corpus.jsonl")
    _save_jsonl(val_qas,  output_dir / "val_qas.jsonl")
    _save_jsonl(test_qas, output_dir / "test_qas.jsonl")
    _save_json(qrels_val,  output_dir / "qrels_val.json")
    _save_json(qrels_test, output_dir / "qrels_test.json")

    quality_report["retained_counts"] = {
        "corpus": len(corpus), "val_questions": len(val_qas), "test_questions": len(test_qas),
    }
    _save_json(quality_report, output_dir / "data_quality_report.json")

    # ── 5. Diagnostics ────────────────────────────────────────────────────────
    _length_stats(corpus)

    return {
        "data_quality_report": quality_report,
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
