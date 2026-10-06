"""Read the project's unchanged files and expose a shared retrieval dataset."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def fingerprint(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def corpus_fingerprint(corpus):
    # Keep the existing BM25 fingerprint format to avoid invalidating its cache.
    digest = hashlib.sha256()
    for doc in corpus:
        digest.update((json.dumps([doc["doc_id"], doc["text"]], ensure_ascii=False) + "\n").encode("utf-8"))
    return digest.hexdigest()


def validate_corpus(corpus):
    if not isinstance(corpus, list) or not corpus:
        raise ValueError("Corpus must be a nonempty list.")
    seen = set()
    for doc in corpus:
        if not isinstance(doc, dict):
            raise ValueError("Corpus records must be dictionaries.")
        did, text = doc.get("doc_id"), doc.get("text")
        if not isinstance(did, str) or not did.strip() or did in seen:
            raise ValueError("Corpus IDs must be unique nonblank strings.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{did}: passage text must be nonblank.")
        seen.add(did)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


@dataclass
class BenchmarkData:
    corpus: list[dict]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]
    split: str
    directory: Path

    @property
    def fingerprints(self):
        return {"corpus": corpus_fingerprint(self.corpus), "queries": fingerprint(self.queries),
                "qrels": fingerprint(self.qrels)}


def load_data(directory=ROOT / "data" / "processed", split="val"):
    if split not in {"val", "test"}:
        raise ValueError("Split must be val or test.")
    directory = Path(directory).resolve()
    corpus = read_jsonl(directory / "corpus.jsonl")
    validate_corpus(corpus)
    queries = {}
    for row in read_jsonl(directory / f"{split}_qas.jsonl"):
        qid, text = row.get("question_idx"), row.get("question")
        if not isinstance(qid, str) or not qid.strip() or qid in queries:
            raise ValueError("Question IDs must be unique nonblank strings.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{qid}: question text must be nonblank.")
        queries[qid] = text
    with (directory / f"qrels_{split}.json").open(encoding="utf-8") as stream:
        labels = json.load(stream)
    if not queries or set(queries) != set(labels):
        raise ValueError("Question/qrels IDs must match and be nonempty.")
    qrels = {qid: {gold: 1} if isinstance(gold, str) else gold for qid, gold in labels.items()}
    from src.evaluation.evaluator import evaluate
    evaluate(qrels, {}, (1,))  # Validate labels before preparing expensive indexes.
    corpus_ids = {doc["doc_id"] for doc in corpus}
    if any(did not in corpus_ids for rels in qrels.values() for did in rels):
        raise ValueError("A labelled document is missing from the corpus.")
    return BenchmarkData(corpus, queries, qrels, split, directory)


def records_to_scores(records):
    scores = {}
    for doc in records:
        did = doc["doc_id"]
        if did in scores:
            raise ValueError(f"Duplicate retrieved document: {did}")
        scores[did] = doc["score"]
    return scores
