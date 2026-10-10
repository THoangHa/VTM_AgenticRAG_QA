"""Load generation data through the repository's file-based benchmark boundary."""

from dataclasses import dataclass
import json
import math
from numbers import Real
from pathlib import Path

from src.evaluation.benchmark_data import ROOT, load_data, read_jsonl
from src.generation.types import GenerationRequest, Passage


@dataclass
class GenerationData:
    split: str
    questions: dict[str, str]
    references: dict[str, str]
    gold_doc_ids: dict[str, list[str]]
    corpus: dict[str, str]
    rankings: dict[str, list[dict]]
    fingerprints: dict[str, str]
    data_directory: Path

    def request(self, qid, mode):
        passages = []
        if mode == "rag_oracle":
            for did in self.gold_doc_ids[qid]:
                passages.append(Passage(did, self.corpus[did], source="oracle"))
        elif mode in {"rag_bm25", "rag_dense"}:
            for row in self.rankings[qid]:
                passages.append(Passage(row["doc_id"], self.corpus[row["doc_id"]], row["score"]))
        return GenerationRequest(self.questions[qid], mode, tuple(passages))


def _qa_records(path, questions):
    records = {}
    for row in read_jsonl(path):
        if not isinstance(row, dict):
            raise ValueError("QA records must be JSON objects.")
        qid = row.get("question_idx")
        if not isinstance(qid, str) or not qid.strip() or qid in records:
            raise ValueError("QA question IDs must be unique nonblank strings.")
        if qid not in questions or row.get("question") != questions[qid]:
            raise ValueError(f"{qid}: QA question does not match the benchmark split.")
        answer = row.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"{qid}: reference answer must be nonblank.")
        records[qid] = row
    if set(records) != set(questions):
        raise ValueError("QA records must cover every question in the requested split.")
    return records


def _rankings(path, questions, corpus_ids, split, fingerprints, expected_model=None):
    path = Path(path).resolve()
    metadata_path = path.parent / "metrics.json"
    if not metadata_path.is_file():
        raise ValueError(f"Ranking sibling metadata is required: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Ranking sibling metrics must be a JSON object.")
    if metadata.get("split") != split:
        raise ValueError("Ranking split does not match the requested QA split.")
    if metadata.get("fingerprints") != fingerprints:
        raise ValueError("Ranking data fingerprints do not match the current split and corpus.")
    status_path = path.parent / "status.json"
    if not status_path.is_file():
        raise ValueError("Only complete retrieval runs can supply generation rankings.")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if not isinstance(status, dict) or status.get("status") != "complete":
        raise ValueError("Only complete retrieval runs can supply generation rankings.")
    if expected_model is not None and metadata.get("model") != expected_model:
        raise ValueError(f"{expected_model} generation requires rankings from the {expected_model} retriever.")
    rankings = {}
    for row in read_jsonl(path):
        if not isinstance(row, dict):
            raise ValueError("Ranking rows must be JSON objects.")
        qid = row.get("question_idx")
        if not isinstance(qid, str) or qid not in questions or qid in rankings:
            raise ValueError("Ranking question IDs must be known and unique.")
        results = row.get("results")
        if not isinstance(results, list):
            raise ValueError(f"{qid}: ranking results must be a list.")
        seen = set()
        checked = []
        for item in results:
            if not isinstance(item, dict):
                raise ValueError(f"{qid}: ranking entries must be JSON objects.")
            did, score = item.get("doc_id"), item.get("score")
            if not isinstance(did, str) or did not in corpus_ids:
                raise ValueError(f"{qid}: ranking contains an unknown corpus document ID.")
            if did in seen:
                raise ValueError(f"{qid}: duplicate ranked document {did}.")
            if isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(score):
                raise ValueError(f"{qid}/{did}: ranking score must be finite.")
            seen.add(did)
            checked.append({"doc_id": did, "score": float(score)})
        rankings[qid] = checked
    if set(rankings) != set(questions):
        missing = sorted(set(questions) - set(rankings))
        raise ValueError(f"Rankings must contain one row per question; missing {missing[:3]}.")
    return rankings


def load_generation_data(data_dir=ROOT / "data" / "processed", split="val", mode="llm_only", ranking_path=None):
    if mode not in {"llm_only", "rag_bm25", "rag_dense", "rag_oracle"}:
        raise ValueError("Unsupported generation mode.")
    if mode in {"rag_bm25", "rag_dense"} and ranking_path is None:
        raise ValueError(f"{mode} requires --rankings from a completed retrieval run.")
    benchmark = load_data(data_dir, split)
    rows = _qa_records(Path(data_dir) / f"{split}_qas.jsonl", benchmark.queries)
    corpus = {doc["doc_id"]: doc["text"] for doc in benchmark.corpus}
    gold = {qid: sorted((did for did, grade in labels.items() if grade > 0), key=lambda did: (-labels[did], did))
            for qid, labels in benchmark.qrels.items()}
    rankings = {}
    if ranking_path is not None:
        expected_model = "bm25" if mode == "rag_bm25" else "bge_m3" if mode == "rag_dense" else None
        rankings = _rankings(ranking_path, benchmark.queries, set(corpus), split,
                             benchmark.fingerprints, expected_model)
    return GenerationData(split, benchmark.queries,
        {qid: rows[qid]["answer"] for qid in benchmark.queries}, gold, corpus, rankings,
        benchmark.fingerprints, benchmark.directory)
