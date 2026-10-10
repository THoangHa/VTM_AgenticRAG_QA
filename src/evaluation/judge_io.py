"""Blinded judge sample selection, export, and strict JSON import."""

import json
from pathlib import Path
import random

from src.evaluation.benchmark_data import load_data, read_jsonl
from src.generation.input_loader import _rankings

CORRECTNESS = {"correct", "partial", "incorrect"}


def make_sample_manifest(data_dir, split, rankings_path, sample_size=150, seed=42, top_k=5):
    data = load_data(data_dir, split)
    rows = _rankings(rankings_path, data.queries, {d["doc_id"] for d in data.corpus}, split, data.fingerprints)
    hit, miss = [], []
    for qid in data.queries:
        target = hit if set(data.qrels[qid]) & {r["doc_id"] for r in rows[qid][:top_k]} else miss
        target.append(qid)
    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or not 2 <= sample_size <= len(data.queries):
        raise ValueError("sample_size must be between 2 and the number of questions.")
    rng = random.Random(seed)
    groups = {"hit": hit, "miss": miss}
    sizes = {}
    if not hit or not miss:
        only = "hit" if hit else "miss"
        sizes[only] = sample_size
    else:
        hit_size = round(sample_size * len(hit) / len(data.queries))
        hit_size = min(max(hit_size, 1), sample_size - 1)
        sizes = {"hit": hit_size, "miss": sample_size - hit_size}
    selected = []
    for name, values in groups.items():
        rng.shuffle(values)
        selected.extend((qid, name) for qid in values[:sizes.get(name, 0)])
    rng.shuffle(selected)
    return {"schema_version": 1, "split": split, "sample_seed": seed, "sample_size": sample_size,
            "stratification_top_k": top_k, "ranking_path": str(Path(rankings_path).resolve()),
            "fingerprints": data.fingerprints,
            "items": [{"question_idx": qid, "retrieval_stratum": group} for qid, group in selected]}


def export_judge_batches(run_dir, data_dir, sample_manifest_path, *, rankings_path=None, batch_size=15,
                         sample_size=150, seed=42):
    run_dir = Path(run_dir).resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    status_path = run_dir / "status.json"
    if not status_path.is_file() or json.loads(status_path.read_text(encoding="utf-8")).get("status") != "complete":
        raise ValueError("Judge export requires a completed generation run.")
    sample_path = Path(sample_manifest_path).resolve()
    if sample_path.exists():
        sample = json.loads(sample_path.read_text(encoding="utf-8"))
    else:
        if rankings_path is None:
            rankings_path = manifest.get("retrieval_path")
        if not rankings_path:
            raise ValueError("Creating the shared judge sample requires a canonical --strata-rankings file.")
        sample = make_sample_manifest(data_dir, manifest["split"], rankings_path, sample_size, seed)
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample_path.write_text(json.dumps(sample, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    data = load_data(data_dir, manifest["split"])
    if sample.get("fingerprints") != data.fingerprints or sample.get("split") != manifest["split"]:
        raise ValueError("Judge sample manifest does not match the run's data split/fingerprints.")
    answers = {row["question_idx"]: row for row in read_jsonl(run_dir / "answers.jsonl")}
    qas = {row["question_idx"]: row for row in read_jsonl(data.directory / f"{data.split}_qas.jsonl")}
    corpus = {row["doc_id"]: row["text"] for row in data.corpus}
    mode = manifest["mode"]
    selected = list(sample["items"])
    if any(item["question_idx"] not in answers or item["question_idx"] not in qas for item in selected):
        raise ValueError("Judge sample IDs are not covered by this completed generation run.")
    judge_root = run_dir / "judge" / "export"
    judge_root.mkdir(parents=True, exist_ok=True)
    controls = {"instructions": "These are rubric sanity checks only; do not include them in imported judgments.",
        "items": [
            {"question_idx": "sanity_correct", "question": "Đoạn văn cho biết nhiệt độ là bao nhiêu?",
             "reference": "37 °C", "passages": [{"doc_id": "control_1", "text": "Nhiệt độ được ghi nhận là 37 °C."}],
             "system_answer": "37 °C"},
            {"question_idx": "sanity_wrong", "question": "Đoạn văn cho biết nhiệt độ là bao nhiêu?",
             "reference": "37 °C", "passages": [{"doc_id": "control_2", "text": "Nhiệt độ được ghi nhận là 37 °C."}],
             "system_answer": "42 °C"}]}
    (judge_root / "sanity_checks.json").write_text(json.dumps(controls, ensure_ascii=False, indent=2) + "\n",
                                                    encoding="utf-8")
    paths = []
    for start in range(0, len(selected), batch_size):
        batch = selected[start:start + batch_size]
        # Shuffle each blinded batch deterministically without exposing the model arm.
        random.Random(sample["sample_seed"] + start).shuffle(batch)
        items = []
        for item in batch:
            qid = item["question_idx"]
            answer = answers[qid]
            passages = answer.get("included_passages", [])
            if not passages:
                included = answer.get("included_doc_ids", [])
                passages = [{"doc_id": did, "text": corpus[did]} for did in included if did in corpus]
            items.append({"question_idx": qid, "question": qas[qid]["question"], "reference": qas[qid]["answer"],
                          "passages": passages if mode != "llm_only" else [],
                          "system_answer": answer["predicted_answer"]})
        path = judge_root / f"batch_{start // batch_size + 1:03d}.json"
        path.write_text(json.dumps({"instructions": "Return only a strict JSON array with one object per item. "
            "Use question_idx, correctness (correct/partial/incorrect), abstention_appropriate (yes/no), "
            "and faithfulness (yes/no) when passages are supplied. Do not add markdown fences or explanations.",
            "rubric": {
            "correctness": sorted(CORRECTNESS),
            "faithfulness": "yes/no; required for RAG modes",
            "abstention_appropriate": "yes/no"}, "items": items}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def import_judgements(run_dir, response_paths, *, rag_mode, sample_manifest_path,
                      judge_model=None, judge_date=None):
    run_dir = Path(run_dir).resolve()
    if judge_model or judge_date:
        if not judge_model or not judge_date:
            raise ValueError("Judge model and date must be recorded together.")
        from datetime import date
        date.fromisoformat(judge_date)
    sample = json.loads(Path(sample_manifest_path).read_text(encoding="utf-8"))
    expected = {item["question_idx"] for item in sample["items"]}
    all_rows, raw_dir = [], run_dir / "judge" / "raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)
    seen = set()
    for index, source in enumerate(map(Path, response_paths), 1):
        payload = source.read_text(encoding="utf-8")
        value = json.loads(payload)
        if not isinstance(value, list):
            raise ValueError("Each pasted judge response must be a strict JSON array.")
        (raw_dir / f"response_{index:03d}.json").write_text(payload, encoding="utf-8")
        for row in value:
            if not isinstance(row, dict):
                raise ValueError("Judge entries must be JSON objects.")
            qid = row.get("question_idx")
            if qid not in expected or qid in seen:
                raise ValueError("Judge response contains an unexpected or duplicate question ID.")
            if row.get("correctness") not in CORRECTNESS:
                raise ValueError(f"{qid}: correctness must be correct, partial, or incorrect.")
            if row.get("abstention_appropriate") not in {"yes", "no"}:
                raise ValueError(f"{qid}: abstention_appropriate must be yes or no.")
            if rag_mode:
                if row.get("faithfulness") not in {"yes", "no"}:
                    raise ValueError(f"{qid}: RAG judgments require faithfulness yes/no.")
            elif "faithfulness" in row:
                raise ValueError(f"{qid}: faithfulness is not applicable to llm_only.")
            all_rows.append(row)
            seen.add(qid)
    if seen != expected:
        raise ValueError(f"Judge sample incomplete; missing {len(expected - seen)} question judgments.")
    output = run_dir / "judge" / "judgments.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in all_rows),
                      encoding="utf-8")
    if judge_model or judge_date:
        (run_dir / "judge" / "session.json").write_text(json.dumps(
            {"judge_model": judge_model, "judge_date": judge_date, "response_count": len(response_paths)},
            ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output
