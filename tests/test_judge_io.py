import json

import pytest

from src.evaluation.benchmark_data import load_data
from src.evaluation.judge_io import export_judge_batches, import_judgements, make_sample_manifest
from test_generation_inputs import make_data, make_rankings


def test_judge_export_import_round_trip_and_validation(tmp_path):
    data_dir = make_data(tmp_path / "data")
    ranking = make_rankings(data_dir, tmp_path / "retrieval" / "rankings.jsonl")
    sample_path = tmp_path / "shared_sample.json"
    sample = make_sample_manifest(data_dir, "val", ranking, sample_size=2, seed=42)
    sample_path.write_text(json.dumps(sample), encoding="utf-8")
    run_dir = tmp_path / "generation"
    (run_dir / "judge").mkdir(parents=True)
    benchmark = load_data(data_dir)
    manifest = {"mode": "rag_dense", "split": "val", "fingerprints": benchmark.fingerprints,
                "retrieval_path": str(ranking)}
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "status.json").write_text('{"status":"complete"}', encoding="utf-8")
    (run_dir / "answers.jsonl").write_text("".join(json.dumps({"question_idx": qid,
        "predicted_answer": "answer", "included_doc_ids": ["d2"],
        "included_passages": [{"doc_id": "d2", "text": "other"}]}) + "\n" for qid in ("q1", "q2")), encoding="utf-8")
    paths = export_judge_batches(run_dir, data_dir, sample_path, batch_size=1)
    exported = json.loads(paths[0].read_text(encoding="utf-8"))
    item = exported["items"][0]
    assert "reference" in item and "system_answer" in item and "passages" in item
    qid = item["question_idx"]
    response = tmp_path / "response.json"
    response.write_text(json.dumps([{"question_idx": item["question_idx"], "correctness": "correct",
        "faithfulness": "yes", "abstention_appropriate": "yes"} for item in sample["items"]]), encoding="utf-8")
    output = import_judgements(run_dir, [response], rag_mode=True, sample_manifest_path=sample_path)
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2


def test_judge_import_rejects_malformed_or_incomplete_labels(tmp_path):
    sample = {"items": [{"question_idx": "q1"}]}
    sample_path = tmp_path / "sample.json"
    sample_path.write_text(json.dumps(sample), encoding="utf-8")
    run = tmp_path / "run"
    run.mkdir()
    bad = tmp_path / "bad.json"
    bad.write_text('[{"question_idx":"q1","correctness":"maybe","abstention_appropriate":"yes"}]', encoding="utf-8")
    with pytest.raises(ValueError, match="correctness"):
        import_judgements(run, [bad], rag_mode=False, sample_manifest_path=sample_path)
    malformed = tmp_path / "malformed.json"
    malformed.write_text("not json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        import_judgements(run, [malformed], rag_mode=False, sample_manifest_path=sample_path)
