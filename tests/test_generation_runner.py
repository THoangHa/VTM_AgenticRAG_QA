import json

import pytest

from src.generation.input_loader import GenerationData
from src.generation.runner import _read_answers, run_generation
from src.generation.types import GenerationOutput


class FakeGenerator:
    torch = None

    def __init__(self, fail_after=None):
        self.calls = 0
        self.fail_after = fail_after
        self.closed = False

    def generate(self, request):
        self.calls += 1
        if self.fail_after and self.calls > self.fail_after:
            raise RuntimeError("synthetic failure")
        return GenerationOutput("raw", "answer", 10, 2, "eos", 0.01,
                                included_doc_ids=[], included_passages=[])

    def close(self):
        self.closed = True


def data():
    return GenerationData("val", {"q1": "one", "q2": "two"}, {"q1": "answer", "q2": "answer"},
        {"q1": ["d1"], "q2": ["d2"]}, {"d1": "first", "d2": "second"}, {},
        {"corpus": "c", "queries": "q", "qrels": "r"}, __import__("pathlib").Path("."))


def config():
    return {"prompt": {"version": "v1"}, "evaluation": {"normalization": "n1", "cleanup_version": "c1"}}


def test_runner_flushes_answers_and_marks_full_run_complete(tmp_path):
    generator = FakeGenerator()
    run_dir = run_generation(data(), "llm_only", config(), generator, run_dir=tmp_path / "run")
    assert generator.closed
    assert json.loads((run_dir / "status.json").read_text())["status"] == "complete"
    assert len((run_dir / "answers.jsonl").read_text().splitlines()) == 2
    assert json.loads((run_dir / "metrics.json").read_text())["query_count"] == 2


def test_failed_run_resumes_from_flushed_record_and_config_hash(tmp_path):
    run_dir = tmp_path / "run"
    with pytest.raises(RuntimeError, match="synthetic"):
        run_generation(data(), "llm_only", config(), FakeGenerator(fail_after=1), run_dir=run_dir)
    assert len((run_dir / "answers.jsonl").read_text().splitlines()) == 1
    run_generation(data(), "llm_only", config(), FakeGenerator(), run_dir=run_dir, resume=True)
    assert len((run_dir / "answers.jsonl").read_text().splitlines()) == 2


def test_partial_smoke_run_is_not_marked_complete(tmp_path):
    run_dir = run_generation(data(), "llm_only", config(), FakeGenerator(),
                             run_dir=tmp_path / "smoke", question_limit=1)
    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "partial"
    assert status["full_query_count"] == 2


def test_rejects_incompatible_resume(tmp_path):
    run_dir = tmp_path / "run"
    failed_generator = FakeGenerator(fail_after=1)
    with pytest.raises(RuntimeError):
        run_generation(data(), "llm_only", config(), failed_generator, run_dir=run_dir)
    assert failed_generator.closed
    changed = config()
    changed["prompt"]["version"] = "other"
    with pytest.raises(ValueError, match="config_hash"):
        run_generation(data(), "llm_only", changed, FakeGenerator(), run_dir=run_dir, resume=True)


def test_resume_discards_only_invalid_incomplete_trailing_line(tmp_path):
    path = tmp_path / "answers.jsonl"
    path.write_text('{"question_idx":"q1","predicted_answer":"a"}\n{"question_idx":"q2"', encoding="utf-8")
    rows, seen = _read_answers(path, {"q1", "q2"})
    assert seen == {"q1"}
    assert len(rows) == 1
    assert path.read_text(encoding="utf-8").endswith("\n")
