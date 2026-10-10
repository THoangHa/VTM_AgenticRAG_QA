"""Resumable single-question generation runs and their artifacts."""

from datetime import datetime, timezone
from importlib import metadata
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import uuid

from src.evaluation.qa_evaluator import evaluate_qa
from src.generation.prompt_builder import ANSWER_STYLES, SYSTEM_LLM, SYSTEM_RAG
from src.generation.types import GenerationRequest


def _canonical_hash(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _git_commit(root):
    try:
        result = subprocess.run(["git", "-c", f"safe.directory={root}", "-C", str(root), "rev-parse", "HEAD"],
                                check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _environment(torch_module=None):
    names = ("torch", "transformers", "bitsandbytes", "accelerate", "huggingface-hub", "numpy", "scipy",
             "PyYAML", "rouge_score")
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    details = {"python": platform.python_version(), "platform": platform.platform(), "dependencies": versions}
    if torch_module is not None and torch_module.cuda.is_available():
        details["gpu"] = torch_module.cuda.get_device_name()
        details["cuda_runtime"] = torch_module.version.cuda
        details["gpu_total_memory_mib"] = torch_module.cuda.get_device_properties(0).total_memory / 2**20
    return details


def _read_answers(path, qids):
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        last_newline = raw.rfind(b"\n")
        trailing = raw[last_newline + 1:]
        try:
            json.loads(trailing)
            raw += b"\n"
        except (json.JSONDecodeError, UnicodeDecodeError):
            raw = raw[:last_newline + 1] if last_newline >= 0 else b""
        path.write_bytes(raw)
    rows, seen = [], set()
    for line in raw.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError("Only an incomplete trailing answer line may be discarded.") from exc
        qid = row.get("question_idx")
        if qid not in qids or qid in seen:
            raise ValueError("Resume file contains an unknown or duplicate question ID.")
        if not isinstance(row.get("predicted_answer"), str):
            raise ValueError("Resume answer record is missing a prediction.")
        seen.add(qid)
        rows.append(row)
    return rows, seen


def _run_generation_impl(data, mode, config, generator, *, run_dir=None, resume=False,
                         results_root=None, model_revision=None, retrieval_path=None, question_limit=None):
    if question_limit is not None and (isinstance(question_limit, bool) or not isinstance(question_limit, int)
                                        or question_limit <= 0):
        raise ValueError("question_limit must be a positive integer.")
    target_questions = list(data.questions.items())
    if question_limit is not None:
        target_questions = target_questions[:question_limit]
    target_ids = {qid for qid, _ in target_questions}
    root = Path(results_root or Path(__file__).resolve().parents[2] / "results" / "qa").resolve()
    if run_dir is None:
        root.mkdir(parents=True, exist_ok=True)
        run_dir = root / f"{mode}_{data.split}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"
    run_dir = Path(run_dir).resolve()
    if run_dir.exists() and not resume:
        raise FileExistsError(f"Run directory already exists; use explicit resume: {run_dir}")
    if resume and not run_dir.is_dir():
        raise FileNotFoundError(f"Resume directory does not exist: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    config_hash = _canonical_hash(config)
    manifest_path, answers_path, status_path = (run_dir / "manifest.json", run_dir / "answers.jsonl", run_dir / "status.json")
    reference_hash = _canonical_hash(data.references)
    expected = {"mode": mode, "split": data.split, "config_hash": config_hash,
                "question_limit": question_limit,
                "fingerprints": data.fingerprints, "reference_fingerprint": reference_hash,
                "retrieval_path": str(Path(retrieval_path).resolve()) if retrieval_path else None}
    if resume:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f"Cannot resume: manifest {key} does not match.")
        if json.loads(status_path.read_text(encoding="utf-8")).get("status") == "complete":
            raise ValueError("Completed runs cannot be resumed.")
        rows, completed = _read_answers(answers_path, set(data.questions))
    else:
        manifest = {**expected, "schema_version": 1, "resolved_config": config,
                    "model_revision": model_revision,
                    "prompt_version": f"{config['prompt']['version']}+{config['prompt'].get('answer_style', 'concise')}",
                    "prompt_templates": {"llm_only": SYSTEM_LLM, "rag": SYSTEM_RAG,
                                         "answer_style": ANSWER_STYLES[config["prompt"].get("answer_style", "concise")]},
                    "metric_version": config["evaluation"]["normalization"],
                    "cleanup_version": config["evaluation"]["cleanup_version"],
                    "environment": _environment(getattr(generator, "torch", None)),
                    "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
                    "created_at": datetime.now(timezone.utc).isoformat()}
        _write_json(manifest_path, manifest)
        answers_path.touch()
        rows, completed = [], set()
    _write_json(status_path, {"status": "running", "completed_answers": len(completed)})
    predictions = {row["question_idx"]: row["predicted_answer"] for row in rows}
    started_all = __import__("time").perf_counter()
    try:
        with answers_path.open("a", encoding="utf-8") as stream:
            for qid, question in target_questions:
                if qid in completed:
                    continue
                passages = data.request(qid, mode).passages
                request = GenerationRequest(question, mode, passages)
                output = generator.generate(request)
                if output.num_llm_calls != 1:
                    raise ValueError("Each question must use exactly one successful model call.")
                answer_row = {"question_idx": qid, "question": question,
                    "reference_answer": data.references[qid], "predicted_answer": output.answer,
                    "raw_completion": output.raw_completion, "prompt_tokens": output.prompt_tokens,
                    "completion_tokens": output.completion_tokens, "finish_reason": output.finish_reason,
                    "latency_seconds": output.latency_seconds, "num_llm_calls": output.num_llm_calls,
                    "abstained": output.abstained, "truncated": output.truncated,
                    "included_doc_ids": output.included_doc_ids, "cleanup_version": output.cleanup_version,
                    "included_passages": output.included_passages,
                    "malformed_thinking": output.malformed_thinking,
                    "peak_cuda_memory_mib": output.peak_cuda_memory_mib,
                    "oom_retries": output.oom_retries}
                stream.write(json.dumps(answer_row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                rows.append(answer_row)
                completed.add(qid)
                predictions[qid] = output.answer
                _write_json(status_path, {"status": "running", "completed_answers": len(completed)})
            target_rows = [row for row in rows if row["question_idx"] in target_ids]
            predictions = {row["question_idx"]: row["predicted_answer"] for row in target_rows}
            references = {qid: data.references[qid] for qid in target_ids}
            evaluated = evaluate_qa(predictions, references)
            denominator = len(target_rows)
            evaluated["rates"] = {
                "empty_answer": sum(not row["predicted_answer"].strip() for row in target_rows) / denominator,
                "abstention": sum(bool(row["abstained"]) for row in target_rows) / denominator,
                "truncation": sum(bool(row["truncated"]) for row in target_rows) / denominator,
                "output_limit": sum(row["finish_reason"] == "output_limit" for row in target_rows) / denominator,
                "malformed_thinking": sum(bool(row["malformed_thinking"]) for row in target_rows) / denominator}
            evaluated["mean_latency_seconds"] = sum(row["latency_seconds"] for row in target_rows) / denominator
            evaluated["mean_prompt_tokens"] = sum(row["prompt_tokens"] for row in target_rows) / denominator
            evaluated["mean_completion_tokens"] = sum(row["completion_tokens"] for row in target_rows) / denominator
            memory_values = [row["peak_cuda_memory_mib"] for row in target_rows
                             if row["peak_cuda_memory_mib"] is not None]
            evaluated["peak_cuda_memory_mib"] = max(memory_values) if memory_values else None
            evaluated["oom_retries"] = sum(row["oom_retries"] for row in target_rows)
            evaluated["wall_seconds"] = __import__("time").perf_counter() - started_all
            evaluated["full_query_count"] = len(data.questions)
            if data.rankings:
                from src.evaluation.qa_statistics import retrieval_hit_slices
                evaluated["retrieval_hit_miss"] = retrieval_hit_slices(
                    list(target_ids), data.gold_doc_ids, data.rankings, predictions, references, k=5)
            _write_json(run_dir / "metrics.json", evaluated)
            state = "complete" if target_ids == set(data.questions) else "partial"
            _write_json(status_path, {"status": state, "completed_answers": len(target_rows),
                                      "query_count": len(target_ids), "full_query_count": len(data.questions)})
        return run_dir
    except Exception as exc:
        _write_json(status_path, {"status": "failed", "completed_answers": len(completed),
                                  "error_type": type(exc).__name__, "error": str(exc)})
        raise


def run_generation(data, mode, config, generator, *, run_dir=None, resume=False,
                   results_root=None, model_revision=None, retrieval_path=None, question_limit=None):
    """Run generation and always close the generator, including setup failures."""
    try:
        return _run_generation_impl(data, mode, config, generator, run_dir=run_dir, resume=resume,
            results_root=results_root, model_revision=model_revision, retrieval_path=retrieval_path,
            question_limit=question_limit)
    finally:
        generator.close()
