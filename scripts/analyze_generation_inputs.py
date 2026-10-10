#!/usr/bin/env python
"""Measure validation prompt/reference token lengths without loading a model."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.generation.input_loader import load_generation_data
from src.generation.prompt_builder import PromptBuilder
from src.generation.types import GenerationRequest


def _percentiles(values):
    values = sorted(values)
    def at(fraction):
        return values[min(len(values) - 1, round((len(values) - 1) * fraction))]
    return {"p50": at(.50), "p90": at(.90), "p95": at(.95), "p99": at(.99), "max": values[-1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--mode", choices=["llm_only", "rag_bm25", "rag_dense", "rag_oracle"], required=True)
    parser.add_argument("--rankings", type=Path)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--candidate-limits", type=int, nargs="+", default=[1024, 1536, 2048, 3072])
    args = parser.parse_args()
    try:
        from huggingface_hub import model_info
        from transformers import AutoTokenizer
        revision = model_info(args.tokenizer, revision=args.revision).sha
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=revision)
        data = load_generation_data(args.data_dir, args.split, args.mode, args.rankings)
        refs = [len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in data.references.values()]
        report = {"split": data.split, "fingerprints": data.fingerprints, "question_count": len(refs),
                  "tokenizer": args.tokenizer, "tokenizer_revision": revision,
                  "reference_tokens": _percentiles(refs), "prompt_profiles": {}}
        for k in (3, 5, 10):
            prompt_lengths = []
            for limit in args.candidate_limits:
                builder = PromptBuilder(tokenizer, max_input_tokens=1_000_000, max_context_docs=k)
                prompt_lengths.clear()
                for qid, question in data.questions.items():
                    request = data.request(qid, args.mode)
                    request = GenerationRequest(question, args.mode, request.passages)
                    prompt_lengths.append(builder.build(request).prompt_tokens)
                counts = [len(tokenizer(text, add_special_tokens=False)["input_ids"])
                          for text in data.references.values()]
                bounded = PromptBuilder(tokenizer, max_input_tokens=limit, max_context_docs=k)
                truncated = 0
                for qid, question in data.questions.items():
                    request = data.request(qid, args.mode)
                    request = GenerationRequest(question, args.mode, request.passages)
                    try:
                        truncated += bounded.build(request).truncated
                    except ValueError:
                        truncated += 1
                report["prompt_profiles"].setdefault(str(k), {})[str(limit)] = {
                    "tokens": _percentiles(prompt_lengths), "truncated_questions": int(truncated),
                    "truncation_rate": truncated / len(data.questions),
                    "reference_tokens": _percentiles(counts)}
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except (ValueError, RuntimeError, FileNotFoundError, OSError) as exc:
        parser.exit(1, f"Input analysis failed: {exc}\n")


if __name__ == "__main__":
    main()
