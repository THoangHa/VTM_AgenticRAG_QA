#!/usr/bin/env python
"""Run one reproducible generation arm or inspect its offline input statistics."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generation.hf_generator import HFGenerator
from src.generation.input_loader import load_generation_data
from src.generation.runner import run_generation


def _config(path):
    import yaml
    with Path(path).open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict) or not {"model", "generation", "prompt", "evaluation", "run"} <= value.keys():
        raise ValueError("Generation configuration is missing required sections.")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["llm_only", "rag_bm25", "rag_dense", "rag_oracle"], required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--rankings", type=Path,
        help="Completed retrieval rankings.jsonl (required for RAG; optional for llm_only hit/miss analysis).")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "generation.yaml")
    parser.add_argument("--run-dir", type=Path, help="New run directory or explicit directory with --resume.")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results" / "qa")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, help="Run a partial validation smoke sample; output is marked partial.")
    parser.add_argument("--max-context-docs", type=int)
    parser.add_argument("--max-input-tokens", type=int)
    parser.add_argument("--answer-style", choices=["concise", "short_span", "brief_sentence"])
    args = parser.parse_args()
    try:
        data = load_generation_data(args.data_dir, args.split, args.mode, args.rankings)
        config = _config(args.config)
        if args.max_context_docs is not None:
            config["prompt"]["max_context_docs"] = args.max_context_docs
        if args.max_input_tokens is not None:
            config["prompt"]["max_input_tokens"] = args.max_input_tokens
        if args.answer_style is not None:
            config["prompt"]["answer_style"] = args.answer_style
        if args.resume and args.run_dir is None:
            raise ValueError("--resume requires an explicit --run-dir.")
        generator = HFGenerator(config)
        resolved = deepcopy(config)
        resolved["model"]["resolved_revision"] = generator.revision
        try:
            path = run_generation(data, args.mode, resolved, generator, run_dir=args.run_dir,
                resume=args.resume, results_root=args.results_root, model_revision=generator.revision,
                retrieval_path=args.rankings, question_limit=args.limit)
        finally:
            generator.close()
        print(f"Saved {args.mode} generation run: {path}", flush=True)
        print(json.dumps(json.loads((path / "metrics.json").read_text(encoding="utf-8"))["aggregate"],
                         ensure_ascii=False, indent=2), flush=True)
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, OSError) as exc:
        parser.exit(1, f"Generation failed: {exc}\n")


if __name__ == "__main__":
    main()
