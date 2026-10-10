"""Aggregate QA metrics independently from model inference."""

import math

from src.evaluation.qa_metrics import exact_match, f1_token, rouge_l

METRICS = {"exact_match": exact_match, "f1_token": f1_token, "rouge_l": rouge_l}


def evaluate_qa(predictions, references):
    if not isinstance(predictions, dict) or not isinstance(references, dict) or not references:
        raise ValueError("Nonempty prediction and reference dictionaries are required.")
    if set(predictions) != set(references):
        raise ValueError("Prediction and reference IDs must match exactly.")
    if any(not isinstance(ref, str) or not ref.strip() for ref in references.values()):
        raise ValueError("References must be nonempty strings.")
    if any(not isinstance(pred, str) for pred in predictions.values()):
        raise ValueError("Predictions must be strings.")
    per_question = {qid: {name: fn(predictions[qid], references[qid]) for name, fn in METRICS.items()}
                    for qid in references}
    aggregate = {name: math.fsum(row[name] for row in per_question.values()) / len(per_question)
                 for name in METRICS}
    return {"aggregate": aggregate, "per_question": per_question, "query_count": len(references)}
