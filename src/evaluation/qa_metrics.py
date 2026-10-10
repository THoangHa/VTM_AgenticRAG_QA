"""Pure Vietnamese-friendly lexical QA metrics."""

import unicodedata
from collections import Counter


def normalize_answer(text):
    if not isinstance(text, str):
        raise TypeError("Answers must be strings.")
    text = unicodedata.normalize("NFC", text).lower()
    text = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in text)
    return " ".join(text.split())


def exact_match(prediction, reference):
    return float(normalize_answer(prediction) == normalize_answer(reference))


def f1_token(prediction, reference):
    pred, ref = normalize_answer(prediction).split(), normalize_answer(reference).split()
    if not pred or not ref:
        return float(not pred and not ref)
    overlap = sum((Counter(pred) & Counter(ref)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction, reference):
    pred, ref = normalize_answer(prediction).split(), normalize_answer(reference).split()
    if not pred or not ref:
        return float(not pred and not ref)
    # One-row dynamic programming keeps memory proportional to the shorter answer.
    if len(ref) > len(pred):
        pred, ref = ref, pred
    previous = [0] * (len(ref) + 1)
    for token in pred:
        current = [0]
        for j, other in enumerate(ref, 1):
            current.append(previous[j - 1] + 1 if token == other else max(previous[j], current[-1]))
        previous = current
    lcs = previous[-1]
    precision, recall = lcs / len(pred), lcs / len(ref)
    return 2 * precision * recall / (precision + recall) if lcs else 0.0
