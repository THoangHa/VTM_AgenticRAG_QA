# Run the BM25 baseline

This baseline retrieves each question's original source passage from ViMedAQA.
The corpus contains drug (topic 2) and medicine (topic 3) passages from train,
validation, and test. Evaluation uses medicine questions only. Only passage text
is indexed; questions and reference answers are not indexed. Use validation to
choose settings and test for final reporting.

## 1. Set up the environment

Run these commands from the repository root in PowerShell. Python 3.13 was used
for the verified run. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies needed for this baseline:

```powershell
python -m pip install datasets==5.0.1 rank-bm25==0.2.2 underthesea==9.5.0 underthesea_core==3.3.2 numpy==2.5.3 jsonlines loguru tqdm pytest
```

These core versions match the verified run; this fresh-install command has not
been independently tested. The project's `requirements.txt` also installs dense
retrieval and LLM dependencies, which are unnecessary for running BM25 alone.

## 2. Check the code

```powershell
python -m pytest tests -q
```

The tests cover processing, retrieval, cache compatibility, metrics, and the CLI.
They do not download ViMedAQA or require an LLM. The verified run passed 53 tests.

## 3. Create the corpus and evaluation data

```powershell
python -m src.data.data_processing
```

The loader downloads the ViMedAQA `all` configuration on the first run and saves
it under `data/raw/vimedaqa`. Subsequent runs reuse that local copy. Internet
access is required for the first download.

Processing preserves the raw data and:

1. Excludes relevant rows with blank or non-string contexts and records an audit.
2. Selects corpus topics 2/3 and evaluation topic 3.
3. Deduplicates exact context strings and assigns stable passage IDs.
4. Creates question-to-source relevance labels (qrels).
5. Checks that every retained evaluation question's gold passage is in the corpus.
6. Saves the benchmark files and audit report.

Passage IDs are the first 12 hexadecimal characters of MD5 over the original
UTF-8 context. Text is not normalized before hashing. An excluded evaluation row
is removed from both its QA file and its qrels. No replacement text is generated.

For the locally verified dataset, expect:

| Output | Count |
|---|---:|
| Excluded source rows | 35: 32 train, 3 validation, 0 test |
| Unique corpus passages | 10,238 |
| Validation questions | 693 |
| Test questions | 694 |
| Gold-passage coverage | 100% for both evaluation splits |

Check `data/processed/data_quality_report.json` for excluded question IDs, reasons,
source URLs, affected uses, policy version, and retained counts. These expected
counts describe the local dataset snapshot; an upstream dataset change may alter them.

## 4. Run validation

Build a fresh index and evaluate with the default settings:

```powershell
python scripts/evaluate_bm25.py --split val --rebuild --show 3
```

`--show 3` prints retrieval examples for the first three questions in addition
to evaluating the full split. Omit it for metrics only.

For later runs, the evaluator reuses a compatible index:

```powershell
python scripts/evaluate_bm25.py --split val --k1 1.5 --b 0.75 --top-k 10
```

| Option | Default | Purpose |
|---|---|---|
| `--split` | `val` | Evaluate `val` or `test` |
| `--top-k` | `10` | Positive number of passages returned per question |
| `--k1` | `1.5` | Positive term-frequency saturation parameter |
| `--b` | `0.75` | Length normalization, between 0 and 1 |
| `--rebuild` | Off | Force a new index |
| `--show N` | `0` | Print examples for the first N questions |

Choose parameters using validation. Each rerun overwrites that split's rankings
and metric summary; copy the summaries before comparing multiple configurations.

## 5. Run the final test evaluation

After fixing the settings, use the same settings on test:

```powershell
python scripts/evaluate_bm25.py --split test --k1 1.5 --b 0.75 --top-k 10
```

Reuse the same corpus, passage IDs, boundaries, and qrels for future dense and
hybrid retrieval comparisons.

## Outputs and metrics

All default output paths are anchored to the repository, regardless of the
current working directory.

| File | Contents |
|---|---|
| `data/processed/corpus.jsonl` | Deduplicated passage documents |
| `data/processed/val_qas.jsonl`, `test_qas.jsonl` | Retained evaluation QA rows |
| `data/processed/qrels_val.json`, `qrels_test.json` | Question ID to gold passage ID |
| `data/processed/data_quality_report.json` | Exclusion audit and counts |
| `data/processed/bm25_index.pkl` | Cached BM25 index |
| `results/bm25_{split}_results.jsonl` | Ranked document IDs and scores per question |
| `results/bm25_{split}_metrics.json` | Metrics, query count, settings, corpus fingerprint, dependency versions |

Recall@k is the fraction of questions whose gold passage appears in the first k
results. MRR@k averages the reciprocal gold-passage rank, using zero for misses.
With one gold passage per question, Precision@k equals Recall@k divided by k.
Recall/Precision at 1, 5, and 10 are reported when `--top-k` supports those cutoffs.
Credit goes to the original source passage; another useful passage may receive
no credit.

Validation on 2026-10-06, with defaults and 693 questions:

| Recall@1 | Recall@5 | Recall@10 | MRR@10 |
|---:|---:|---:|---:|
| 0.3983 | 0.5758 | 0.6508 | 0.4763 |

Test evaluation has not been run as of that verification date.

## How retrieval works

`BM25Retriever.fit()` segments lowercase Vietnamese passage text with
`underthesea` and builds corpus statistics using `rank_bm25.BM25Okapi`.
`retrieve()` tokenizes the question the same way, scores the passages, and sorts
by descending score, then document ID for ties. It returns passage dictionaries
with a `score` field. Empty tokenized queries return no results. Nonempty queries
can return zero-score candidates; BM25 scores are not relevance probabilities.

## Troubleshooting

- **Missing corpus or qrels:** run `python -m src.data.data_processing` first.
- **Invalid IDs, metadata, empty splits, or missing gold passages:** inspect the
  reported rows and audit. Processing validates before writing; failed runs can
  leave older files in place. Do not evaluate those files as a successful new run.
- **Cache mismatch:** the evaluator automatically rebuilds when corpus ID/text
  content or order, configuration, schema, or dependency versions change.
  `CacheMismatchError` is a named exception used to trigger that behavior.
- **Corrupt/unreadable cache:** rerun evaluation with `--rebuild`.
- **Metadata-only corpus changes:** use `--rebuild` to refresh returned metadata;
  the corpus fingerprint covers document IDs and text only.
- **Validation/test question-overlap warning:** inspect it before interpreting
  metrics. Identical question text is reported and retained.

## Code locations

- [Dataset loading](src/data/load_vimedaqa.py)
- [Corpus processing and exclusion audit](src/data/data_processing.py)
- [BM25 retriever and cache](src/retrieval/bm25.py)
- [Evaluation CLI](scripts/evaluate_bm25.py)
- [Metric functions](src/evaluation/metrics.py)
- [Tests](tests/test_bm25_baseline.py)
