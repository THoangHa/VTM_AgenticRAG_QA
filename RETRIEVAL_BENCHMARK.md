# Shared retrieval benchmark

This benchmark uses the project's ViMedAQA passages and qrels. It follows BEIR's
separation of dataset, retriever, run, and evaluator; it does not download BEIR
datasets or require BEIR at runtime. Selection measures original-source passage
retrieval, not answer generation or overall medical relevance.

## Environment

Run from the repository root in PowerShell. Python 3.13 and Windows are used for
the local comparison. The RTX 3060 laptop GPU has 6 GiB VRAM. Install only the
retrieval dependencies into a project-local environment:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install -r requirements-benchmark.txt
.venv/Scripts/python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
```

The first model run needs internet access and several GiB of disk space. Model
weights stay under `.cache/models`; each model is pinned to its resolved Hub
revision. The comparison requires CUDA and never silently falls back to CPU.
The tested full environment is recorded in `requirements-benchmark-lock.txt`
after setup. To reproduce it, install CUDA PyTorch as above and then the lock.

## Offline tests and real-model smoke tests

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m pytest tests/test_dense_encode.py --run-model-tests -q
```

Default tests do not download or load models. Test scratch files stay inside
`.cache/pytest_<unique-id>/tmp`, with a fresh directory for each session.
Persistent pytest caching is disabled to avoid Windows folder permission
conflicts between terminal and sandbox accounts. An explicit `--basetemp` is
still respected. Real-model smoke checks use the actual model wrappers sequentially.
The metric reference fixture was generated using `pytrec-eval-terrier==0.5.10`;
this reference package is not a runtime dependency.

## Compare and select the dense baseline

```powershell
.venv/Scripts/python.exe scripts/benchmark_retrieval.py compare-dense
```

This loads validation only, benchmarks BGE-M3 and the Vietnamese bi-encoder,
then exports `configs/dense_baseline.yaml`. It preserves the existing corpus:
10,238 passages and 693 validation questions in the current snapshot. Questions
and reference answers are never indexed. Passage titles are not concatenated.

Both models use normalized float32 vectors and CPU FAISS exact inner-product
search, which computes cosine similarity. Encoding uses CUDA FP16 and initial
batch size 8, retrying CUDA out-of-memory at 4, 2, then 1. BGE-M3 uses raw text
and 512 tokens; the Vietnamese encoder uses PyVi segmentation and 256 tokens.
Tokenizer audits include special tokens and report truncation without changing
passage boundaries or labels.

Selection uses the highest unrounded mean nDCG@10. Exact ties use Recall@10,
MRR@10, then the Vietnamese encoder. Reported speed/memory do not change this
quality-based rule. A paired percentile bootstrap uses 10,000 query resamples,
seed 42, and a 95% interval for BGE-M3 minus Vietnamese encoder nDCG@10. An
interval including zero is inconclusive; it does not prevent the predefined
selection. Question resampling can underestimate uncertainty when questions
share the same source passage.

Recompare completed runs without reloading models:

```powershell
.venv/Scripts/python.exe scripts/benchmark_retrieval.py compare-dense --runs <bge-run-directory> <vi-run-directory>
```

## Evaluate an individual method

```powershell
.venv/Scripts/python.exe scripts/benchmark_retrieval.py evaluate --model bm25
.venv/Scripts/python.exe scripts/benchmark_retrieval.py evaluate --model bge_m3
.venv/Scripts/python.exe scripts/benchmark_retrieval.py evaluate --model vi_bi_encoder
```

Defaults: validation, depth 100, cutoffs 1/5/10/100, BM25 k1=1.5 and b=0.75.
Use `--top-k`, `--k-values`, `--batch-size`, `--data-dir`, `--index-root`,
`--results-root`, and `--dense-config` to override settings. Paths supplied by
users are relative to their current directory; default paths are repository
anchored. `evaluate --device cpu` explicitly allows CPU dense evaluation.
Selection comparisons still run on CUDA. `--split test` exists for subsequent
final reporting; it is never used by `compare-dense`.

Load the frozen baseline with the selected model key and config file:

```python
import yaml
from src.retrieval.dense.registry import RETRIEVERS

path = "configs/dense_baseline.yaml"
with open(path, encoding="utf-8") as stream:
    config = yaml.safe_load(stream)
retriever = RETRIEVERS[config["baseline_model"]](config_path=path, require_cuda=True)
retriever.load_index()
passages = retriever.retrieve("Cam thảo có tác dụng gì?", top_k=10)
```

The YAML identifies the selected model and exact revision. Its retriever reads
the matching cached index, encodes the question, and returns the top passages.
Rebuild the index if environment versions differ from the original run.

## Outputs and cache rules

Start with `results/benchmarks/LATEST.txt` for a readable summary in any editor.
For formatted tables, double-click `results/benchmarks/latest.html` in File Explorer
to view a formatted report in your web browser. This stable file updates after
each successful dense comparison and names the source comparison folder and
model runs. Opening HTML in a code editor shows its source instead.
The in-app browser may block local HTML URLs; the text summary works without a
browser. `latest.json` provides the same comparison pointer for scripts. Each comparison
also keeps its own `REPORT.html` alongside the Markdown report.

Every run has a unique directory under `results/benchmarks/`, containing:

- `metrics.json`: aggregate scores, settings, fingerprints, dependencies,
  truncation, OOM retries, build/evaluation timing, and CUDA memory.
- `rankings.jsonl` and `per_query.jsonl`: ranked IDs/scores and per-question metrics.
- `status.json`: running, complete, or failed. Failed runs are never compared.

Comparison directories contain `REPORT.md`, `comparison.json`, per-question
differences, and an immutable selected YAML snapshot. The convenience export
at `configs/dense_baseline.yaml` points back to its source runs. Compact reports
are versioned; large per-query files, caches, and indexes are ignored.

Warm-up uses the first ten validation questions. Latency then measures all
validation questions separately, with CUDA synchronization. Batched retrieval
throughput excludes audits and latency measurements. Preparation timing includes
model loading and index preparation; cached index build time is labelled as
historical. CUDA figures measure PyTorch allocated memory, not whole-process
GPU usage. Timing depends on laptop power/thermal conditions.

Readable caches with changed corpus, model revision, preprocessing, precision,
token limit, normalization, or encoding dependencies are rebuilt. Corrupt or
incomplete caches fail and require `--rebuild`. Checksums and snapshot validation
prevent loading misaligned or partially replaced indexes. Existing BM25 CLI
outputs and its legacy cache remain untouched.

## Extending retrieval

An adapter implements `search(queries, top_k)`, returning
`{query_id: {document_id: score}}`. The runner prepares its index, executes it,
validates IDs, saves the run, and invokes the shared evaluator. The evaluator
knows only rankings and relevance labels, so dense, lexical, hybrid, sparse,
or reranked methods can share it.

```python
from src.evaluation.evaluator import evaluate
report = evaluate({"q1": {"d7": 1}}, {"q1": {"d2": .9, "d7": .8}}, [1, 5, 10])
print(report.metrics, report.per_query)
```

The relevant passage is second: Recall@1 is zero, Recall@5 is one, and MRR@5
is one half. Metrics average across all labelled questions, including misses.
nDCG uses linear grades; AP divides by all relevant labels and Precision@k by
k even for short rankings. Ties use descending score then ascending document
ID. Identical query/document IDs are retained. These explicit tie and missing
query policies may differ from BEIR defaults; reference fixtures use complete
coverage and unique scores to verify the metric formulas.
