"""
Smoke tests for BaseDenseRetriever._encode (Step 2).

Run from the repo root:
    python -m tests.test_dense_encode                 # both models
    python -m tests.test_dense_encode --model bge_m3  # one model
    python -m tests.test_dense_encode --device cpu

The first run downloads the models (BGE-M3 is about 2.2 GB).
"""

import argparse
import sys
import traceback

import numpy as np

MODEL_KEYS = ["bge_m3", "vi_bi_encoder"]

SENTENCES = [
    "Cam thảo có tác dụng gì trong y học cổ truyền?",   # query
    "Tác dụng của cam thảo trong đông y",               # paraphrase (related)
    "Giá vàng trong nước hôm nay tăng mạnh",            # unrelated
]


# ── Helpers ──────────────────────────────────────────────────────────────────
def _emb(out):
    """
    _encode should return just the embeddings array. If it still returns a
    (embeddings, norms) tuple, unwrap it so the other checks can run, and warn.
    """
    if isinstance(out, tuple):
        print("    [warn] _encode returns a tuple; it should return only the embeddings")
        return out[0]
    return out


# ── Tests (each takes a loaded retriever `r`) ────────────────────────────────
def test_abstract_base_cannot_be_instantiated(r):
    from src.retrieval.dense.dense_retriever import BaseDenseRetriever
    try:
        BaseDenseRetriever()
    except TypeError:
        return
    raise AssertionError("BaseDenseRetriever should be abstract")


def test_shape_and_dtype(r):
    e = _emb(r._encode(SENTENCES, show_progress=False))
    assert e.shape == (len(SENTENCES), r.dim), f"shape {e.shape}"
    assert e.dtype == np.float32, f"dtype {e.dtype}"
    assert e.flags["C_CONTIGUOUS"], "array must be C-contiguous for FAISS"


def test_unit_norm(r):
    e = _emb(r._encode(SENTENCES, show_progress=False))
    norms = np.linalg.norm(e, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), f"norms {norms}"


def test_no_nan_or_inf(r):
    e = _emb(r._encode(SENTENCES, show_progress=False))
    assert np.isfinite(e).all(), "NaN/Inf in embeddings (fp16 overflow?)"


def test_semantic_similarity(r):
    e = _emb(r._encode(SENTENCES, show_progress=False))
    sim = e @ e.T
    related, unrelated = sim[0, 1], sim[0, 2]
    print(f"    related={related:.3f}  unrelated={unrelated:.3f}")
    assert related > unrelated, "paraphrase should score higher than unrelated text"


def test_empty_list(r):
    e = r._encode([], show_progress=False)
    e = e[0] if isinstance(e, tuple) else e
    assert e.shape == (0, r.dim), f"shape {e.shape}"
    assert e.dtype == np.float32


def test_single_string(r):
    e = _emb(r._encode(SENTENCES[0], show_progress=False))
    assert e.shape == (r.dim,), f"shape {e.shape}"
    assert abs(np.linalg.norm(e) - 1.0) < 1e-4


def test_deterministic(r):
    a = _emb(r._encode(SENTENCES, show_progress=False))
    b = _emb(r._encode(SENTENCES, show_progress=False))
    assert np.allclose(a, b, atol=1e-4), "same input should give same output"


def test_batch_consistency(r):
    together = _emb(r._encode(SENTENCES, show_progress=False))
    alone = np.vstack([_emb(r._encode([s], show_progress=False)) for s in SENTENCES])
    diff = np.abs(together - alone).max()
    print(f"    max diff batch vs single = {diff:.5f}")
    assert diff < 5e-3, "batching/padding should not change embeddings noticeably"


def test_long_text_is_truncated(r):
    long_text = "Cam thảo là một vị thuốc quý trong y học cổ truyền. " * 400
    e = _emb(r._encode([long_text], show_progress=False))
    assert e.shape == (1, r.dim)
    assert np.isfinite(e).all()


def test_prep_is_applied(r):
    calls = []
    original = r._prep
    r._prep = lambda t: (calls.append(t), original(t))[1]
    try:
        r._encode(SENTENCES, show_progress=False)
    finally:
        r._prep = original
    assert calls == SENTENCES, "_encode must call _prep on every text"


def test_index_compatible(r):
    """Embeddings must be accepted by FAISS IndexFlatIP and self-match first."""
    import faiss

    e = _emb(r._encode(SENTENCES, show_progress=False))
    index = faiss.IndexFlatIP(r.dim)
    index.add(e)
    scores, idx = index.search(e[:1], 3)
    assert idx[0][0] == 0, f"query should retrieve itself first, got {idx[0]}"
    assert abs(scores[0][0] - 1.0) < 1e-3, f"self cosine should be ~1, got {scores[0][0]}"


TESTS = [
    test_abstract_base_cannot_be_instantiated,
    test_shape_and_dtype,
    test_unit_norm,
    test_no_nan_or_inf,
    test_semantic_similarity,
    test_empty_list,
    test_single_string,
    test_deterministic,
    test_batch_consistency,
    test_long_text_is_truncated,
    test_prep_is_applied,
    test_index_compatible,
]


# ── Runner ───────────────────────────────────────────────────────────────────
def run_for_model(key, device):
    print(f"\n=== {key} ===")
    from src.retrieval.dense.registry import RETRIEVERS
    r = RETRIEVERS[key](device=device, batch_size=8, require_cuda=device != "cpu")
    print(f"device={r.device}  dim={r.dim}  batch_size={r.batch_size}  index_dir={r.index_dir}")

    failed = 0
    for test in TESTS:
        try:
            test(r)
            print(f"  PASS  {test.__name__}")
        except Exception:
            failed += 1
            print(f"  FAIL  {test.__name__}")
            traceback.print_exc(limit=2)
    return failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_KEYS, default=None)
    parser.add_argument("--device", default=None, help="override device, e.g. cpu")
    args = parser.parse_args()

    keys = [args.model] if args.model else MODEL_KEYS
    total_failed = sum(run_for_model(k, args.device) for k in keys)

    print(f"\n{'ALL TESTS PASSED' if total_failed == 0 else f'{total_failed} TEST(S) FAILED'}")
    sys.exit(1 if total_failed else 0)


if __name__ == "__main__":
    main()
