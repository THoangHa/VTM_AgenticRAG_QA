import pytest
from uuid import uuid4


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    # Admin and sandbox sessions can own different Windows directories. A fresh
    # root avoids deleting or writing another session's protected scratch files.
    cache_root = config.rootpath / ".cache"
    cache_root.mkdir(exist_ok=True)
    session_root = cache_root / f"pytest_{uuid4().hex}"
    if config.option.basetemp is None:
        session_root.mkdir()
        config.option.basetemp = str(session_root / "tmp")


def pytest_addoption(parser):
    parser.addoption("--run-model-tests", action="store_true", default=False,
                     help="Download/load actual embedding models and run GPU smoke tests")


@pytest.fixture(scope="session", params=["bge_m3", "vi_bi_encoder"])
def r(request):
    if not request.config.getoption("--run-model-tests"):
        pytest.skip("Real models require --run-model-tests; offline tests do not download weights.")
    from src.retrieval.dense.registry import RETRIEVERS
    retriever = RETRIEVERS[request.param](device="cuda", batch_size=8, require_cuda=True)
    yield retriever
    del retriever
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
