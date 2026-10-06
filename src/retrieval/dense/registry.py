'''
Registry of dense retrievers, keyed by the model names in config.yaml.

Kept in its own module to avoid a circular import with dense_retriever.py.
'''

from src.retrieval.dense.BGEM3Retriever import BGEM3Retriever
from src.retrieval.dense.ViBiEncoderRetriever import ViBiEncoderRetriever

RETRIEVERS = {
    BGEM3Retriever.MODEL_KEY: BGEM3Retriever,
    ViBiEncoderRetriever.MODEL_KEY: ViBiEncoderRetriever,
}


def check_registry(config: dict) -> None:
    '''
    Raise if the registered retrievers and the YAML `models` keys differ.
    '''
    registered, configured = set(RETRIEVERS), set(config.get("models", {}))
    if registered != configured:
        raise ValueError(
            f"Registry/config mismatch. Only in code: {sorted(registered - configured)}; "
            f"only in config.yaml: {sorted(configured - registered)}"
        )
