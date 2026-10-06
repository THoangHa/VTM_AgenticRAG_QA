'''
Dense retriever for BAAI/bge-m3.

BGE-M3 is multilingual and trained on raw text, so no preprocessing is applied.
'''

from src.retrieval.dense.dense_retriever import BaseDenseRetriever


class BGEM3Retriever(BaseDenseRetriever):
    # Must match the key under `models` in config.yaml
    MODEL_KEY = "bge_m3"

    def _prep(self, text: str) -> str:
        '''
        Identity: BGE-M3 takes raw Vietnamese text (no word segmentation needed).
        '''
        return text
