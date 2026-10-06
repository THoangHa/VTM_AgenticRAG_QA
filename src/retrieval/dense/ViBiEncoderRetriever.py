'''
Dense retriever for bkai-foundation-models/vietnamese-bi-encoder (PhoBERT-based).

The model was trained on word-segmented Vietnamese, so passages and queries
must be segmented with pyvi before encoding (done in _prep).
'''

from pyvi import ViTokenizer

from src.retrieval.dense.dense_retriever import BaseDenseRetriever


class ViBiEncoderRetriever(BaseDenseRetriever):
    # Must match the key under `models` in config.yaml
    MODEL_KEY = "vi_bi_encoder"
    PREPROCESSING = "pyvi_whitespace_collapse_v1"

    def _prep(self, text: str) -> str:
        '''
        Collapse whitespace/newlines, then word-segment with pyvi
        (e.g. "y học cổ truyền" -> "y_học cổ_truyền").
        Casing and diacritics are kept: PhoBERT is a cased model.
        '''
        return ViTokenizer.tokenize(" ".join(text.split()))
