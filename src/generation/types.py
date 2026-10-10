"""Small data contracts shared by prompt construction and generators."""

from dataclasses import dataclass, field
import math
from numbers import Real
from typing import Literal

Mode = Literal["llm_only", "rag_bm25", "rag_dense", "rag_oracle"]


@dataclass(frozen=True)
class Passage:
    doc_id: str
    text: str
    score: float | None = None
    source: Literal["retrieved", "oracle"] = "retrieved"

    def __post_init__(self):
        if not isinstance(self.doc_id, str) or not self.doc_id.strip():
            raise ValueError("Passage ID must be a nonblank string.")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("Passage text must be a nonblank string.")
        if self.score is not None and (isinstance(self.score, bool) or not isinstance(self.score, Real)
                                       or not math.isfinite(self.score)):
            raise ValueError("Passage score must be finite when provided.")
        if self.source not in {"retrieved", "oracle"}:
            raise ValueError("Passage source must be retrieved or oracle.")


@dataclass(frozen=True)
class GenerationRequest:
    question: str
    mode: Mode
    passages: tuple[Passage, ...] = field(default_factory=tuple)

    def __post_init__(self):
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("Question must be a nonblank string.")
        if self.mode not in {"llm_only", "rag_bm25", "rag_dense", "rag_oracle"}:
            raise ValueError(f"Unsupported generation mode: {self.mode}")
        if self.mode == "llm_only" and self.passages:
            raise ValueError("llm_only requests cannot include passages.")
        if self.mode != "rag_oracle" and any(p.source == "oracle" for p in self.passages):
            raise ValueError("Oracle passages cannot be passed to normal generation modes.")
        if self.mode == "rag_oracle" and any(p.source != "oracle" for p in self.passages):
            raise ValueError("rag_oracle requests may contain only gold oracle passages.")


@dataclass
class GenerationOutput:
    raw_completion: str
    answer: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    latency_seconds: float
    num_llm_calls: int = 1
    abstained: bool = False
    truncated: bool = False
    included_doc_ids: list[str] = field(default_factory=list)
    included_passages: list[dict[str, str]] = field(default_factory=list)
    cleanup_version: str = "cl_v1"
    malformed_thinking: bool = False
    peak_cuda_memory_mib: float | None = None
    oom_retries: int = 0
