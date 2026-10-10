"""Generator interface. Implementations own their model/tokenizer lifecycle."""

from abc import ABC, abstractmethod

from src.generation.types import GenerationOutput, GenerationRequest


class BaseGenerator(ABC):
    """One request produces one model call; close() must be safe to repeat."""

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationOutput:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError
