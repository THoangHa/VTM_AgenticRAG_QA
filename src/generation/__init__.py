"""Import-safe generation interfaces; model libraries load only on demand."""

from src.generation.types import GenerationOutput, GenerationRequest
from src.generation.base_generator import BaseGenerator

__all__ = ["BaseGenerator", "GenerationOutput", "GenerationRequest"]
