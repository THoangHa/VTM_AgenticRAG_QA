import pytest

from src.generation.prompt_builder import PromptBuilder
from src.generation.types import GenerationRequest, Passage


class FakeTokenizer:
    eos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["add_generation_prompt"] is True
        assert kwargs["enable_thinking"] is False
        return "|".join(message["content"] for message in messages) + "|assistant"

    def __call__(self, value, return_tensors=None, add_special_tokens=False):
        import torch
        ids = [ord(ch) for ch in value]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones((1, len(ids)), dtype=torch.long)}
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(int(item)) for item in ids if not skip_special_tokens or int(item) != self.eos_token_id)


def test_rag_prompt_truncates_last_passage_to_budget_and_records_ids():
    tokenizer = FakeTokenizer()
    request = GenerationRequest("question", "rag_dense", (Passage("d1", "short"), Passage("d2", "x" * 400)))
    base = PromptBuilder(tokenizer, 1000, 5).build(request)
    budget = max(base.prompt_tokens - 50, 100)
    result = PromptBuilder(tokenizer, budget, 5).build(request)
    assert result.prompt_tokens <= budget
    assert result.included_doc_ids == ["d1", "d2"]
    assert result.truncated


def test_question_and_instructions_must_fit_and_llm_only_has_no_evidence():
    tokenizer = FakeTokenizer()
    request = GenerationRequest("question", "llm_only")
    with pytest.raises(ValueError, match="question"):
        PromptBuilder(tokenizer, 3, 5).build(request)
    result = PromptBuilder(tokenizer, 1000, 5).build(request)
    assert result.included_doc_ids == []
    assert "Ngữ cảnh" not in result.rendered


def test_oracle_marker_cannot_enter_regular_mode():
    with pytest.raises(ValueError, match="Oracle"):
        GenerationRequest("q", "rag_bm25", (Passage("d1", "secret", source="oracle"),))
    with pytest.raises(ValueError, match="only gold"):
        GenerationRequest("q", "rag_oracle", (Passage("d1", "retrieved"),))
