import torch
import pytest

from src.generation.hf_generator import HFGenerator, _abstained, clean_completion
from src.generation.prompt_builder import PromptBuilder
from src.generation.types import GenerationRequest
from test_generation_prompts import FakeTokenizer


class FakeModel:
    def __init__(self):
        self.parameter = torch.nn.Parameter(torch.zeros(1))

    def eval(self):
        return self

    def parameters(self):
        return iter([self.parameter])

    def generate(self, input_ids, **kwargs):
        return type("Result", (), {"sequences": torch.cat([input_ids, torch.tensor([[65, 2]])], dim=1)})()


def test_cleanup_and_malformed_thinking_flag():
    assert clean_completion("<think>private</think>Answer") == ("Answer", False)
    assert clean_completion("<think>unfinished") == ("<think>unfinished", True)
    assert _abstained("Không có, thông tin!")
    assert _abstained("Khong co thong tin")


def test_config_rejects_unimplemented_sampling_and_retry_policy():
    config = {"model": {}, "generation": {"max_new_tokens": 4, "do_sample": True},
              "run": {"oom_retry": 1, "batch_size": 1, "checkpoint_every": 1, "fail_on_error": True}}
    with pytest.raises(ValueError, match="greedy"):
        HFGenerator._validate_config(config)


def test_injected_generator_decodes_only_new_tokens_and_closes_idempotently():
    config = {"model": {"revision": "abc"}, "generation": {"max_new_tokens": 4},
        "prompt": {"version": "vi_qa_v1", "max_input_tokens": 1000, "max_context_docs": 5},
        "evaluation": {"cleanup_version": "cl_v1", "normalization": "vi_unicode_v1",
                       "metrics": ["exact_match", "f1_token", "rouge_l"]}, "run": {"oom_retry": 1}}
    generator = HFGenerator(config, model=FakeModel(), tokenizer=FakeTokenizer(), torch_module=torch)
    generator.prompt_builder = PromptBuilder(generator.tokenizer, 1000, 5)
    output = generator.generate(GenerationRequest("q", "llm_only"))
    assert output.answer == "A"
    assert output.num_llm_calls == 1
    assert output.finish_reason == "eos"
    generator.close()
    generator.close()
