"""Hugging Face causal-LM generator; heavyweight imports are deferred to init."""

import gc
import re
import time

from src.generation.base_generator import BaseGenerator
from src.evaluation.qa_metrics import normalize_answer
from src.generation.prompt_builder import PromptBuilder
from src.generation.types import GenerationOutput

THINKING_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def clean_completion(raw):
    opens = len(re.findall(r"<think\s*>", raw, flags=re.IGNORECASE))
    closes = len(re.findall(r"</think\s*>", raw, flags=re.IGNORECASE))
    malformed = opens != closes
    cleaned = THINKING_BLOCK.sub("", raw).strip()
    return cleaned, malformed


def _abstained(answer):
    normalized = normalize_answer(answer)
    return "không có thông tin" in normalized or "khong co thong tin" in normalized


class HFGenerator(BaseGenerator):
    def __init__(self, config, *, model=None, tokenizer=None, torch_module=None, revision_resolver=None):
        self._validate_config(config)
        self.config = config
        self._closed = False
        if model is not None and tokenizer is not None:
            if torch_module is None:
                import torch as torch_module
            self.torch, self.model, self.tokenizer = torch_module, model, tokenizer
            self.revision = config["model"].get("resolved_revision", config["model"].get("revision"))
        else:
            import torch
            from huggingface_hub import model_info
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            self.torch = torch
            model_cfg = config["model"]
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as exc:
                raise RuntimeError("NF4 inference requires the bitsandbytes dependency from requirements.txt.") from exc
            if model_cfg.get("device", "cuda") == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA is required by the generation configuration.")
            model_name = model_cfg["name_or_path"]
            revision = model_cfg["revision"]
            info = revision_resolver(model_name, revision) if revision_resolver else model_info(model_name, revision=revision)
            self.revision = getattr(info, "sha", info)
            if not isinstance(self.revision, str) or not self.revision:
                raise RuntimeError("Could not resolve model revision to a commit hash.")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=self.revision)
            quant = BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_quant_type=model_cfg.get("quantization", "nf4"),
                bnb_4bit_compute_dtype=getattr(torch, model_cfg.get("compute_dtype", "float16")),
                bnb_4bit_use_double_quant=model_cfg.get("double_quantization", True))
            if model_cfg.get("device", "cuda") != "cuda":
                raise ValueError("NF4 generation configuration currently requires device: cuda.")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, revision=self.revision, quantization_config=quant,
                device_map={"": 0})
        self.model.eval()
        prompt_cfg = config["prompt"]
        self.prompt_builder = PromptBuilder(self.tokenizer, prompt_cfg["max_input_tokens"],
                                            prompt_cfg["max_context_docs"], prompt_cfg["version"],
                                            prompt_cfg.get("answer_style", "concise"))

    @staticmethod
    def _validate_config(config):
        generation, run, model_cfg = config["generation"], config["run"], config["model"]
        if generation.get("do_sample", False) is not False or generation.get("enable_thinking", False) is not False:
            raise ValueError("This baseline requires greedy decoding with thinking disabled.")
        max_new = generation.get("max_new_tokens")
        if isinstance(max_new, bool) or not isinstance(max_new, int) or max_new <= 0:
            raise ValueError("max_new_tokens must be a positive integer.")
        if model_cfg.get("quantization", "nf4") != "nf4":
            raise ValueError("This baseline supports only NF4 quantization.")
        if model_cfg.get("compute_dtype", "float16") != "float16":
            raise ValueError("This baseline fixes the NF4 compute dtype to float16.")
        if config["prompt"].get("version") != "vi_qa_v1":
            raise ValueError("Unsupported prompt template version.")
        if config["evaluation"].get("normalization") != "vi_unicode_v1":
            raise ValueError("Unsupported QA normalization version.")
        if config["evaluation"].get("cleanup_version") != "cl_v1":
            raise ValueError("Unsupported completion cleanup version.")
        if config["evaluation"].get("metrics") != ["exact_match", "f1_token", "rouge_l"]:
            raise ValueError("This baseline computes exact_match, f1_token, and rouge_l.")
        if run.get("oom_retry", 1) != 1 or run.get("batch_size", 1) != 1 or run.get("checkpoint_every", 1) != 1:
            raise ValueError("Generation is fixed to one OOM retry, one question per call, and per-answer checkpoints.")
        if run.get("fail_on_error", True) is not True:
            raise ValueError("Generation runs must stop on errors.")

    def generate(self, request):
        if self._closed:
            raise RuntimeError("Generator is closed.")
        built = self.prompt_builder.build(request)
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in self.tokenizer(
            built.rendered, return_tensors="pt", add_special_tokens=False).items()}
        using_cuda = torch_cuda_available(self.torch) and str(device).startswith("cuda")
        if using_cuda:
            self.torch.cuda.synchronize()
            self.torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        max_new = self.config["generation"]["max_new_tokens"]
        retries = self.config["run"].get("oom_retry", 1)
        retry_count = 0
        try:
            with self.torch.inference_mode():
                for attempt in range(retries + 1):
                    try:
                        result = self.model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                            return_dict_in_generate=True, use_cache=True,
                            eos_token_id=self.tokenizer.eos_token_id)
                        break
                    except RuntimeError as exc:
                        if "out of memory" not in str(exc).lower() or attempt >= retries:
                            raise
                        if torch_cuda_available(self.torch):
                            self.torch.cuda.empty_cache()
                        retry_count += 1
                sequence = result.sequences[0]
                prompt_len = inputs["input_ids"].shape[-1]
                new_tokens = sequence[prompt_len:]
                raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                completion_count = int(new_tokens.shape[-1])
                eos = self.tokenizer.eos_token_id is not None and self.tokenizer.eos_token_id in new_tokens.tolist()
                finish = "eos" if eos else "output_limit" if completion_count >= max_new else "stopped"
                if using_cuda:
                    self.torch.cuda.synchronize()
                    peak_memory = self.torch.cuda.max_memory_allocated() / 2**20
                else:
                    peak_memory = None
        finally:
            del inputs
        answer, malformed = clean_completion(raw)
        return GenerationOutput(raw, answer, built.prompt_tokens, completion_count, finish,
            time.perf_counter() - started, abstained=_abstained(answer), truncated=built.truncated,
            included_doc_ids=built.included_doc_ids,
            included_passages=[{"doc_id": p.doc_id, "text": p.text}
                               for p in request.passages if p.doc_id in built.included_doc_ids],
            cleanup_version=self.config["evaluation"].get("cleanup_version", "cl_v1"),
            malformed_thinking=malformed, peak_cuda_memory_mib=peak_memory, oom_retries=retry_count)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch_cuda_available(self.torch):
            self.torch.cuda.empty_cache()


def torch_cuda_available(torch_module):
    return hasattr(torch_module, "cuda") and torch_module.cuda.is_available()
