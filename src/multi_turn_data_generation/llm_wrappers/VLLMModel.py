# vllm_wrapper.py
import random
import time
from typing import List, Union, Any, Optional
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

import torch
from safetensors import SafetensorError


import gc

from src.multi_turn_data_generation.config.config import HF_MODEL_NAMES
from src.multi_turn_data_generation.llm_wrappers.LanguageModel import LanguageModel
from src.multi_turn_data_generation.llm_wrappers.VLLMModelProcess import logger


class VLLMModel(LanguageModel):
    """
    Simple wrapper around vLLM's LLM to present a batched_generate(...) compatible API.
    Designed to be a drop-in replacement for HFLLM's batched_generate for the attacker model.
    """

    def __init__(self, model_name: str, dtype=torch.bfloat16, n_ctx: int = 2048, device='cpu', lazy_load=False,
                 **vllm_kwargs):
        super().__init__(model_name)
        model_path = HF_MODEL_NAMES[self.model_name]
        self.lazy_load = lazy_load
        # model_path: local HF repo id or HF hub id matching vLLM expectations
        # dtype: "bfloat16" or "float16"
        # n_ctx: context length for generated tokens and caching
        self.model_path = model_path
        self.dtype = dtype
        self.n_ctx = n_ctx
        self._llm = None
        # additional kwargs passed to LLM constructor (eg. tensor_parallel_size, num_shards, etc.)

        logger.info(f"Initialized vLLM model at {model_path} dtype={dtype}")

    def _flatten_prompt(self, item: Union[str, List[dict]]) -> str:
        if isinstance(item, str):
            return item
        # Otherwise a list of {"role":..., "content":...}
        # Keep same flattening style as HFLLM
        if item is None or len(item) == 0:
            prefix = None
        else:
            prefix = "".join([str(m["content"]) if ('content' in m and m is not None) else "" for m in item])
        return prefix + "\n\n### RESPONSE BEGINS HERE:\n"

    def batched_generate(
            self,
            prompts_list: List[Union[str, List[dict]]],
            max_n_tokens: int,
            temperature: float,
            batch_size: Optional[int] = None,
            top_p: Optional[float] = None,
            top_k: Optional[int] = None,
            do_sample: Optional[bool] = None,
            destroy_on_finish=True,
            **kwargs: Any,
    ) -> List[str]:
        """
        Match signature of HFLLM.batched_generate.
        Uses vLLM to generate completions. Returns only the generated suffixes (no prompt).
        """
        if self._llm is None:
            self.load_model()
            print("model loaded successfully2")

        batch_size = batch_size or len(prompts_list)
        results: List[str] = []

        # vLLM prefers a list of prompt strings
        flat_prompts = [self._flatten_prompt(x) for x in prompts_list]

        # process in chunks to avoid huge single batches
        for start in range(0, len(flat_prompts), batch_size):
            sub_prompts = flat_prompts[start:start + batch_size]

            # Build sampling params; align semantics with HF's do_sample / temperature
            # vLLM's SamplingParams takes temperature, top_p, top_k, max_tokens
            sampling = SamplingParams(
                temperature=temperature if (do_sample is None or do_sample) else 0.0,
                top_p=top_p if top_p is not None else 1.0,
                top_k=top_k if top_k is not None else 0,
                max_tokens=max_n_tokens,
                # you can add other sampling options here, e.g., repetition_penalty, remove_invalid_utf8
            )

            # Call vLLM generate: returns generator of generation results in request order.
            # Each result corresponds to a request and contains outputs; output.text is the generated text.
            print("starting to generate")

            gen_iter = self._llm.generate(sub_prompts, sampling_params=sampling)

            # vLLM returns a generator of 'Generation' objects; iterate and extract text
            for gen in gen_iter:
                # Many vLLM versions expose gen.outputs[0].text; fallback to gen.text if needed.
                try:
                    # vLLM Generation object may include outputs list
                    out_text = gen.outputs[0].text
                except Exception:
                    # older/newer API compatibility fallback
                    out_text = getattr(gen, "text", "")
                logger.debug(f"vllm Generated text: {out_text}")
                # vLLM returns *only* the completion (not the prompt) by default,
                # but be defensive and if prompt included, strip prompt prefix.
                # We will attempt to rstrip whitespace.
                results.append(out_text.strip())
        print(f"end of generation, lazy_load: {self.lazy_load} destroy_on_finish: {destroy_on_finish}")
        if self.lazy_load and destroy_on_finish:
            self.shutdown()
        return results

    def shutdown(self):
        print("shutting down vllm model")
        """Gracefully shutdown vLLM engine (release GPU memory)."""
        try:
            if self._llm is not None:
                del self._llm
                self._llm = None
            try:
                destroy_model_parallel()
            except Exception as e:
                # It's possible this fails if not initialized, but we try anyway
                print(f"Cleanup Note: {e}")

            # 3. Destroy PyTorch distributed process groups
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()

            # 4. Shutdown Ray (if vLLM spawned it)
            try:
                import ray
                if ray.is_initialized():
                    ray.shutdown()
            except ImportError:
                pass

            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            logger.exception(f"Error shutting down vLLM instance. because {e}")

    def load_model_aux(self):
        hf_login()
        # self._llm = LLM(model=self.model_path, dtype=self.dtype, gpu_memory_utilization=0.75)
        self._llm = LLM(
            model=self.model_path,
            dtype=self.dtype,
            gpu_memory_utilization=0.8,
            max_num_seqs=128,  # Lowered from 256 to match your comment; saves KV cache space
            max_model_len=8192,
            # <-- Increased to 8K to accommodate your long prompt
            enforce_eager=True)  # <-- THE FIX: Disables CUDA graph capture which is causing the crash        )

    def load_model(self):
        max_retries = 6
        base_delay = 5  # seconds

        for attempt in range(max_retries):
            try:
                # Your original loading code
                self.load_model_aux()
                break

            except (SafetensorError, OSError, ValueError) as e:
                # Check if it's the specific header error or a generic file access error
                print(f"Attempt {attempt + 1}/{max_retries} failed to load model: {e}")
                self.shutdown()
                if attempt == max_retries - 1:
                    raise e  # Crash if we fail all retries

                # Exponential backoff with jitter (randomness) to desynchronize processes
                min_sleep = base_delay * (2 ** attempt)
                max_sleep = base_delay * (2 ** (attempt + 1))
                sleep_time = random.uniform(min_sleep, max_sleep)
                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
        print("model loaded successfully")
