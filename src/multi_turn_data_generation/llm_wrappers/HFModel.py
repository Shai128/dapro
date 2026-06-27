import json
import os
import random
import time
from typing import List, Union, Dict, Optional, Any
import torch
from safetensors import SafetensorError
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

from src.multi_turn_data_generation.config.config import HF_MODEL_NAMES
from src.multi_turn_data_generation.llm_wrappers.LanguageModel import LanguageModel
from src.multi_turn_data_generation.utils.loggers import logger


# Set this BEFORE loading any models

def get_hf_key():
    with open("./config/keys.json", "r") as f:
        keys = json.load(f)
        key = keys["hf_key"]
    return key


os.environ["HF_TOKEN"] = get_hf_key()  # Replace 'key' with your actual token variable


def hf_login():
    pass
    """Performs Hugging Face login safely."""
    # token = HfFolder.get_token()
    # key = get_hf_key()
    # login(token=key, add_to_git_credential=False)
    # if token is None:
    #     print("⚠️ Not logged in yet. Proceeding to login...")
    #     key = get_hf_key()
    #     login(token=key, add_to_git_credential=False)
    # else:
    #     print("✅ Already logged in to Hugging Face.")


class HFLLM(LanguageModel):

    def __init__(self, model_name, dtype=torch.bfloat16, device='cpu', lazy_load=False, **tokenizer_kwargs):
        super().__init__(model_name)

        model_path = HF_MODEL_NAMES[self.model_name]
        self.tok = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
        self.lazy_load = lazy_load
        self.device = device

        self.model = None
        self.model_path = model_path
        self.dtype = dtype
        # if hasattr(self.model.config, "sliding_window"):
        #     self.model.config.sliding_window = None
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

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

    def load_model_aux(self):
        hf_login()
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            low_cpu_mem_usage=True,
            dtype=self.dtype,
            device_map=self.device,
        )
        self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False)
        logger.info(f"Model {self.model_name} uses device {self.device}")
        self.model.eval()

    def batched_generate(
            self,
            prompts_list: List[Union[str, List[Dict[str, str]]]],
            max_n_tokens: int,
            temperature: float,
            batch_size: Optional[int] = None,
            destroy_on_finish: bool = True,
            **kwargs: Any,
    ) -> List[str]:
        # return ['{"prompt": "Shai", "improvement": "Shai"}'] * len(prompts_list)

        """
        Generate in batches, stripping out the input-prompt from each response.

        prompts_list: each element is either
          • a str (already flattened), or
          • a List[{"role": str, "content": str}] which we will flatten by concatenating "content".
        """
        if self.model is None:
            self.load_model()
        # default to processing the entire list in one go
        batch_size = batch_size or len(prompts_list)
        responses: List[str] = []

        for start in range(0, len(prompts_list), batch_size):
            batch = prompts_list[start: start + batch_size]

            # 1) Flatten each element to a single prompt string
            texts: List[str] = []
            for item in batch:
                if isinstance(item, str):
                    texts.append(item)
                else:
                    # assume List[{"role":..., "content":...}]
                    texts.append("".join(m["content"] for m in item) + "\n\n### RESPONSE BEGINS HERE:\n")

            if self.tok.pad_token is None:
                self.tok.pad_token = self.tok.eos_token

            self.tok.padding_side = "left"

            # 2) Tokenize and move to device
            inputs = self.tok(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                return_length=True,  # <-- get prompt lengths in the batch
            ).to(self.model.device)
            prompt_lengths = inputs["length"]  # a torch.Tensor of shape (batch_size,)

            # 3) Generate
            generate_kwargs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "max_new_tokens": max_n_tokens,
                **kwargs,
            }
            if temperature <= 1e-10:
                generate_kwargs['temperature'] = None
                generate_kwargs['top_k'] = None
                generate_kwargs["do_sample"] = False
            else:
                generate_kwargs["temperature"] = temperature
                generate_kwargs["do_sample"] = True

            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                out_ids = self.model.generate(**generate_kwargs)

            # start_idx = inputs["attention_mask"].shape[1]
            # skipped_out_ids = out_ids[:, start_idx:]

            # 4) Decode and strip out the original prompt
            for i, gen_ids in enumerate(out_ids):
                gen_only = gen_ids[prompt_lengths[i].item():]  # strip the prompt token IDs
                resp = self.tok.decode(gen_only, skip_special_tokens=True).strip()
                # logger.debug(f"model input: {texts[i]}\n response: {resp}")
                responses.append(resp)
        if self.lazy_load and destroy_on_finish:
            self.shutdown()
        return responses

    def shutdown(self):
        print("shutting down hfllm model")
        del self.model
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()
