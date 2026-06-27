import json
import os
import random
import time
import traceback
import multiprocessing as mp
from typing import List, Union, Dict, Optional, Any

import torch
from safetensors import SafetensorError
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.multi_turn_data_generation.config.config import HF_MODEL_NAMES
from src.multi_turn_data_generation.llm_wrappers.LanguageModel import LanguageModel
from src.multi_turn_data_generation.utils.loggers import logger


def get_hf_key():
    with open("./config/keys.json", "r") as f:
        keys = json.load(f)
        key = keys["hf_key"]
    return key


os.environ["HF_TOKEN"] = get_hf_key()


def hf_login():
    # key = get_hf_key()
    # login(token=key, add_to_git_credential=False)
    pass


# --- TOP LEVEL WORKER FUNCTION FOR MULTIPROCESSING ---
def _hf_worker_process(init_kwargs, generate_kwargs, return_dict):
    """Runs the HFLLM generation completely isolated from the main process."""
    try:
        # Recreate the model inside the fresh subprocess
        model = HFLLMProcess(**init_kwargs)
        # Execute the actual generation
        responses = model._core_batched_generate(**generate_kwargs)
        # Return results via IPC dictionary
        return_dict['responses'] = responses
    except Exception as e:
        return_dict['error'] = str(e)
        return_dict['traceback'] = traceback.format_exc()


class HFLLMProcess(LanguageModel):

    def __init__(self, model_name, dtype=torch.bfloat16, device='cpu', lazy_load=False, **tokenizer_kwargs):
        super().__init__(model_name)

        # 1. SAVE KWARGS: We need these to recreate the class inside the subprocess
        self.init_kwargs = {
            'model_name': model_name,
            'dtype': dtype,
            'device': device,
            'lazy_load': False,  # Force false in worker; OS handles destruction
        }
        self.init_kwargs.update(tokenizer_kwargs)

        model_path = HF_MODEL_NAMES[self.model_name]
        self.tok = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
        self.lazy_load = lazy_load
        self.device = device

        self.model = None
        self.model_path = model_path
        self.dtype = dtype

        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

    def load_model(self):
        max_retries = 6
        base_delay = 5

        for attempt in range(max_retries):
            try:
                self.load_model_aux()
                break
            except (SafetensorError, OSError, ValueError) as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed to load model: {e}")
                self.shutdown()
                if attempt == max_retries - 1:
                    raise e
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
            batch_size: Optional[int] = 64,
            destroy_on_finish: bool = True,
            **kwargs: Any,
    ) -> List[str]:
        """
        PUBLIC API: Spawns an isolated process, scripts generation, and terminates.
        Your main loop calls this, but the main loop never touches the GPU.
        """
        # Force 'spawn' method to ensure a clean CUDA context
        ctx = mp.get_context('spawn')
        print(f"batch_size 1: {batch_size}")
        manager = ctx.Manager()
        return_dict = manager.dict()

        generate_kwargs = {
            'prompts_list': prompts_list,
            'max_n_tokens': max_n_tokens,
            'temperature': temperature,
            'batch_size': batch_size,
            **kwargs
        }

        # Launch the isolated worker
        p = ctx.Process(target=_hf_worker_process, args=(self.init_kwargs, generate_kwargs, return_dict))
        p.start()
        p.join()  # Script pauses here until the subprocess is completely finished and dead

        if 'error' in return_dict:
            print(f"Subprocess Traceback:\n{return_dict.get('traceback', '')}")
            raise RuntimeError(f"HF Generation subprocess failed: {return_dict['error']}")

        return return_dict['responses']

    def _core_batched_generate(
            self,
            prompts_list: List[Union[str, List[Dict[str, str]]]],
            max_n_tokens: int,
            temperature: float,
            batch_size: Optional[int] = 64,
            **kwargs: Any,
    ) -> List[str]:
        """
        INTERNAL API: This only scripts inside the isolated subprocess.
        """
        if self.model is None:
            self.load_model()

        batch_size = batch_size if batch_size is not None else len(prompts_list)
        print(f"batch_size2: {batch_size}")
        responses: List[str] = []

        for start in range(0, len(prompts_list), batch_size):
            batch = prompts_list[start: start + batch_size]
            texts: List[str] = []
            for item in batch:
                if isinstance(item, str):
                    texts.append(item)
                else:
                    texts.append("".join(m["content"] for m in item) + "\n\n### RESPONSE BEGINS HERE:\n")

            if self.tok.pad_token is None:
                self.tok.pad_token = self.tok.eos_token
            self.tok.padding_side = "left"

            inputs = self.tok(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                return_length=True,
            ).to(self.model.device)
            prompt_lengths = inputs["length"]

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

            for i, gen_ids in enumerate(out_ids):
                gen_only = gen_ids[prompt_lengths[i].item():]
                resp = self.tok.decode(gen_only, skip_special_tokens=True).strip()
                responses.append(resp)

        # Notice we removed the manual shutdown here. The OS takes care of it.
        return responses

    def shutdown(self):
        """Kept as a dummy function so your existing scripts don't crash if they call it."""
        pass