import random
import time
import logging
import traceback
import queue
import multiprocessing as mp
from typing import List, Union, Any, Optional

import torch
from safetensors import SafetensorError

from vllm import LLM, SamplingParams

from src.multi_turn_data_generation.config.config import HF_MODEL_NAMES
from src.multi_turn_data_generation.llm_wrappers.HFModel import hf_login
from src.multi_turn_data_generation.llm_wrappers.LanguageModel import LanguageModel
from src.multi_turn_data_generation.utils.loggers import logger


# --- TOP LEVEL PERSISTENT WORKER FUNCTION ---
def _vllm_persistent_worker(init_kwargs, task_queue, result_queue):
    """
    Runs continuously in an isolated process.
    Initializes the model ONCE, then waits for tasks via the queue.
    """
    try:
        # Initialize the model exactly once
        model = VLLMModelProcess(**init_kwargs)

        while True:
            # Wait for the next generation request
            task = task_queue.get()

            # A 'None' task is our poison pill to gracefully shut down
            if task is None:
                break

            # Execute generation
            responses = model._core_batched_generate(**task)

            # Send results back
            result_queue.put({'responses': responses})

    except Exception as e:
        # Catch initialization or generation errors and send them back
        result_queue.put({'error': str(e), 'traceback': traceback.format_exc()})


class VLLMModelProcess(LanguageModel):

    def __init__(self, model_name: str, dtype=torch.bfloat16, n_ctx: int = 2048, device='cpu', lazy_load=False,
                 **vllm_kwargs):
        super().__init__(model_name)

        # SAVE KWARGS to recreate the class inside the subprocess
        self.init_kwargs = {
            'model_name': model_name,
            'dtype': dtype,
            'n_ctx': n_ctx,
            'device': device,
            'lazy_load': False,
        }
        self.init_kwargs.update(vllm_kwargs)

        model_path = HF_MODEL_NAMES[self.model_name]
        self.lazy_load = lazy_load
        self.model_path = model_path
        self.dtype = dtype
        self.n_ctx = n_ctx
        self._llm = None

        # State tracking for the persistent worker
        self._worker_process = None
        self._task_queue = None
        self._result_queue = None

        logger.info(f"Initialized vLLM wrapper for {model_path} dtype={dtype}")

    def _start_worker(self):
        """Spawns the background process and sets up communication queues."""
        ctx = mp.get_context('spawn')
        self._task_queue = ctx.Queue()
        self._result_queue = ctx.Queue()

        self._worker_process = ctx.Process(
            target=_vllm_persistent_worker,
            args=(self.init_kwargs, self._task_queue, self._result_queue)
        )
        self._worker_process.start()

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
        PUBLIC API: Sends generation tasks to the persistent worker.
        """
        # 1. Start the worker if it doesn't exist or was killed
        if self._worker_process is None or not self._worker_process.is_alive():
            logger.info("Starting background vLLM worker process...")
            self._start_worker()

        generate_kwargs = {
            'prompts_list': prompts_list,
            'max_n_tokens': max_n_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k,
            'do_sample': do_sample,
            **kwargs
        }

        # 2. Send the task to the worker
        self._task_queue.put(generate_kwargs)

        # 3. Wait for the result safely (polling prevents hanging if Slurm SIGKILLs the subprocess)
        result_dict = None
        while True:
            try:
                result_dict = self._result_queue.get(timeout=1.0)
                break
            except queue.Empty:
                if not self._worker_process.is_alive():
                    self.shutdown()
                    raise RuntimeError("vLLM Subprocess died unexpectedly (likely SIGKILLed by OS due to OOM).")

        # 4. Handle errors inside the worker
        if 'error' in result_dict:
            print(f"Subprocess Traceback:\n{result_dict.get('traceback', '')}")
            self.shutdown()
            raise RuntimeError(f"vLLM Generation subprocess failed: {result_dict['error']}")

        responses = result_dict['responses']

        # 5. Shut down only if explicitly requested
        if destroy_on_finish:
            self.shutdown()

        return responses

    def shutdown(self):
        """Sends the poison pill to the worker and physically terminates it."""
        if self._worker_process is not None and self._worker_process.is_alive():
            logger.info("Sending poison pill to shut down vLLM worker...")
            self._task_queue.put(None)  # The poison pill
            self._worker_process.join(timeout=10)

            # Force kill if it's hanging
            if self._worker_process.is_alive():
                self._worker_process.terminate()
                self._worker_process.join()

        self._worker_process = None
        self._task_queue = None
        self._result_queue = None
        logger.info("vLLM worker shutdown complete. VRAM is completely free.")

    def _core_batched_generate(
            self,
            prompts_list: List[Union[str, List[dict]]],
            max_n_tokens: int,
            temperature: float,
            top_p: Optional[float] = None,
            top_k: Optional[int] = None,
            do_sample: Optional[bool] = None,
            **kwargs: Any,
    ) -> List[str]:
        """
        INTERNAL API: This only scripts inside the isolated subprocess.
        """
        if self._llm is None:
            self.load_model()
            print("model loaded successfully2")

        results: List[str] = []
        flat_prompts = [self._flatten_prompt(x) for x in prompts_list]

        sampling = SamplingParams(
            temperature=temperature if (do_sample is None or do_sample) else 0.0,
            top_p=top_p if top_p is not None else 1.0,
            top_k=top_k if top_k is not None else -1,  # Fixed to -1 for vLLM API
            max_tokens=max_n_tokens,
        )

        print("starting to generate (Un-chunked Queue)")
        # Fixed the chunking: Feed the whole queue directly to vLLM
        outputs = self._llm.generate(flat_prompts, sampling_params=sampling)

        for output in outputs:
            out_text = output.outputs[0].text
            results.append(out_text.strip())

        print(f"end of generation")
        return results

    def load_model_aux(self):
        hf_login()
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
        base_delay = 5
        for attempt in range(max_retries):
            try:
                self.load_model_aux()
                break
            except (SafetensorError, OSError, ValueError) as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed to load model: {e}")
                if attempt == max_retries - 1:
                    raise e
                min_sleep = base_delay * (2 ** attempt)
                max_sleep = base_delay * (2 ** (attempt + 1))
                sleep_time = random.uniform(min_sleep, max_sleep)
                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
        print("model loaded successfully")

    def _flatten_prompt(self, item: Union[str, List[dict]]) -> str:
        if isinstance(item, str):
            return item
        if item is None or len(item) == 0:
            prefix = None
        else:
            prefix = "".join([str(m["content"]) if ('content' in m and m is not None) else "" for m in item])
        return prefix + "\n\n### RESPONSE BEGINS HERE:\n"
