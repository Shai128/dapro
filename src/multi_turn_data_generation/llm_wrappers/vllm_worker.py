# vllm_worker.py
import torch
from vllm import LLM, SamplingParams

from src.multi_turn_data_generation.config.config import HF_MODEL_NAMES


def model_worker(model_name, task_queue, result_queue, device):
    model_path = HF_MODEL_NAMES[model_name]
    engine = LLM(
        model=model_path,
        dtype=torch.bfloat16,
        device=device,
        trust_remote_code=True,
    )

    while True:
        task = task_queue.get()
        if task == "STOP":
            break

        prompts_list, max_n_tokens, temperature, sampling_kwargs = task
        params = SamplingParams(max_tokens=max_n_tokens, temperature=temperature, **sampling_kwargs)
        outputs = engine.generate(prompts_list, sampling_params=params, use_streaming=False)

        results = [output.outputs[0].text for output in outputs]
        result_queue.put(results)
