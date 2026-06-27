from src.multi_turn_data_generation.llm_wrappers.HFModel import HFLLM
from src.multi_turn_data_generation.llm_wrappers.HFModelProcess import HFLLMProcess

try:
    import vllm

    print(f"vLLM version {vllm.__version__} is installed.")
    vllm_available = True
    from llm_models.VLLMModel import VLLMModel

    from llm_models.VLLMModelProcess import VLLMModelProcess
except ImportError as e:
    print(f"vLLM is not installed in this environment, error: {e}.")
    vllm_available = False




def load_hf_model(model_name, dtype, device, lazy_load):
    if lazy_load:
        model = HFLLMProcess(model_name, dtype=dtype, device=device, lazy_load=lazy_load)
    else:
        model = HFLLM(model_name, dtype=dtype, device=device, lazy_load=lazy_load)
    return model


def load_model(model_name, dtype, device, lazy_load):
    if vllm_available:
        if lazy_load:
            model = VLLMModelProcess(model_name, dtype=dtype, device=device, lazy_load=lazy_load)
        else:
            model = VLLMModel(model_name, dtype=dtype, device=device, lazy_load=lazy_load)
    else:
        model = load_hf_model(model_name, dtype, device, lazy_load)
    return model


