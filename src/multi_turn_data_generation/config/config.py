from enum import Enum

from typing import Dict

VICUNA_PATH = "/home/pchao/vicuna-13b-v1.5"
LLAMA_PATH = "/home/pchao/Llama-2-7b-chat-hf"

ATTACK_TEMP = 1.
TARGET_TEMP = 0.
ATTACK_TOP_P = 0.9
TARGET_TOP_P = 1.


## MODEL PARAMETERS ##
class Model(Enum):
    vicuna = "vicuna-13b-v1.5"
    llama_2 = "llama-2-7b-chat-hf"
    gpt_3_5 = "gpt-3.5-turbo-1106"
    gpt_4 = "gpt-4-0125-preview"
    claude_1 = "claude-instant-1.2"
    claude_2 = "claude-2.1"
    gemini = "gemini-pro"
    mixtral = "mixtral"
    llama_32_1B = 'llama-3.2-1B'
    llama_31_8B = 'llama-3.1-8B'
    llama_32_3B_instruct = 'llama-32-3B-instruct'
    llama_31_8B_instruct = 'llama-3.1-8B-instruct'
    mistral_7B_instruct = 'mistral-7B‑instruct'
    mistral_8x7B_instruct = 'mistral-8x7B‑instruct'
    llama_2_13B_chat = 'llama-2-13B-chat'
    llama_guard_3_1B = 'llama-guard-3-1B'
    mixtral_8x22B = 'mixtral-8x22B-v0.1'
    qwen25_7b_instruct = 'qwen25-7b-instruct'
    qwen25_14b_instruct = 'qwen25-14b-instruct'
    phi_4 = 'phi_4'
    mini_phi_4_instruct = 'mini_phi_4_instruct'
    gemma2_9b_it = 'gemma2_9b_it'
    gemma3_4b_it = 'gemma3_4b_it'
    gemma3_12b_it = 'gemma3_12b_it'
    qwen3_4b = 'qwen3_4b'
    qwen3_8b = 'qwen3_8b'
    qwen3_30b = 'qwen3_30b'
    neural_chat_7b = 'neural_chat_7b'
    gpt_oss_20b = 'gpt_oss_20b'


HF_MODEL_NAMES: Dict[Model, str] = {
    Model.llama_2: "meta-llama/Llama-2-7b-chat-hf",
    Model.vicuna: "lmsys/vicuna-13b-v1.5",
    Model.mixtral: "mistralai/Mixtral-8x7B-Instruct-v0.1",
    Model.llama_32_1B: 'meta-llama/Llama-3.2-1B',
    Model.llama_31_8B: 'meta-llama/Llama-3.1-8B',
    Model.llama_32_3B_instruct: 'meta-llama/Llama-3.2-3B-Instruct',
    Model.llama_31_8B_instruct: 'meta-llama/Llama-3.1-8B-Instruct',
    Model.mistral_7B_instruct: 'mistralai/Mistral-7B-Instruct-v0.3',
    Model.mistral_8x7B_instruct: 'mistralai/Mixtral-8x7B-Instruct-v0.1',
    Model.llama_2_13B_chat: 'meta-llama/Llama-2-13b-chat-hf',
    Model.llama_guard_3_1B: 'meta-llama/Llama-Guard-3-1B',
    Model.mixtral_8x22B: 'mistralai/Mixtral-8x22B-v0.1',
    Model.qwen25_7b_instruct: 'Qwen/Qwen2.5-7B-Instruct',
    Model.qwen25_14b_instruct: 'Qwen/Qwen2.5-14B-Instruct',
    Model.phi_4: 'microsoft/phi-4',
    Model.mini_phi_4_instruct: 'microsoft/Phi-4-mini-instruct',
    Model.gemma2_9b_it: 'google/gemma-2-9b-it',
    Model.gemma3_4b_it: 'google/gemma-3-4b-it',
    Model.gemma3_12b_it: 'google/gemma-3-12b-it',
    Model.qwen3_4b: 'Qwen/Qwen3-4B',
    Model.qwen3_8b: 'Qwen/Qwen3-8B',
    Model.qwen3_30b: 'Qwen/Qwen3-30B-A3B',
    Model.neural_chat_7b: "Intel/neural-chat-7b-v3-1",
    Model.gpt_oss_20b: "openai/gpt-oss-20b",
}

MODEL_NAMES = [model.value for model in Model]

FASTCHAT_TEMPLATE_NAMES: Dict[Model, str] = {model: model.name for model in Model}

TOGETHER_MODEL_NAMES: Dict[Model, str] = {
    Model.llama_2: "together_ai/togethercomputer/llama-2-7b-chat",
    Model.vicuna: "together_ai/lmsys/vicuna-13b-v1.5",
    Model.mixtral: "together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1"
}

API_KEY_NAMES: Dict[Model, str] = {
    Model.gpt_3_5: "OPENAI_API_KEY",
    Model.gpt_4: "OPENAI_API_KEY",
    Model.claude_1: "ANTHROPIC_API_KEY",
    Model.claude_2: "ANTHROPIC_API_KEY",
    Model.gemini: "GEMINI_API_KEY",
    Model.vicuna: "TOGETHER_API_KEY",
    Model.llama_2: "TOGETHER_API_KEY",
    Model.mixtral: "TOGETHER_API_KEY",
}

LITELLM_TEMPLATES: Dict[Model, dict] = {
    Model.vicuna: {"roles": {
        "system": {"pre_message": "", "post_message": " "},
        "user": {"pre_message": "USER: ", "post_message": " ASSISTANT:"},
        "assistant": {
            "pre_message": "",
            "post_message": "",
        },
    },
        "post_message": "</s>",
        "initial_prompt_value": "",
        "eos_tokens": ["</s>"]
    },
    Model.llama_2: {"roles": {
        "system": {"pre_message": "[INST] <<SYS>>\n", "post_message": "\n<</SYS>>\n\n"},
        "user": {"pre_message": "", "post_message": " [/INST]"},
        "assistant": {"pre_message": "", "post_message": ""},
    },
        "post_message": " </s><s>",
        "initial_prompt_value": "",
        "eos_tokens": ["</s>", "[/INST]"]
    },
    Model.mixtral: {"roles": {
        "system": {
            "pre_message": "[INST] ",
            "post_message": " [/INST]"
        },
        "user": {
            "pre_message": "[INST] ",
            "post_message": " [/INST]"
        },
        "assistant": {
            "pre_message": " ",
            "post_message": "",
        }
    },
        "post_message": "</s>",
        "initial_prompt_value": "<s>",
        "eos_tokens": ["</s>", "[/INST]"]
    }
}
