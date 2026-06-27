import os
from typing import Union

import litellm
from src.multi_turn_data_generation.config.config import TOGETHER_MODEL_NAMES, LITELLM_TEMPLATES, API_KEY_NAMES, Model
from src.multi_turn_data_generation.llm_wrappers.LanguageModel import LanguageModel
from src.multi_turn_data_generation.utils.common import get_api_key


class APILiteLLM(LanguageModel):
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "ERROR: API CALL FAILED."
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20

    def __init__(self, model_name):
        super().__init__(model_name)
        self.api_key = get_api_key(self.model_name)
        self.litellm_model_name = self.get_litellm_model_name(self.model_name)
        litellm.drop_params = True
        self.set_eos_tokens(self.model_name)

    def get_litellm_model_name(self, model_name):
        if model_name in TOGETHER_MODEL_NAMES:
            litellm_name = TOGETHER_MODEL_NAMES[model_name]
            self.use_open_source_model = True
        else:
            self.use_open_source_model = False
            # if self.use_open_source_model:
            # Output warning, there should be a TogetherAI model name
            # logger.warning(f"Warning: No TogetherAI model name for {model_name}.")
            litellm_name = model_name.value
        return litellm_name

    def set_eos_tokens(self, model_name):
        if self.use_open_source_model:
            self.eos_tokens = LITELLM_TEMPLATES[model_name]["eos_tokens"]
        else:
            self.eos_tokens = []

    def _update_prompt_template(self):
        # We manually add the post_message later if we want to seed the model response
        if self.model_name in LITELLM_TEMPLATES:
            litellm.register_prompt_template(
                initial_prompt_value=LITELLM_TEMPLATES[self.model_name]["initial_prompt_value"],
                model=self.litellm_model_name,
                roles=LITELLM_TEMPLATES[self.model_name]["roles"]
            )
            self.post_message = LITELLM_TEMPLATES[self.model_name]["post_message"]
        else:
            self.post_message = ""

    def batched_generate(self, prompts_list: list, max_n_tokens: int, temperature: float,
                         batch_size: Union[int, None]=None, top_p: float=None,
                         extra_eos_tokens: list[str] = None, **kwargs) -> list[str]:

        eos_tokens = self.eos_tokens

        if extra_eos_tokens:
            eos_tokens.extend(extra_eos_tokens)
        if self.use_open_source_model:
            self._update_prompt_template()

        outputs = litellm.batch_completion(
            model=self.litellm_model_name,
            messages=prompts_list,
            api_key=self.api_key,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_n_tokens,
            num_retries=self.API_MAX_RETRY,
            seed=0,
            stop=eos_tokens,
        )

        responses = [output["choices"][0]["message"].content for output in outputs]

        return responses
