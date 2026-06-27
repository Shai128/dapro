from typing import Dict

from src.multi_turn_data_generation.config.config import TARGET_TEMP, TARGET_TOP_P, FASTCHAT_TEMPLATE_NAMES
from src.multi_turn_data_generation.llm_wrappers.LanguageModel import LanguageModel
from src.multi_turn_data_generation.utils.common import conv_template
from src.multi_turn_data_generation.utils.data_types import SampleId
from src.multi_turn_data_generation.utils.loggers import logger


class TargetLM:
    """
        JailbreakBench class for target language models.
    """

    def __init__(self,
                 model: LanguageModel,
                 category: str,
                 max_n_tokens: int,
                 phase: str,
                 ):

        self.max_n_tokens = max_n_tokens
        self.phase = phase
        self.temperature = TARGET_TEMP
        self.top_p = TARGET_TOP_P

        self.category = category
        self.model = model
        self.model_name = model.model_name
        self.template = FASTCHAT_TEMPLATE_NAMES[self.model_name]
        self.name = model.model_name.name


    def get_response(self, prompts: Dict[SampleId, str], **kwargs) -> Dict[SampleId, str]:
        logger.debug("Starting to generate responses")

        batchsize = len(prompts)
        convs_list = [conv_template(self.template) for _ in range(batchsize)]
        full_prompts = []
        for conv, i in zip(convs_list, prompts):
            conv.append_message(conv.roles[0], prompts[i])
            full_prompts.append(conv.to_openai_api_messages())

        responses = self.model.batched_generate(full_prompts,
                                                max_n_tokens=self.max_n_tokens,
                                                temperature=self.temperature,
                                                top_p=self.top_p
                                                )
        logger.debug("Finished to generate responses")
        responses_dict = {i: res for i, res in zip(prompts, responses)}
        return responses_dict