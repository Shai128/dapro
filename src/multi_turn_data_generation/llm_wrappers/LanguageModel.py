
from typing import Union, List

from src.multi_turn_data_generation.config.config import Model


class LanguageModel:
    def __init__(self, model_name):
        self.model_name = Model(model_name)

    def batched_generate(self, prompts_list: list, max_n_tokens: int, temperature: float, batch_size: Union[int, None]=None, **kwargs) -> List[str]:
        """
        Generates responses for a batch of prompts using a language model.
        """
        raise NotImplementedError

    def shutdown(self):
        pass


