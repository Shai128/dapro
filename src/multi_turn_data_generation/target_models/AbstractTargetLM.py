import abc
from typing import Dict

from src.multi_turn_data_generation.utils.data_types import SampleId


class AbstractTargetLM(abc.ABC):

    def __init__(self, ):
        pass

    @abc.abstractmethod
    def get_response(self, prompts: Dict[SampleId, str], **kwargs) -> Dict[SampleId, str]:
        pass
