from typing import List, Tuple, Dict, Any

from src.multi_turn_data_generation.utils.data_types import SampleId, JudgeInput, JudgeOutput


class BatchJudgeBase:
    def __init__(self, args):
        self.max_n_tokens = args.judge_max_n_tokens
        self.temperature = args.judge_temperature
        self.judge_name = args.judge_model

    def get_score(self, judge_input: Dict[SampleId, JudgeInput]) -> Dict[SampleId, JudgeOutput]:
        raise NotImplementedError
