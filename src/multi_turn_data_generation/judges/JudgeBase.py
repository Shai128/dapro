from typing import List, Tuple, Dict, Any


class JudgeBase:
    def __init__(self, args):
        self.max_n_tokens = args.judge_max_n_tokens
        self.temperature = args.judge_temperature
        self.judge_name = args.judge_model
        self.goal = args.goal
        self.target_str = args.target_str


    def score(self, attack_prompt_list, target_response_list) -> List[Tuple[int, Dict[str, Any]]]:
        raise NotImplementedError

