from src.multi_turn_data_generation.judges.JudgeBase import JudgeBase


class NoJudge(JudgeBase):
    def __init__(self, args):
        super(NoJudge, self).__init__(args)
        self.judge_model = None

    def score(self, prompt_list, response_list):
        return [(1, {}) for _ in prompt_list]
