from typing import Union

import torch

from src.safety_evaluation.reward_functions.abstract_reward import BanditsRewardFunction


class RewardByProbabilityDiff(BanditsRewardFunction):

    def __init__(self, probability_est: Union[torch.Tensor, None]):
        super().__init__()
        self.probability_est = probability_est

    def get_reward(self, arm_idx: int, draw_number: int, **kwargs) -> torch.Tensor:
        if self.probability_est is None:
            assert 'probability_est' in kwargs
            self.probability_est = kwargs['probability_est']
        assert draw_number >= 1
        probability_est = self.probability_est
        prev_event_prob = probability_est[arm_idx, draw_number - 1]
        new_event_prob = probability_est[arm_idx, draw_number]
        # samplings_left = max_c[sample_idx] - temp_c
        # option 2: (new_event_prob - prev_event_prob) / 2 * (1/ samplings_left)
        reward = (new_event_prob - prev_event_prob) / 2 + 0.5
        return reward

    def get_rewards(self, arm_idx: torch.Tensor, draw_number: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.probability_est is None:
            assert 'probability_est' in kwargs
            self.probability_est = kwargs['probability_est']
        probability_est = self.probability_est
        prev_event_prob = probability_est[arm_idx, draw_number - 1]
        new_event_prob = probability_est[arm_idx, draw_number]
        # samplings_left = max_c[sample_idx] - temp_c
        # option 2: (new_event_prob - prev_event_prob) / 2 * (1/ samplings_left)
        rewards = (new_event_prob - prev_event_prob) / 2 + 0.5
        return rewards

    def get_reward_upper_bound(self) -> float:
        return 1

    @property
    def name(self) -> str:
        return "reward_by_probability_diff"
