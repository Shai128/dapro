from typing import Union

import torch

from src.safety_evaluation.reward_functions.abstract_reward import BanditsRewardFunction



class RewardByProbability(BanditsRewardFunction):

    def __init__(self, probability_est: Union[torch.Tensor, None]):
        super().__init__()
        self.probability_est = probability_est

    def get_reward(self, arm_idx: int, draw_number: int, **kwargs) -> torch.Tensor:
        probability_est = self.probability_est
        new_event_prob = probability_est[arm_idx, draw_number]
        reward = new_event_prob
        return reward

    def get_rewards(self, arm_idx: torch.Tensor, draw_number: torch.Tensor, **kwargs) -> torch.Tensor:
        probability_est = self.probability_est
        new_event_prob = probability_est[arm_idx, draw_number, draw_number]
        rewards = new_event_prob
        return rewards

    def get_reward_upper_bound(self) -> float:
        return 1

    @property
    def name(self) -> str:
        return "reward_by_probability"
