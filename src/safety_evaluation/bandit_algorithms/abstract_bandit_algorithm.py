import abc

import torch

from src.safety_evaluation.reward_functions.abstract_reward import BanditsRewardFunction


class BanditAlgorithm:
    def __init__(self, reward_function: BanditsRewardFunction, device: torch.device, ):
        self.device = device
        self.reward_function = reward_function

    @abc.abstractmethod
    def run(self, total_budget: int, n_arms: int, max_draws_per_arm: torch.Tensor, max_time:int, **kwargs) -> torch.Tensor:
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass