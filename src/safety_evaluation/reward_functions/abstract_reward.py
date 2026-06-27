import abc

import torch


class BanditsRewardFunction(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def get_reward(self, arm: int, draw_number: int, **kwargs):
        pass

    @abc.abstractmethod
    def get_rewards(self, arms: torch.Tensor, draw_numbers: torch.Tensor, **kwargs) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def get_reward_upper_bound(self) -> float:
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass