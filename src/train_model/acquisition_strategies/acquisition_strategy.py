
from abc import ABC, abstractmethod
from typing import List,  Dict

import torch
from torch import nn



class AcquisitionStrategy(ABC):
    def __init__(self, strategy_name :str, device: torch.device = torch.device('cpu')):
        self.device = device
        self.strategy_name = strategy_name

    @abstractmethod
    def score(self, model: nn.Module, dataset: torch.utils.data.Dataset, pool_indices: List[int],
              batch_size: int = 64) -> Dict[int, float]:
        pass

    def select(self, model: nn.Module, dataset: torch.utils.data.Dataset, pool_indices: List[int],
               k: int, batch_size: int = 64) -> List[int]:
        scores = self.score(model, dataset, pool_indices, batch_size=batch_size)
        selected = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [idx for idx, _ in selected[:k]]

    @property
    def name(self) -> str:
        return self.strategy_name
