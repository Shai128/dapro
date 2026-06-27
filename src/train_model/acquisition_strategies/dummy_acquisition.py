

import torch
from torch import nn

from src.train_model.acquisition_strategies.acquisition_strategy import AcquisitionStrategy


class DummyAcquisition(AcquisitionStrategy):
    def __init__(self, device: torch.device = torch.device('cpu')):
        super().__init__("dummy", device=device)

    def _predict_probs(self, model: nn.Module, batch_x: torch.Tensor, times) -> torch.Tensor:
        raise Exception("Not implemented")


    def score(self, model, dataset, pool_indices, batch_size=64):
        raise Exception("Not implemented")