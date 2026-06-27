import abc
from typing import Dict

import torch

from src.train_model.models.utils import ModelPrediction


class SurvivalLPBCalibration(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def calibrate(self, x_cal: torch.Tensor, t_tilde_cal: torch.Tensor, model_prediction_cal: ModelPrediction):
        pass

    @abc.abstractmethod
    def get_calibrated_lpb(self, target_taus: torch.Tensor, x: torch.Tensor, model_prediction: ModelPrediction):
        pass

    @abc.abstractmethod
    def compute_metrics(self, model_prediction, target_taus) -> Dict[str, float]:
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

class SurvivalUPBCalibration(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def calibrate(self, x_cal: torch.Tensor, t_tilde_cal: torch.Tensor, model_prediction_cal: ModelPrediction):
        pass

    @abc.abstractmethod
    def get_calibrated_upb(self, target_taus: torch.Tensor, x: torch.Tensor, model_prediction: ModelPrediction):
        pass

    @abc.abstractmethod
    def compute_metrics(self, model_prediction, target_taus) -> Dict[str, float]:
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass