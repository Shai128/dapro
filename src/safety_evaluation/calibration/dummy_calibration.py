from typing import Dict

import numpy as np
import torch

from src.safety_evaluation.calibration.abstract_calibration import SurvivalLPBCalibration, SurvivalUPBCalibration
from src.train_model.models.utils import ModelPrediction, SurvivalModelPrediction


class UncalibratedLPBSurvivalCalibration(SurvivalLPBCalibration):
    def __init__(self, taus_range: torch.Tensor):
        super().__init__()
        self.taus_range = taus_range
        self.miscoverage = taus_range

    def calibrate(self, x_cal: torch.Tensor, t_tilde_cal: torch.Tensor, model_prediction_cal: ModelPrediction):
        pass

    def get_calibrated_lpb(self, target_taus: torch.Tensor, x: torch.Tensor, model_prediction: ModelPrediction):
        if not isinstance(model_prediction, SurvivalModelPrediction):
            raise Exception("error")
        miscoverage = self.miscoverage
        quantile_est = model_prediction.quantile_est
        target_taus = target_taus.to(quantile_est.device)
        tau_diff = target_taus - miscoverage[:, np.newaxis]
        smallest_pos = torch.where(tau_diff > 0, 1, -1.0 * np.inf).cumsum(dim=0).argmax(dim=0)
        calibrated_test_quantile_est = quantile_est[:, smallest_pos].squeeze()
        return calibrated_test_quantile_est

    def compute_metrics(self, model_prediction, target_taus) -> Dict[str, float]:
        return {}

    @property
    def name(self) -> str:
        return "uncalibrated"


class UncalibratedUPBSurvivalCalibration(SurvivalUPBCalibration):
    def __init__(self, taus_range: torch.Tensor):
        super().__init__()
        self.taus_range = taus_range
        self.miscoverage = taus_range

    def calibrate(self, x_cal: torch.Tensor, t_tilde_cal: torch.Tensor, model_prediction_cal: ModelPrediction):
        pass

    def get_calibrated_upb(self, target_taus: torch.Tensor, x: torch.Tensor, model_prediction: ModelPrediction):
        if not isinstance(model_prediction, SurvivalModelPrediction):
            raise Exception("error")
        miscoverage = self.miscoverage
        quantile_est = model_prediction.quantile_est
        target_taus = target_taus.to(quantile_est.device)
        tau_diff = target_taus - miscoverage[:, np.newaxis]
        smallest_pos = torch.where(tau_diff > 0, 1, -1.0 * np.inf).cumsum(dim=0).argmax(dim=0)
        calibrated_test_quantile_est = quantile_est[:, smallest_pos].squeeze()
        return calibrated_test_quantile_est

    def compute_metrics(self, model_prediction, target_taus) -> Dict[str, float]:
        return {}

    @property
    def name(self) -> str:
        return "uncalibrated"
