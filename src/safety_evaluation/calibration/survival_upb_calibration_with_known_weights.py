from typing import Dict, Union

import numpy as np
import torch

from src.safety_evaluation.budget_allocators.budget_allocator import BudgetAllocator, BudgetAllocationResult
from src.safety_evaluation.calibration.abstract_calibration import  SurvivalUPBCalibration
from src.safety_evaluation.calibration.calibration_utils import get_prior
from src.train_model.models.utils import ModelPrediction, SurvivalModelPrediction


def get_gamma(m_upper_bound: float, budget_per_sample: float):
    gamma = (m_upper_bound / budget_per_sample)
    return gamma


def get_coverage_rate_deviation(gamma: float, cal_size: int, delta: float):
    coverage_rate_bound = np.sqrt((2 * gamma ** 2 + 5) / cal_size * np.log(1 / delta))
    return coverage_rate_bound


class SurvivalUPBCalibrationWithKnownWeights(SurvivalUPBCalibration):

    def __init__(self, budget_allocator: BudgetAllocator, taus_range: torch.Tensor, tau_prior: float):
        super().__init__()
        self.budget_allocator = budget_allocator
        self.miscoverage = None
        self.taus_range = taus_range
        self.allocation_result: Union[BudgetAllocationResult, None] = None
        self.t_cal = None
        self.tau_prior = tau_prior

    def calibrate(self, x_cal: torch.Tensor, t_cal: torch.Tensor, model_prediction_cal: ModelPrediction):
        if not isinstance(model_prediction_cal, SurvivalModelPrediction):
            raise Exception("error")
        allocation_result = self.budget_allocator.allocate_budget(model_prediction_cal.probability_est, x_cal, t_cal,
                                                                  model_prediction_cal.quantile_est)
        self.allocation_result = allocation_result
        self.t_cal = t_cal
        f, C, C_probs = allocation_result.f, allocation_result.C, allocation_result.C_probs
        # tau_idx = 1200
        # gt_coverage_rate = ((t_cal <= f[:, tau_idx]) | (f[:, tau_idx] == 200)).float().mean()
        # est_coverage_rate = ((1.0 / C_probs)*( ((f[:, tau_idx] <= C)) & ((t_cal <= f[:, tau_idx]) | (f[:, tau_idx] == 200))).float()).mean()
        # print(f"tau: {self.taus_range[tau_idx]}, gt_coverage_rate: {gt_coverage_rate}, est_coverage_rate: {est_coverage_rate}")

        weights = (1.0 / C_probs.reshape(-1, 1)).repeat(1, f.shape[1])  # [N, n_taus]
        weights[t_cal.reshape(-1, 1) <= f] = 0  # keeps T_i > f̂_τ only
        weights[f > C.reshape(-1, 1)] = 0
        weights[f == 200] = 0
        self.miscoverage = weights.mean(dim=0)  # shape: [n_taus], decreasing in τ


    def get_calibrated_upb(self, target_taus: torch.Tensor, x: torch.Tensor, model_prediction: ModelPrediction):
        if not isinstance(model_prediction, SurvivalModelPrediction):
            raise Exception("error")
        miscoverage = self.miscoverage
        quantile_est = model_prediction.quantile_est

        target_coverages = target_taus.to(quantile_est.device)
        alpha_levels = (1.0 - target_coverages).to(quantile_est.device)  # [n_targets]

        valid = (miscoverage[:, None] <= alpha_levels[None, :])  # decreasing → F..F,T..T

        hat_tau_idx = valid.float().argmax(dim=0)  # [n_targets]

        calibrated_test_quantile_est = quantile_est[:, hat_tau_idx].squeeze()

        # tau_diff = target_taus - miscoverage[:, np.newaxis]
        # smallest_pos = torch.where(tau_diff > 0, 1, -1.0 * np.inf).cumsum(dim=0).argmax(dim=0)
        # calibrated_test_quantile_est = quantile_est[:, smallest_pos].squeeze()
        if hasattr(self.budget_allocator, 'max_estimator'):
            max_estimator = self.budget_allocator.max_estimator
            calibrated_test_quantile_est = torch.min(calibrated_test_quantile_est,
                                                     max_estimator * torch.ones_like(calibrated_test_quantile_est))
        return calibrated_test_quantile_est

    @property
    def name(self) -> str:
        return f"calibration_{self.budget_allocator.name}_allocation"

    def compute_metrics(self, model_prediction, target_taus) -> Dict[str, float]:
        t = self.t_cal
        f = self.allocation_result.f
        C = self.allocation_result.C
        C_probs = self.allocation_result.C_probs
        mean_weight = self.allocation_result.mean_weight
        max_weight = self.allocation_result.max_weight
        if mean_weight is None:
            mean_weight = (1 / C_probs).mean().item()
        if max_weight is None:
            max_weight = (1 / C_probs).max().item()
        budget_used = self.allocation_result.total_budget_used
        if budget_used is None:
            budget_used = C.sum().item()
        f_prior = get_prior(f, self.taus_range, self.tau_prior)
        prior_observed_jailbreaks = (t.squeeze() <= f_prior.squeeze()).float().sum().item()
        prior_observed_f_lower_c = (f_prior.squeeze() <= C.squeeze()).float().sum().item()
        prior_observed_both = (
                (f_prior.squeeze() <= C.squeeze()) & (t.squeeze() <= f_prior.squeeze())).float().sum().item()
        n_observed_events = (C.squeeze() > t).float().sum().item()
        n_achieved_q_prior1 = (C.squeeze() >= f_prior).float().sum().item()
        n_achieved_q_prior2 = (C.squeeze() > f_prior).float().sum().item()

        miscoverage = self.miscoverage
        quantile_est = model_prediction.quantile_est
        target_taus = target_taus.to(quantile_est.device)
        tau_diff = target_taus - miscoverage[:, np.newaxis]
        smallest_pos = torch.where(tau_diff > 0, 1, -1.0 * np.inf).cumsum(dim=0).argmax(dim=0)
        calibrated_test_quantile_est = f[:, smallest_pos].squeeze()
        alpha_hat_per_tau = miscoverage[smallest_pos]

        all_observed_jailbreaks = (t.squeeze().unsqueeze(-1) <= calibrated_test_quantile_est.squeeze()).float().sum(
            dim=0)
        all_f_lower_c = (calibrated_test_quantile_est.squeeze() <= C.squeeze().unsqueeze(-1)).float().sum(dim=0)
        all_observed_both = ((calibrated_test_quantile_est.squeeze() <= C.squeeze().unsqueeze(-1)) & (
                t.squeeze().unsqueeze(-1) <= calibrated_test_quantile_est.squeeze())).float().sum(dim=0)
        if hasattr(self.budget_allocator, "cal_size") and hasattr(self.budget_allocator, "max_estimator"):
            cal_size = self.budget_allocator.cal_size
            max_estimator = self.budget_allocator.max_estimator
            budget_per_sample = self.budget_allocator.budget_per_sample
            gamma = get_gamma(max_estimator, budget_per_sample)
            coverage_deviation = get_coverage_rate_deviation(gamma, cal_size, delta=0.1)
        else:
            coverage_deviation = None
        additional_metrics = self.allocation_result.additional_metrics # if self.allocation_result.additional_metrics else {}
        if additional_metrics is None:
            additional_metrics = {}
        metrics = {
            'coverage_deviation': coverage_deviation,
            'prior_observed_jailbreaks': prior_observed_jailbreaks,
            'prior_observed_f_lower_c': prior_observed_f_lower_c,
            'prior_observed_both': prior_observed_both,
            'n_observed_events': n_observed_events,
            'n_achieved_q_prior1': n_achieved_q_prior1,
            'n_achieved_q_prior2': n_achieved_q_prior2,
            'budget_used': budget_used,
            'mean_weight': mean_weight,
            'max_weight': max_weight,
            **{f'all_observed_jailbreaks_{i}': all_observed_jailbreaks[i].item() for i in
               range(len(all_observed_jailbreaks))},
            **{f'all_f_lower_c_{i}': all_f_lower_c[i].item() for i in range(len(all_f_lower_c))},
            **{f'all_observed_both_{i}': all_observed_both[i].item() for i in range(len(all_observed_both))},
            **{f'alpha_hat_per_tau_{i}': alpha_hat_per_tau[i].item() for i in range(len(alpha_hat_per_tau))},
            **additional_metrics
        }
        return metrics
