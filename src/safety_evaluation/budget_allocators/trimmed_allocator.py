import torch

from src.safety_evaluation.budget_allocators.budget_allocator import BudgetAllocator, BudgetAllocationResult
from src.safety_evaluation.budget_allocators.utils import sample_c
from src.safety_evaluation.calibration.calibration_utils import get_prior

class TrimmedBudgetAllocator(BudgetAllocator):

    def __init__(self, budget_per_sample: float, taus_range: torch.Tensor, tau_prior: float, max_estimator: int):
        super().__init__(budget_per_sample, taus_range, tau_prior)
        self.max_estimator = max_estimator

    def allocate_budget(self, probability_est: torch.Tensor, x: torch.Tensor, t: torch.Tensor,
                        quantile_est: torch.Tensor) -> BudgetAllocationResult:
        self.cal_size = len(probability_est)

        prior_quantile_est = get_prior(quantile_est, self.taus_range, self.tau_prior)

        C_probs = self.budget_per_sample / prior_quantile_est
        C_probs = torch.minimum(C_probs, torch.ones_like(C_probs))
        if self.max_estimator is not None:
            min_sample_size =  self.budget_per_sample / self.max_estimator
            C_probs = torch.maximum(C_probs, min_sample_size * torch.ones_like(C_probs))
        trimmed_prior_quantile_est = torch.minimum(prior_quantile_est,
                                                   self.max_estimator * torch.ones_like(prior_quantile_est))
        trimmed_quantile_est = torch.minimum(quantile_est, self.max_estimator * torch.ones_like(quantile_est))
        C = sample_c(C_probs, trimmed_prior_quantile_est)
        return BudgetAllocationResult(trimmed_quantile_est, C, C_probs)

    @property
    def name(self) -> str:
        return "trimmed"