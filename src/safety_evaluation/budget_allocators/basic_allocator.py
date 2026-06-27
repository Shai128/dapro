import torch

from src.safety_evaluation.budget_allocators.budget_allocator import BudgetAllocator, BudgetAllocationResult
from src.safety_evaluation.budget_allocators.utils import sample_c
from src.safety_evaluation.calibration.calibration_utils import get_prior


class BasicBudgetAllocator(BudgetAllocator):

    def __init__(self, budget_per_sample: float, taus_range: torch.Tensor, tau_prior: float):
        super().__init__(budget_per_sample, taus_range, tau_prior)

    def allocate_budget(self, probability_est: torch.Tensor, x: torch.Tensor, t: torch.Tensor,
                        quantile_est: torch.Tensor) -> BudgetAllocationResult:
        self.cal_size = len(probability_est)
        prior_quantile_est = get_prior(quantile_est, self.taus_range, self.tau_prior)
        C_probs = self.budget_per_sample / prior_quantile_est
        C_probs = torch.minimum(C_probs, torch.ones_like(C_probs))
        C = sample_c(C_probs, prior_quantile_est)
        total_budget_used = C.squeeze().sum().item()

        return BudgetAllocationResult(quantile_est, C, C_probs, total_budget_used=total_budget_used)

    @property
    def name(self) -> str:
        return "basic"

