import torch

from src.safety_evaluation.budget_allocators.budget_allocator import BudgetAllocator, BudgetAllocationResult
from src.safety_evaluation.budget_allocators.utils import sample_c
from src.safety_evaluation.calibration.calibration_utils import get_prior
from src.safety_evaluation.utils.utils import geom_cdf_stable


class NaiveBudgetAllocator(BudgetAllocator):

    def __init__(self, budget_per_sample: float, taus_range: torch.Tensor, tau_prior: float):
        super().__init__(budget_per_sample, taus_range, tau_prior)

    def allocate_budget(self, probability_est: torch.Tensor, x: torch.Tensor, t: torch.Tensor,
                        quantile_est: torch.Tensor) -> BudgetAllocationResult:
        prior_quantile_est = get_prior(quantile_est, self.taus_range, self.tau_prior)

        p_C = 1 / self.budget_per_sample
        C_probs = 1 - geom_cdf_stable(p=p_C, k=prior_quantile_est.cpu().detach().numpy())
        C_probs = torch.Tensor(C_probs).to(prior_quantile_est.device)

        C = sample_c(C_probs, prior_quantile_est)
        total_budget_used = C.squeeze().sum().item()

        return BudgetAllocationResult(quantile_est, C, C_probs, total_budget_used=total_budget_used)

    @property
    def name(self) -> str:
        return "naive"