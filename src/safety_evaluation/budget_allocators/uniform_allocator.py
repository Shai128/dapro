import torch
from src.safety_evaluation.budget_allocators.budget_allocator import BudgetAllocator, BudgetAllocationResult

class UniformBudgetAllocator(BudgetAllocator):
    """Allocates exactly max_time budget with probability p, and 0 otherwise."""

    def __init__(self, budget_per_sample: float, taus_range: torch.Tensor, tau_prior: float, m_upper_bound: int):
        super().__init__(budget_per_sample, taus_range, tau_prior)
        self.m_upper_bound = m_upper_bound
        # p = E[C_i] / max_time. Clamped to 1.0 in case budget > max_time
        self.p = min(1.0, float(self.budget_per_sample) / float(self.m_upper_bound))

    def allocate_budget(self, probability_est: torch.Tensor, x: torch.Tensor, t: torch.Tensor,
                        quantile_est: torch.Tensor) -> BudgetAllocationResult:
        N = quantile_est.shape[0]

        # Sample Bernoulli(p): 1 if sample gets max_time, 0 if it gets 0 budget
        rand_vals = torch.rand(N, device=quantile_est.device)
        allocate_mask = (rand_vals < self.p).float()

        C = allocate_mask * self.m_upper_bound

        # P(C >= t) for any t > 0 is exactly p.
        # This acts as our C_probs for IPCW re-weighting.
        C_probs = torch.full((N,), float(self.p), device=quantile_est.device)

        # Safety clamp to prevent Division by Zero in metrics if budget is 0
        C_probs = torch.clamp(C_probs, min=1e-5)
        total_budget_used = C.sum().item()
        return BudgetAllocationResult(quantile_est, C, C_probs, total_budget_used=total_budget_used)

    @property
    def name(self) -> str:
        return "UniformBudgetAllocator"


class UnweightedUniformBudgetAllocator(UniformBudgetAllocator):
    """Uniform allocator across samples using Geometric dist, WITHOUT metric re-weighting."""

    def allocate_budget(self, probability_est: torch.Tensor, x: torch.Tensor, t: torch.Tensor,
                        quantile_est: torch.Tensor) -> BudgetAllocationResult:
        res = super().allocate_budget(probability_est, x, t, quantile_est)
        # Override weights so no IPCW re-weighting occurs
        res.C_probs = torch.ones_like(res.C_probs)
        return res

    @property
    def name(self) -> str:
        return "UnweightedUniformBudgetAllocator"
