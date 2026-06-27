import torch

from src.safety_evaluation.budget_allocators.budget_allocator import BudgetAllocator, BudgetAllocationResult
from src.safety_evaluation.budget_allocators.utils import sample_c
from src.safety_evaluation.calibration.calibration_utils import get_prior, solve_optimization


class OptimizedBudgetAllocator(BudgetAllocator):

    def __init__(self, budget_per_sample: float, taus_range: torch.Tensor, tau_prior: float, max_estimator: int):
        super().__init__(budget_per_sample, taus_range, tau_prior)
        self.max_estimator = max_estimator

    def allocate_budget(self, probability_est: torch.Tensor, x: torch.Tensor, t: torch.Tensor,
                        quantile_est: torch.Tensor) -> BudgetAllocationResult:
        self.cal_size = len(probability_est)

        prior_quantile_est = get_prior(quantile_est, self.taus_range, self.tau_prior)

        device = probability_est.device
        trimmed_prior_quantile_est = torch.minimum(prior_quantile_est,
                                                   self.max_estimator * torch.ones_like(prior_quantile_est))
        trimmed_quantile_est = torch.minimum(quantile_est, self.max_estimator * torch.ones_like(quantile_est))
        C_probs, _ = solve_optimization(trimmed_prior_quantile_est.cpu().detach().numpy(),
                                        self.budget_per_sample * len(prior_quantile_est), tol=1e-8)
        C_probs = torch.Tensor(C_probs).to(device)

        C = sample_c(C_probs, prior_quantile_est)
        print()
        print(f"optimized mean weights: {(1/(C_probs.squeeze())).mean()} total_budget_used: { C.sum().item()} total_budget: {self.cal_size * self.budget_per_sample} # observed: {(C.squeeze() > t).float().sum().item()}"
              f" achieved prior: {(C.squeeze() >= prior_quantile_est).float().sum().item()}")
        print(C_probs)
        total_budget_used = C.squeeze().sum().item()

        # store(t, C_probs, prior_quantile_est, C)
        return BudgetAllocationResult(trimmed_quantile_est, C, C_probs, total_budget_used=total_budget_used)

    @property
    def name(self) -> str:
        return "optimized"

def store(t, C_probs, prior_quantile_est, C):

    curr_event_time = t.cpu()
    curr_prior_q = prior_quantile_est.cpu()
    curr_final_C = C.cpu()
    expected_c = (C_probs * prior_quantile_est).cpu()

    data_to_save = {
        'expected_c': expected_c,
        'event_times': curr_event_time,
        'prior_qs':curr_prior_q,
        'final_Cs': curr_final_C,
    }

    # 4. Save to a file
    torch.save(data_to_save, 'static_optimized_evaluation_plot_data.pt')
    print("stored static optimized")
