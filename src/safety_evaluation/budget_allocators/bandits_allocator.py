import numpy as np
import torch

from src.safety_evaluation.bandit_algorithms.abstract_bandit_algorithm import BanditAlgorithm
from src.safety_evaluation.budget_allocators.budget_allocator import BudgetAllocator, BudgetAllocationResult
from src.safety_evaluation.calibration.calibration_utils import get_prior, solve_optimization, sample_calibration_set


class BanditsBudgetAllocator(BudgetAllocator):

    def __init__(self, budget_per_sample: float, taus_range: torch.Tensor, tau_prior: float, max_estimator: int,
                 first_step_budget_ratio: float, bandits_algorithm: BanditAlgorithm, do_diff: bool):
        super().__init__(budget_per_sample, taus_range, tau_prior)
        self.max_estimator = max_estimator
        self.bandits_algorithm = bandits_algorithm
        self.first_step_budget_ratio = first_step_budget_ratio
        self.do_diff = do_diff


    def allocate_budget(self, probability_est: torch.Tensor, x: torch.Tensor, t: torch.Tensor,
                        quantile_est: torch.Tensor) -> BudgetAllocationResult:

        self.cal_size = len(probability_est)

        budget_per_sample = self.budget_per_sample
        prior_quantile_est = get_prior(quantile_est, self.taus_range, self.tau_prior)
        total_budget = budget_per_sample * quantile_est.shape[0]
        first_step_budget = int(total_budget * self.first_step_budget_ratio)
        second_step_budget = total_budget - first_step_budget
        n_arms = len(probability_est)
        max_c = torch.minimum(t, torch.minimum(prior_quantile_est + 1, torch.ones_like(prior_quantile_est) *
                                               probability_est.shape[1]))
        max_time = probability_est.shape[1]
        tmp_c = self.bandits_algorithm.run(first_step_budget, n_arms, max_c, max_time, probability_est=probability_est)
        temp_c_improved = tmp_c.clone()
        temp_c_improved[temp_c_improved == t] = prior_quantile_est[temp_c_improved == t].to(temp_c_improved.dtype)
        C = temp_c_improved
        C_probs = torch.zeros_like(C).to(prior_quantile_est.dtype)
        # observed_events = (prior_quantile_est.unsqueeze(0).to(temp_c_improved.dtype) <= temp_c_improved).float().mean(
        #     dim=0).bool()
        # remaining_budget_per_sample = second_step_budget / (~observed_events).sum().item()

        trimmed_prior_quantile_est = torch.minimum(prior_quantile_est,
                                                   self.max_estimator * torch.ones_like(prior_quantile_est))
        observed_events = (
                trimmed_prior_quantile_est.unsqueeze(0).to(temp_c_improved.dtype) <= temp_c_improved).float().mean(
            dim=0).bool()
        C_probs[observed_events] = 1
        if (~observed_events).sum().item() == 0:
            trimmed_quantile_est = torch.minimum(quantile_est, self.max_estimator * torch.ones_like(quantile_est))
            return BudgetAllocationResult(trimmed_quantile_est, C, C_probs, total_budget_used=total_budget)
        else:
            remaining_budget_per_sample = second_step_budget / (~observed_events).sum().item()
        first_stage_c = C.clone()
        remaining_generations = trimmed_prior_quantile_est - first_stage_c
        do_diff = self.do_diff
        if do_diff:
            opt_c_prob, _ = solve_optimization(remaining_generations[~observed_events].cpu().detach().numpy(),
                                               remaining_budget_per_sample * (~observed_events).sum().item(), tol=1e-8)
            C_probs[~observed_events] = torch.Tensor(opt_c_prob).to(C_probs.device).to(C_probs.dtype)

            not_obsereved_T_tilde, not_obsereved_C = sample_calibration_set(
                remaining_generations[~observed_events],
                C_probs[~observed_events],
                t[~observed_events]
            )
            C[observed_events] = first_stage_c[observed_events]
            C[~observed_events] = not_obsereved_C.squeeze() + first_stage_c[~observed_events].squeeze()
            total_budget_used = first_step_budget + not_obsereved_C.sum().item()
        else:
            opt_c_prob, _ = solve_optimization(trimmed_prior_quantile_est[~observed_events].cpu().detach().numpy(),
                                               remaining_budget_per_sample * (~observed_events).sum().item(), tol=1e-8)
            C_probs[~observed_events] = torch.Tensor(opt_c_prob).to(C_probs.device).to(C_probs.dtype)

            T_tilde, C = sample_calibration_set(
                trimmed_prior_quantile_est,
                C_probs,
                t
            )
            total_budget_used = first_step_budget + C[~observed_events].sum().item()

        trimmed_quantile_est = torch.minimum(quantile_est, self.max_estimator * torch.ones_like(quantile_est))

        return BudgetAllocationResult(trimmed_quantile_est, C, C_probs, total_budget_used)

    @property
    def name(self) -> str:
        return f"{self.bandits_algorithm.name}_diff={self.do_diff}_first_step_budget={np.round(self.first_step_budget_ratio, 3)}"