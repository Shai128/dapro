import numpy as np

from src.safety_evaluation.budget_allocators.budget_allocator import BudgetAllocator, BudgetAllocationResult
from src.safety_evaluation.budget_allocators.optimization_solver_utils import solve_exact_fast
from src.safety_evaluation.budget_allocators.projected_optimization_utils import adaptive_budget_allocation, \
    construct_final_result, split_to_two_sets, project_to_test_platt, project_to_test_ir, project_to_test_beta
from src.safety_evaluation.calibration.calibration_utils import get_prior

import torch

from src.safety_evaluation.survival_utils.compute_mean_time_given_pmf import compute_quantile_survival_time


class ProjectedOptimizationBudgetAllocatorScoreError(BudgetAllocator):

    def __init__(self, conditional_grid, budget_per_sample, taus_range, tau_prior, m_upper_bound, projection: str, score: str,
                 score_error_lambda:float):
        super().__init__(budget_per_sample, taus_range, tau_prior)
        self.conditional_grid = conditional_grid
        self.min_pi = 0.005
        self.projection = projection
        self.score = score
        self.score_error_lambda = score_error_lambda
        if projection not in ['ir', 'platt', 'beta']:
            raise Exception(f"unknown projection {projection}, must be ir or platt")
        if score not in ['prob', 'quantile']:
            raise Exception(f"unknown projection {score}, must be 'prob', 'quantile'")
        # self.budget_per_sample *= 500

    @property
    def name(self) -> str:
        return f"projected_optimization_{self.projection}_{self.score}_lambda_{np.round(self.score_error_lambda,3)}"

    def allocate_budget(self, probability_est: torch.Tensor, x: torch.Tensor, t: torch.Tensor,
                        quantile_est: torch.Tensor) -> BudgetAllocationResult:
        device = self.conditional_grid.device
        N, T_max_curr, T_max_future = self.conditional_grid.shape
        prior_q = get_prior(quantile_est, self.taus_range, self.tau_prior)
        if self.score == 'prob':
            scores = self.conditional_grid[:, torch.arange(0, T_max_curr), torch.arange(0, T_max_curr)]
        elif self.score == 'quantile':
            scores = 1 / compute_quantile_survival_time(self.conditional_grid, quantile=0.9,
                                                        tail_distribution='geometric').squeeze()
        else:
            assert False
        rand_score = torch.rand_like(scores)
        scores = self.score_error_lambda * rand_score + (1-self.score_error_lambda) * scores
        val_idxs, test_idxs, val_grid, val_prior_q, t_val, val_scores, test_grid, test_prior_q, t_test, test_scores, val_max_steps, \
            target_budget_avg, val_budget_used = split_to_two_sets(self.conditional_grid, prior_q, t, scores,
                                                                   self.budget_per_sample, val_size=100)

        optimal_P = solve_exact_fast(val_scores, val_max_steps, target_budget_avg, verbose=False)
        optimal_P[optimal_P == 0] = 1
        if self.projection == 'ir':
            p_test = project_to_test_ir(optimal_P, val_scores, test_scores, val_prior_q, t_val, T_max_curr, device)
        elif self.projection == 'platt':
            p_test = project_to_test_platt(optimal_P, val_scores, test_scores, val_prior_q, t_val, T_max_curr, device)
        elif self.projection == 'beta':
            p_test = project_to_test_beta(optimal_P, val_scores, test_scores, val_prior_q, t_val, T_max_curr, device)
        else:
            assert False
        test_C, test_total_used, test_C_probs = adaptive_budget_allocation(p_test, test_prior_q, t_test, T_max_curr,
                                                                           device)
        total_budget_used = test_total_used + val_budget_used

        final_C, final_C_probs = construct_final_result(N, val_idxs, val_prior_q, test_idxs, test_prior_q, test_C,
                                                        test_C_probs, device)
        # p_val = torch.Tensor(optimal_P)
        # store(p_val, p_test, test_idxs, val_idxs, t, prior_q, final_C)
        # test_weights = (1 / test_C_probs).mean().item()
        # all_weights = (1 / final_C_probs).mean().item()
        # print(
        #     f"projected optimized new test weights {test_weights} | overall weights {all_weights} | total_budget_used {total_budget_used} "
        #     f"| total_budget {total_budget} | # observed: {(final_C > t).float().sum().item()}"
        #     f"| achieved prior: {(final_C.squeeze() >= prior_q).float().sum().item()}")
        t_range = np.arange(optimal_P.shape[1])

        mask = (t_range[None, :] < val_max_steps.cpu().detach().numpy()[:, None]).astype(np.float64)
        mask2 = (t_range[None, :] <= val_max_steps.cpu().detach().numpy()[:, None]).astype(np.float64)
        Y_best = np.log(optimal_P)
        val_obj = float(np.mean(np.exp(-np.sum(np.where(mask > 0, Y_best, 0.0), axis=1))))
        val_obj2 = float(np.mean(np.exp(-np.sum(np.where(mask2 > 0, Y_best, 0.0), axis=1))))
        val_budget = float(np.mean(np.sum(np.exp(np.cumsum(Y_best, axis=1)) * mask, axis=1)))
        valid_budget = int(val_budget < target_budget_avg)

        range_t = torch.arange(T_max_curr, device=device).unsqueeze(0)
        max_test = torch.minimum(test_prior_q, t_test)
        mask = range_t < max_test.unsqueeze(1)
        masked_p_test = p_test.clone()
        masked_p_test[~mask] = 1
        C_probs = masked_p_test.prod(dim=-1)  # should be equal to sim_cum_prob for samples with C > min(q,t)

        additional_metrics = {
            'val_obj': val_obj,
            'val_obj2': val_obj2,
            'val_icw': (1/optimal_P).mean().item(),
            'test_icw': (1/test_C_probs).mean().item(),
            'test_icw2': (1/C_probs).mean().item(),
            'all_icw': (1/final_C_probs).mean().item(),
            'valid_budget': valid_budget,
            'val_budget': val_budget,
        }
        return BudgetAllocationResult(quantile_est, final_C, final_C_probs, total_budget_used, additional_metrics=additional_metrics)


def store(p_val, p_test, test_idxs, val_idxs, t, prior_q, final_C):

    # Time arrangement is usually the same for all, so we only need it once
    time_arrange = torch.arange(0, p_test.shape[-1]).cpu()

    # curr_probabilities = p_test.cpu()  # Move to CPU to save GPU memory
    all_probabilities = torch.ones(prior_q.shape[0], p_test.shape[1]).cpu()
    all_probabilities[test_idxs] = p_test.cpu()
    all_probabilities[val_idxs] = p_val.cpu()
    expected_c1 = all_probabilities * (prior_q.cpu().unsqueeze(1) - time_arrange.unsqueeze(0)) + time_arrange.unsqueeze(0)
    # expected_c2 = ((time_arrange * all_probabilities.cumprod(dim=-1)[:, :-1]) * (1 - all_probabilities[:, -1])).sum(dim=-1) + \
    #               all_probabilities.prod(dim=-1) * prior_q

    N, T = all_probabilities.shape
    device = all_probabilities.device
    S = all_probabilities.cumprod(dim=-1)
    S_padded = torch.cat([torch.ones(N, 1, device=device), S], dim=-1)

    S_j = S.unsqueeze(1).expand(N, T, T)
    S_i_minus_1 = S_padded[:, :-1].unsqueeze(2).expand(N, T, T)

    S_rel = torch.triu(S_j / S_i_minus_1)

    ones_matrix = torch.ones(N, T, 1, device=device)
    S_rel_prev = torch.cat([ones_matrix, S_rel[:, :, :-1]], dim=-1)
    S_rel_prev = torch.triu(S_rel_prev)

    p_fail = (1 - all_probabilities).unsqueeze(1).expand(N, T, T)
    prob_die_at_k = S_rel_prev * p_fail

    time_matrix = time_arrange.view(1, 1, T).expand(N, T, T)
    sum_term = (time_matrix * prob_die_at_k).sum(dim=2)

    prob_survive_all = S_rel[:, :, -1]
    expected_c2 = sum_term + (prob_survive_all * prior_q.unsqueeze(1).cpu())


    data_to_save = {
        'time_arrange': time_arrange,
        'expected_c': expected_c1,
        'expected_c2': expected_c2,
        'event_times': t.cpu(),
        'prior_q': prior_q.cpu(),
        'final_Cs': final_C.cpu(),
        'test_idxs': torch.Tensor(test_idxs),
        'val_idxs': torch.Tensor(val_idxs),
    }

    # 4. Save to a file
    torch.save(data_to_save, 'projected_optimized_evaluation_plot_data.pt')

