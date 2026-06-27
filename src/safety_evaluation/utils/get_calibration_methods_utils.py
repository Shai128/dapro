import numpy as np

from src.safety_evaluation.bandit_algorithms.discounted_ucb import DiscountedUCB
from src.safety_evaluation.bandit_algorithms.new_bandits_algorithm import NewBanditsAlgorithm
from src.safety_evaluation.bandit_algorithms.rexp3 import Rexp3
from src.safety_evaluation.budget_allocators.adaptive_optimized_allocator import AdaptiveOptimizedBudgetAllocator
from src.safety_evaluation.budget_allocators.bandits_allocator import BanditsBudgetAllocator
from src.safety_evaluation.budget_allocators.basic_allocator import BasicBudgetAllocator
from src.safety_evaluation.budget_allocators.budget_allocator import BudgetAllocator
from src.safety_evaluation.budget_allocators.naive_allocator import NaiveBudgetAllocator
from src.safety_evaluation.budget_allocators.optimized_allocators import OptimizedBudgetAllocator
from src.safety_evaluation.budget_allocators.DAPRO import DAPRO
from src.safety_evaluation.budget_allocators.projected_optimization_allocator_score_error import \
    ProjectedOptimizationBudgetAllocatorScoreError
from src.safety_evaluation.budget_allocators.trimmed_allocator import TrimmedBudgetAllocator
from src.safety_evaluation.calibration.abstract_calibration import SurvivalLPBCalibration
from src.safety_evaluation.calibration.dummy_calibration import UncalibratedLPBSurvivalCalibration
from src.safety_evaluation.calibration.survival_calibration_with_known_weights import SurvivalCalibrationWithKnownWeights
from src.safety_evaluation.reward_functions.reward_by_probability import RewardByProbability
from src.safety_evaluation.reward_functions.reward_by_probability_diff import RewardByProbabilityDiff
from src.safety_evaluation.utils.get_best_params_utils import get_best_rexp3_params, get_best_discounted_ucb_params, \
    new_alg_best_params
from src.safety_evaluation.construct_calibrated_bound import is_budget_sufficient_for_split
from src.safety_evaluation.budget_allocators.uniform_allocator import UniformBudgetAllocator, UnweightedUniformBudgetAllocator
from typing import List


def get_baseline_calibrations(conditional_grid, budget_per_sample, taus_range, tau_prior, m_upper_bound):
    naive_allocation = NaiveBudgetAllocator(budget_per_sample, taus_range, tau_prior)
    basic_allocation = BasicBudgetAllocator(budget_per_sample, taus_range, tau_prior)
    trimmed_allocation = TrimmedBudgetAllocator(budget_per_sample, taus_range, tau_prior, m_upper_bound)
    optimized_allocation = OptimizedBudgetAllocator(budget_per_sample, taus_range, tau_prior, m_upper_bound)
    adaptive_allocation = AdaptiveOptimizedBudgetAllocator(conditional_grid, budget_per_sample, taus_range, tau_prior, m_upper_bound)

    all_allocations: List[BudgetAllocator] = [basic_allocation, trimmed_allocation, optimized_allocation, adaptive_allocation]

    for projection in ['ir', 'platt', 'beta']:
        for score in ['prob', 'quantile']:
            all_allocations.append(DAPRO(conditional_grid,
                                         budget_per_sample, taus_range,
                                         tau_prior, m_upper_bound,
                                         projection=projection, score=score))
            for n1 in [25, 50, 100, 150, 200, 250, 300, 400, 500, 750, 1000, 1250, 1500]:
                all_allocations.append(DAPRO(conditional_grid,
                                                       budget_per_sample, taus_range,
                                                       tau_prior, m_upper_bound,
                                                       projection=projection, score=score,
                                                       n1=n1))
            for score_error_lambda in list(np.arange(0, 1, 0.1)) + [0.95, 0.99]:
                score_error_lambda = np.round(score_error_lambda, 2)
                all_allocations.append(ProjectedOptimizationBudgetAllocatorScoreError(conditional_grid,
                                                       budget_per_sample, taus_range,
                                                       tau_prior, m_upper_bound,
                                                       projection=projection, score=score,
                                                       score_error_lambda=score_error_lambda))

    dummy_calibration = UncalibratedLPBSurvivalCalibration(taus_range)
    all_calibrations: List[SurvivalLPBCalibration] = [dummy_calibration]
    all_calibrations.extend([SurvivalCalibrationWithKnownWeights(allocation, taus_range, tau_prior) for
                             allocation in all_allocations])

    return all_calibrations


def get_new_allocation_algorithms(conditional_grid, budget_per_sample, taus_range, tau_prior, m_upper_bound, allocations, device):
    all_allocations = []
    do_diffs = [False, True]
    if allocations == 'none':
        return []
    elif allocations == 'all':
        k_values = [1,2, 3,4, 5, 10, 50, 5000]
        first_step_budget_values = [0.01, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.75, 0.95]
    elif allocations == 'one':
        k,b = new_alg_best_params()
        k_values = [k]
        first_step_budget_values = [b, 0.95]
        # first_step_budget_values = [b]
    else:
        raise Exception("not valid allocations")
    for do_diff in do_diffs:
        for k in k_values:
            for b in first_step_budget_values:
                reward_function = RewardByProbability(conditional_grid)
                bandits_algorithm = NewBanditsAlgorithm(reward_function, device=device, k=k, n_monte_carlo=1)
                new_allocation = BanditsBudgetAllocator(budget_per_sample, taus_range, tau_prior, m_upper_bound,
                 first_step_budget_ratio=b, bandits_algorithm=bandits_algorithm, do_diff=do_diff)
                # new_allocation = NewBudgetAllocator(budget_per_sample, taus_range, tau_prior, m_upper_bound,
                #                                     k=k, first_step_budget_ratio=b, do_diff=do_diff)
                all_allocations.append(new_allocation)
    return all_allocations


def get_discounted_ucb_allocation_algorithms(budget_per_sample, taus_range, tau_prior, m_upper_bound, allocations, device):
    all_allocations = []
    do_diffs = [False, True]
    if allocations == 'none':
        return []
    elif allocations == 'all':
        xi_values = [0.01, 0.05, 0.1, 0.2, 0.5, 0.75, 0.9, 1.]
        gamma_values = [0.01, 0.05, 0.1, 0.2, 0.5, 0.75, 0.9, 1.]
        first_step_budget_values = [ 0.1, 0.2, ]
    elif allocations == 'one':
        xi, gamma, b = get_best_discounted_ucb_params()
        xi_values = [xi]
        gamma_values = [gamma]
        first_step_budget_values = [b]
    else:
        raise Exception("not valid allocations")
    for do_diff in do_diffs:
        for gamma in gamma_values:
            for xi in xi_values:
                for b in first_step_budget_values:
                    reward_function = RewardByProbabilityDiff(None)
                    bandits_algorithm = DiscountedUCB(reward_function, gamma=gamma, xi=xi, device=device)
                    new_allocation = BanditsBudgetAllocator(budget_per_sample, taus_range, tau_prior, m_upper_bound,
                                                            first_step_budget_ratio=b,
                                                            bandits_algorithm=bandits_algorithm, do_diff=do_diff)
                    all_allocations.append(new_allocation)
    return all_allocations


def get_bandits_allocation_algorithms(budget_per_sample, taus_range, tau_prior, m_upper_bound, allocations, device):
    all_allocations = []
    do_diffs = [False, True]
    # return []
    if allocations == 'none':
        return []
    elif allocations == 'all':
        first_step_budget_values = [0.1, 0.2]
        gamma_values = [0., 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.75, 0.8, 0.95, 1.]
        delta_values = [0., 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    elif allocations == 'one':
        g,d, rb = get_best_rexp3_params()
        first_step_budget_values = [rb]
        gamma_values = [g]
        delta_values = [d]
    else:
        raise Exception("not valid allocations")
    for do_diff in do_diffs:
        for gamma in gamma_values:
            for delta in delta_values:
                for b in first_step_budget_values:
                    reward_function = RewardByProbabilityDiff(None)
                    bandits_algorithm = Rexp3(reward_function, gamma=gamma, delta_t=delta, device=device)
                    new_allocation = BanditsBudgetAllocator(budget_per_sample, taus_range, tau_prior, m_upper_bound,
                                                            first_step_budget_ratio=b,
                                                            bandits_algorithm=bandits_algorithm, do_diff=do_diff)
                    all_allocations.append(new_allocation)

    return all_allocations


import torch

def get_metric_allocators(conditional_grid, budget_per_sample, m_upper_bound, taus_range, tau_prior, device, t_tilde_cal=None, cal_model_prediction=None) -> List[BudgetAllocator]:
    """Factory method for allocators used in metric estimation."""
    allocators = []

    allocators.append(NaiveBudgetAllocator(budget_per_sample, taus_range, tau_prior))
    allocators.append(OptimizedBudgetAllocator(budget_per_sample, taus_range, tau_prior, m_upper_bound))
    allocators.append(
        AdaptiveOptimizedBudgetAllocator(conditional_grid, budget_per_sample, taus_range, tau_prior, m_upper_bound))
        
    N = len(conditional_grid) if conditional_grid is not None else 0
    total_budget = budget_per_sample * N
    
    prior_q = None
    censored_event_time = None
    if t_tilde_cal is not None and cal_model_prediction is not None and conditional_grid is not None:
        censored_event_time = t_tilde_cal
        tau_idx = np.argmin(np.abs(taus_range.cpu().numpy() - tau_prior))
        prior_q = torch.ones_like(cal_model_prediction.quantile_est[:, tau_idx]) * conditional_grid.shape[1]

    def check_budget(n1):
        if prior_q is None or censored_event_time is None:
            return True
        return is_budget_sufficient_for_split(N, n1, total_budget, censored_event_time, prior_q)

    if check_budget(100):
        allocators.append(DAPRO(
            conditional_grid, budget_per_sample, taus_range, tau_prior, m_upper_bound,
            projection='platt', score='prob'
        ))

    allocators.append(UniformBudgetAllocator(budget_per_sample, taus_range, tau_prior, m_upper_bound))
    allocators.append(UnweightedUniformBudgetAllocator(budget_per_sample, taus_range, tau_prior, m_upper_bound))
    allocators.extend(get_new_allocation_algorithms(conditional_grid, budget_per_sample, taus_range, tau_prior, m_upper_bound, allocations='one', device=device))
    
    for n1 in [25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 750, 1000]:
        if check_budget(n1):
            for projection in ['platt', 'beta']:
                for score in ['prob', 'quantile']:
                    allocators.append(DAPRO(conditional_grid,
                                            budget_per_sample, taus_range,
                                            tau_prior, m_upper_bound,
                                            projection=projection, score=score,
                                            n1=n1))
    return allocators
