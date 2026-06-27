import re
import os
import traceback
import pandas as pd
import torch
import tqdm
import numpy as np
from concurrent.futures import as_completed, ThreadPoolExecutor
from typing import List
import argparse

from src.safety_evaluation.budget_allocators.adaptive_optimized_allocator import AdaptiveOptimizedBudgetAllocator
from src.safety_evaluation.budget_allocators.basic_allocator import BasicBudgetAllocator
from src.safety_evaluation.budget_allocators.budget_allocator import BudgetAllocator
from src.safety_evaluation.budget_allocators.naive_allocator import NaiveBudgetAllocator
from src.safety_evaluation.budget_allocators.optimized_allocators import OptimizedBudgetAllocator
from src.safety_evaluation.budget_allocators.DAPRO import DAPRO
from src.safety_evaluation.budget_allocators.trimmed_allocator import TrimmedBudgetAllocator

# LPB Calibrations
from src.safety_evaluation.calibration.abstract_calibration import SurvivalLPBCalibration
from src.safety_evaluation.calibration.dummy_calibration import UncalibratedLPBSurvivalCalibration
from src.safety_evaluation.calibration.survival_calibration_with_known_weights import get_gamma, SurvivalCalibrationWithKnownWeights

# UPB Calibrations
from src.safety_evaluation.calibration.abstract_calibration import SurvivalUPBCalibration
from src.safety_evaluation.calibration.dummy_calibration import UncalibratedUPBSurvivalCalibration
from src.safety_evaluation.calibration.survival_upb_calibration_with_known_weights import SurvivalUPBCalibrationWithKnownWeights

from src.dataset_utils.data_utils import get_data
from src.train_model.models.utils import SurvivalModelPrediction

from src.safety_evaluation.utils.get_calibration_methods_utils import get_new_allocation_algorithms
from src.safety_evaluation.utils.utils import (
    compute_probabilities_and_quantiles,
    split_data,
    get_tmp_calibration_result_path,
    get_tmp_upb_calibration_result_path,
    setup_experiment_data
)
from src.utils.utils import set_seeds


def compute_metrics_bound(bound: torch.Tensor, t_tilde: torch.Tensor, bound_type: str):
    if bound_type == 'lpb':
        coverage_rate = (t_tilde.unsqueeze(1) >= bound).float().mean(dim=0)
    elif bound_type == 'upb':
        coverage_rate = ((t_tilde.unsqueeze(1) <= bound) | (bound == 200)).float().mean(dim=0)
    else:
        raise ValueError("Invalid bound_type. Must be 'lpb' or 'upb'")

    length = bound.mean(dim=0)
    return coverage_rate, length


def preprocess_df(df, target_taus_list, defaults=None):
    if defaults is None:
        defaults = {
            'budget_used': 0, 'mean_weight': 0, 'n_observed_events': 0,
            'n_achieved_q_prior1': 0, 'n_achieved_q_prior2': 0
        }

    row_series = df.rename(columns={"Unnamed: 0": "seed"}, errors='ignore').iloc[0]
    row_dict = row_series.to_dict()

    for key, default_val in defaults.items():
        if key not in row_dict:
            row_dict[key] = default_val

    static_data = {}
    dynamic_data = {}
    n_taus = len(target_taus_list)

    for key, val in row_dict.items():
        match = re.search(r'^(.*)_(\d+)$', str(key))
        if match:
            base_name, idx_str = match.groups()
            idx = int(idx_str)

            if 0 <= idx < n_taus:
                if base_name not in dynamic_data:
                    dynamic_data[base_name] = {}
                dynamic_data[base_name][idx] = val
        else:
            static_data[key] = val

    rows = []
    for i in range(n_taus):
        new_row = static_data.copy()
        for base_name, values in dynamic_data.items():
            if i in values:
                new_row[base_name] = values[i]
        rows.append(new_row)

    return pd.DataFrame(rows)


def run_one_experiment(experiments_name, seed, calibration, x_cal, t_tilde_cal, cal_model_prediction, x_test,
                       t_tilde_test, test_model_prediction, target_taus_list, bound_type, skip_existing=True):
    try:
        if bound_type == 'lpb':
            dir_path = get_tmp_calibration_result_path(experiments_name, calibration.name)
        else:
            dir_path = get_tmp_upb_calibration_result_path(experiments_name, calibration.name)

        save_path = os.path.join(f"{dir_path}", f"seed={seed}.csv")
        if os.path.exists(save_path) and skip_existing:
            return

        set_seeds(seed)
        calibration.calibrate(x_cal, t_tilde_cal, cal_model_prediction)
        target_taus = torch.Tensor(target_taus_list)

        with torch.no_grad():
            if bound_type == 'lpb':
                calibrated_test_bound = calibration.get_calibrated_lpb(target_taus, x_test, test_model_prediction)
            else:
                calibrated_test_bound = calibration.get_calibrated_upb(target_taus, x_test, test_model_prediction)

        coverage_rate, length = compute_metrics_bound(calibrated_test_bound, t_tilde_test, bound_type)
        calibration_metrics = calibration.compute_metrics(cal_model_prediction, target_taus)

        if bound_type == 'lpb':
            target_coverage = {f'target_coverage_{i}': 1 - target_taus_list[i] for i in range(len(target_taus_list))}
        else:
            target_coverage = {f'target_coverage_{i}': target_taus_list[i] for i in range(len(target_taus_list))}

        all_metrics = {
            'seed': seed,
            'calibration_name': calibration.name,
            **{f'coverage_{i}': coverage_rate[i].item() for i in range(len(coverage_rate))},
            **{f'size_{i}': length[i].item() for i in range(len(length))},
            **target_coverage,
            **calibration_metrics,
        }

        df = pd.DataFrame(all_metrics, index=[seed])
        processed_df = preprocess_df(df, target_taus_list)
        abs_path = os.path.abspath(save_path)

        if os.name == 'nt' and not abs_path.startswith('\\\\?\\'):
            abs_path = f"\\\\?\\{abs_path}"
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)

        processed_df.to_csv(abs_path)

    except Exception as e:
        raise Exception(traceback.print_exc())


def store(method_name, lengths):
    data_to_save = {
        'lengths': lengths.cpu(),
    }
    torch.save(data_to_save, f'{method_name}_lengths.pt')


def run_experiments(cal_size, is_real, device, dataset_name, data_setup, experiments_name, seeds,
                    taus_range, budget_per_sample, tau_prior, m_upper_bound, target_taus_list, skip_existing,
                    allocations, bound_type):
    max_time, t_tilde_cal_test, quantile_est_cal_test, probability_est, conditional_grid, test_size = setup_experiment_data(
        cal_size, is_real, device, dataset_name, data_setup, taus_range, m_upper_bound
    )
    taus_range = taus_range.detach()
    futures = []
    num_cpus = os.cpu_count()

    for seed in tqdm.tqdm(range(seeds[0], seeds[1]), desc="running calibration algorithms"):
        x_cal, x_test, t_tilde_cal, probability_est_cal, quantile_est_cal, t_tilde_test, quantile_est_test, \
            probability_est_test, cal_idx, test_idx = split_data(seed, cal_size, test_size, None, t_tilde_cal_test,
                                                                 probability_est, quantile_est_cal_test)

        curr_conditional_grid = conditional_grid[cal_idx]

        if bound_type == 'lpb':
            quantile_est_cal = quantile_est_cal.clip(max=max_time)

        cal_model_prediction = SurvivalModelPrediction(quantile_est_cal, probability_est_cal)
        test_model_prediction = SurvivalModelPrediction(quantile_est_test, probability_est_test)

        all_calibrations = get_calibration_methods(
            curr_conditional_grid, budget_per_sample, taus_range, tau_prior,
            m_upper_bound, allocations, cal_model_prediction, t_tilde_cal, device, bound_type
        )

        with ThreadPoolExecutor(max_workers=num_cpus) as executor:
            futures += [
                executor.submit(run_one_experiment, experiments_name, seed, calibration,
                                x_cal, t_tilde_cal, cal_model_prediction,
                                x_test, t_tilde_test, test_model_prediction,
                                target_taus_list, bound_type, skip_existing)
                for calibration in all_calibrations
            ]

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Calibration failed with error: {e}")


def set_m_upper_bound(gamma: float, budget_per_sample: float):
    m_upper_bound = gamma * budget_per_sample
    if m_upper_bound < budget_per_sample:
        print(f"warning, m_upper_bound = {m_upper_bound} which leads to gamma lower than 1")
    if abs(gamma - (m_upper_bound / budget_per_sample)) > 0.01:
        print(f"warning, gamma is: {gamma} but the bound leads to {(m_upper_bound / budget_per_sample)}")
    return m_upper_bound


def is_budget_sufficient_for_split(N, n1, total_budget, censored_event_time, prior_q):
    if n1 > N:
        return False
    perm = np.random.permutation(N)
    val_idxs = perm[:n1]

    t_val = censored_event_time[val_idxs]
    val_prior_q = prior_q[val_idxs]
    val_budget_used = torch.minimum(t_val + 1, val_prior_q + 1).sum().item()

    if total_budget > val_budget_used:
        return True


def get_baseline_calibrations(conditional_grid, budget_per_sample, taus_range, tau_prior, m_upper_bound,
                              cal_model_prediction, t_tilde_cal, bound_type):
    naive_allocation = NaiveBudgetAllocator(budget_per_sample, taus_range, tau_prior)
    basic_allocation = BasicBudgetAllocator(budget_per_sample, taus_range, tau_prior)
    trimmed_allocation = TrimmedBudgetAllocator(budget_per_sample, taus_range, tau_prior, m_upper_bound)
    optimized_allocation = OptimizedBudgetAllocator(budget_per_sample, taus_range, tau_prior, m_upper_bound)

    alloc_kwargs = {}
    if bound_type == 'upb':
        alloc_kwargs['reach_t_max_is_success'] = True

    adaptive_allocation = AdaptiveOptimizedBudgetAllocator(
        conditional_grid, budget_per_sample, taus_range, tau_prior, m_upper_bound, **alloc_kwargs
    )

    all_allocations: List[BudgetAllocator] = [
        basic_allocation, trimmed_allocation, optimized_allocation, adaptive_allocation
    ]

    N = len(conditional_grid)
    total_budget = budget_per_sample * N
    censored_event_time = t_tilde_cal
    tau_idx = torch.argmin(torch.abs(taus_range - tau_prior))
    prior_q = cal_model_prediction.quantile_est[:, tau_idx]

    for projection in ['platt', 'beta']:
        for score in ['prob', 'quantile']:
            for n1 in [25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 750, 1000]:
                if is_budget_sufficient_for_split(N, n1, total_budget, censored_event_time, prior_q):
                    all_allocations.append(DAPRO(conditional_grid, budget_per_sample, taus_range,
                                                 tau_prior, m_upper_bound, projection=projection,
                                                 score=score, n1=n1, **alloc_kwargs))

    if bound_type == 'lpb':
        dummy_calibration = UncalibratedLPBSurvivalCalibration(taus_range)
        all_calibrations: List[SurvivalLPBCalibration] = [dummy_calibration]
        all_calibrations.extend([SurvivalCalibrationWithKnownWeights(allocation, taus_range, tau_prior) for
                                 allocation in all_allocations])
    else:
        dummy_calibration = UncalibratedUPBSurvivalCalibration(taus_range)
        all_calibrations: List[SurvivalUPBCalibration] = [dummy_calibration]
        all_calibrations.extend([SurvivalUPBCalibrationWithKnownWeights(allocation, taus_range, tau_prior) for
                                 allocation in all_allocations])

    return all_calibrations


def get_calibration_methods(conditional_grid, budget_per_sample, taus_range, tau_prior, m_upper_bound, allocations,
                            cal_model_prediction, t_tilde_cal, device, bound_type):
    baseline_calibrations = get_baseline_calibrations(
        conditional_grid, budget_per_sample, taus_range, tau_prior,
        m_upper_bound, cal_model_prediction, t_tilde_cal, bound_type
    )
    new_allocations = get_new_allocation_algorithms(conditional_grid, budget_per_sample, taus_range, tau_prior,
                                                    m_upper_bound, allocations, device=device)

    all_allocations = new_allocations

    if bound_type == 'lpb':
        new_calibrations = [SurvivalCalibrationWithKnownWeights(allocation, taus_range, tau_prior) for allocation in
                            all_allocations]
    else:
        new_calibrations = [SurvivalUPBCalibrationWithKnownWeights(allocation, taus_range, tau_prior) for allocation in
                            all_allocations]

    all_calibrations = baseline_calibrations + new_calibrations
    return all_calibrations


def main():
    parser = argparse.ArgumentParser(description="Construct Calibrated LPB or UPB")
    parser.add_argument('--bound-type', type=str, choices=['lpb', 'upb'], default='lpb',
                        help="Whether to compute LPB or UPB")
    parser.add_argument('--seed-start', type=int, default=0)
    parser.add_argument('--seed-end', type=int, default=1)
    parser.add_argument('--allocations', type=str, default='all')
    parser.add_argument('--data-type', type=str, default='real')
    parser.add_argument('--dataset-name', type=str, default='')
    parser.add_argument('--dataset-setup', type=str, default='')
    parser.add_argument('--budget-per-sample', type=float, default=1)
    parser.add_argument('--cal-size', type=int, default=4000)
    parser.add_argument('--tau-prior', type=float, default=None, help="Prior for tau (defaults depend on bound-type)")
    parser.add_argument('--gamma', type=float, default=10)
    parser.add_argument('--device', type=str, default='cuda:0')

    args = parser.parse_args()
    args.is_real = True if args.data_type.lower() == 'real' else False
    bound_type = args.bound_type

    seed_start = args.seed_start
    seed_end = args.seed_end
    dataset_name = args.dataset_name
    data_setup = args.dataset_setup
    is_real = args.is_real
    cal_size = args.cal_size

    # Contextual initialization parameters dependent on bound-type choice
    if bound_type == 'lpb':
        tau_prior = args.tau_prior if args.tau_prior is not None else 0.56
        target_taus_list = np.arange(0.01, 0.5, 0.01)
        num_taus = 1000
        min_tau_exp, max_tau_exp = -3, -0.01
    else:
        tau_prior = args.tau_prior if args.tau_prior is not None else 0.98
        target_taus_list = 1 - np.arange(0.01, 0.5, 0.01)
        num_taus = 3000

    device = 'cuda:0' if torch.cuda.is_available() and 'cuda' in args.device else 'cpu'
    set_seeds(0)

    seeds = (seed_start, seed_end)

    if bound_type == 'lpb':
        taus_range = torch.tensor(np.logspace(min_tau_exp, max_tau_exp, num_taus)).to(device)
    else:
        taus_range = torch.tensor(np.linspace(0.5, 0.95, num_taus)).to(device)

    budget_per_sample = args.budget_per_sample

    if not is_real:
        m_upper_bound = 20
    else:
        m_upper_bound = 200

    gamma = get_gamma(m_upper_bound, budget_per_sample)
    allocations = args.allocations

    print(f"Executing for Bound Type: {bound_type.upper()}")
    print(f"budget_per_sample: {budget_per_sample}, gamma: {gamma}, m_upper_bound: {m_upper_bound}")

    # File naming adjustments for rounded inputs distinct between modes
    experiments_name = f"{dataset_name}_{data_setup}_{budget_per_sample}_{cal_size}_{tau_prior}_{np.round(gamma, 3)}"

    run_experiments(cal_size, is_real, device, dataset_name, data_setup, experiments_name, seeds,
                    taus_range, budget_per_sample, tau_prior, m_upper_bound, target_taus_list,
                    skip_existing=True, allocations=allocations, bound_type=bound_type)

    print("Finished")


if __name__ == '__main__':
    main()