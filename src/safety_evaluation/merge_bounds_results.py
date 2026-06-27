import os
import traceback
import pandas as pd
import torch
import tqdm
import argparse
import numpy as np
from concurrent.futures import as_completed, ThreadPoolExecutor

from src.safety_evaluation.calibration.survival_calibration_with_known_weights import get_gamma, SurvivalCalibrationWithKnownWeights
from src.safety_evaluation.utils.get_calibration_methods_utils import get_baseline_calibrations, get_new_allocation_algorithms
from src.safety_evaluation.utils.utils import (
    get_tmp_calibration_result_path,
    get_merged_calibration_result_path,
    get_tmp_upb_calibration_result_path,
    get_merged_upb_calibration_result_path
)
from src.utils.utils import set_seeds

# Attempt to import UPB wrapper if it exists in your repository structure
try:
    from src.safety_evaluation.calibration.survival_upb_calibration_with_known_weights import SurvivalUPBCalibrationWithKnownWeights
except ImportError:
    # Fallback to the known weights if the specific UPB class is missing from imports
    SurvivalUPBCalibrationWithKnownWeights = SurvivalCalibrationWithKnownWeights


def process_calibration(calibration, seed, experiments_name, bound_type):
    """Process a single calibration and return all computed rows."""
    try:
        if bound_type == 'lpb':
            dir_path = get_tmp_calibration_result_path(experiments_name, calibration.name)
        else:
            dir_path = get_tmp_upb_calibration_result_path(experiments_name, calibration.name)

        csv_path = f"{dir_path}/seed={seed}.csv"
        abs_path = os.path.abspath(csv_path)

        # If on Windows, prepend the "Long Path" magic string
        if os.name == 'nt' and not abs_path.startswith('\\\\?\\'):
            abs_path = f"\\\\?\\{abs_path}"
        if not os.path.exists(abs_path):
            raise Exception(f"Warning, method does not exist: {calibration.name} at {abs_path}")

        df = pd.read_csv(abs_path)
        if "Unnamed: 0" in df.columns:
            df.drop("Unnamed: 0", axis=1, inplace=True)
        return df
    except Exception as e:
        raise Exception(f"Loading calibration {calibration.name} failed with error: {traceback.print_exc()}")


def get_calibration_methods(conditional_grid, budget_per_sample, taus_range, tau_prior, m_upper_bound, allocations: str,
                            device, bound_type):
    baseline_calibrations = get_baseline_calibrations(conditional_grid, budget_per_sample, taus_range, tau_prior,
                                                      m_upper_bound)
    new_allocations = get_new_allocation_algorithms(conditional_grid, budget_per_sample, taus_range, tau_prior,
                                                    m_upper_bound,
                                                    allocations, device=device)

    all_allocations = new_allocations

    if bound_type == 'lpb':
        all_calibrations = baseline_calibrations + [
            SurvivalCalibrationWithKnownWeights(allocation, taus_range, tau_prior) for
            allocation in all_allocations]
    else:
        all_calibrations = baseline_calibrations + [
            SurvivalUPBCalibrationWithKnownWeights(allocation, taus_range, tau_prior) for
            allocation in all_allocations]

    return all_calibrations


def merge_results(experiments_name, seeds, budget_per_sample, taus_range, tau_prior, m_upper_bound, target_taus_list,
                  allocations, device, bound_type):
    all_dfs = []
    for seed in tqdm.tqdm(range(seeds[0], seeds[1]), desc="merging csvs"):

        all_calibrations = get_calibration_methods(None, budget_per_sample, taus_range, tau_prior, m_upper_bound,
                                                   allocations, device=device, bound_type=bound_type)

        num_cpus = os.cpu_count()

        with ThreadPoolExecutor(max_workers=num_cpus) as executor:
            futures = [executor.submit(process_calibration, calibration, seed, experiments_name, bound_type) for
                       calibration in all_calibrations]

            for future in as_completed(futures):
                try:
                    result_df = future.result()
                    all_dfs.append(result_df)
                except Exception as e:
                    print(f"Error processing calibration: {e}")

    if len(all_dfs) == 0:
        raise Exception("No calibrations found")

    all_df = pd.concat(all_dfs)

    if bound_type == 'lpb':
        results_dir = get_merged_calibration_result_path(experiments_name)
    else:
        results_dir = get_merged_upb_calibration_result_path(experiments_name)

    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "all_df.csv")
    all_df.to_csv(results_path, index=False)
    print(f"stored successfully at {os.path.abspath(results_path)}")


def main():
    parser = argparse.ArgumentParser(description="Merge Calibrated LPB or UPB Results")
    parser.add_argument('--bound-type', type=str, choices=['lpb', 'upb'], default='lpb',
                        help="Whether to merge LPB or UPB results")
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

    device = 'cuda:0' if torch.cuda.is_available() and 'cuda' in args.device else 'cpu'
    set_seeds(0)

    seeds = (seed_start, seed_end)

    # Contextual initialization parameters dependent on bound-type choice
    if bound_type == 'lpb':
        tau_prior = args.tau_prior if args.tau_prior is not None else 0.56
        target_taus_list = np.arange(0.01, 0.5, 0.01)
        num_taus = 1000
        min_tau_exp = -3
        max_tau_exp = -0.01
        taus_range = torch.tensor(np.logspace(min_tau_exp, max_tau_exp, num_taus)).to(device)
    else:
        tau_prior = args.tau_prior if args.tau_prior is not None else 0.98
        target_taus_list = 1 - np.arange(0.01, 0.5, 0.01)
        num_taus = 3000
        taus_range = torch.tensor(np.linspace(0.5, 0.95, num_taus)).to(device)

    budget_per_sample = args.budget_per_sample

    if not is_real:
        m_upper_bound = 20
    else:
        m_upper_bound = 200

    gamma = get_gamma(m_upper_bound, budget_per_sample)
    allocations = args.allocations

    global global_calibration_set_size
    global global_budget_per_sample
    global_calibration_set_size = cal_size
    global_budget_per_sample = budget_per_sample

    print(f"Executing for Bound Type: {bound_type.upper()}")
    print(f"budget_per_sample: {budget_per_sample}, gamma: {gamma}, m_upper_bound: {m_upper_bound}")

    # File naming adjustments distinct between modes
    experiments_name = f"{dataset_name}_{data_setup}_{budget_per_sample}_{cal_size}_{tau_prior}_{np.round(gamma, 3)}"

    merge_results(experiments_name, seeds, budget_per_sample, taus_range, tau_prior, m_upper_bound,
                  target_taus_list, allocations=allocations, device=device, bound_type=bound_type)

    print("Finished")


if __name__ == '__main__':
    main()