import os
import argparse
import traceback
import pandas as pd
import numpy as np
import torch
import tqdm
from concurrent.futures import as_completed, ThreadPoolExecutor

from src.safety_evaluation.budget_allocators.adaptive_optimized_allocator import AdaptiveOptimizedBudgetAllocator
# --- Your Imports ---
# Make sure these match your actual project structure
from src.safety_evaluation.budget_allocators.naive_allocator import NaiveBudgetAllocator
from src.safety_evaluation.budget_allocators.optimized_allocators import OptimizedBudgetAllocator
from src.safety_evaluation.utils.get_calibration_methods_utils import get_metric_allocators
from src.safety_evaluation.utils.utils import get_tmp_metric_calibration_result_path, get_merged_metric_calibration_result_path





def process_calibration(calibration, seed, experiments_name):
    """Process a single calibration and return all computed rows."""
    try:
        dir_path = get_tmp_metric_calibration_result_path(experiments_name, calibration.name)
        csv_path = os.path.join(dir_path, f"seed={seed}.csv")
        abs_path = os.path.abspath(csv_path)

        # Handle Windows long paths
        if os.name == 'nt' and not abs_path.startswith('\\\\?\\'):
            abs_path = f"\\\\?\\{abs_path}"

        if not os.path.exists(abs_path):
            # Print warning but do not crash the ThreadPool
            print(f"Warning: File {abs_path} does not exist for {calibration.name} at seed {seed}")
            return None

        df = pd.read_csv(abs_path)
        if "Unnamed: 0" in df.columns:
            df.drop("Unnamed: 0", axis=1, inplace=True)

        return df

    except Exception as e:
        print(f"Loading calibration {calibration.name} failed at seed {seed}")
        traceback.print_exc()
        return None


def merge_results(experiments_name, seeds, budget_per_sample, taus_range, tau_prior, m_upper_bound,device):
    all_dfs = []
    num_cpus = os.cpu_count()

    # Pass None for conditional_grid to avoid loading data just for names
    all_calibrations = get_metric_allocators(
        conditional_grid=None,
        budget_per_sample=budget_per_sample,
        taus_range=taus_range,
        tau_prior=tau_prior,
        m_upper_bound=m_upper_bound,
        device=device
    )

    print(f"Starting merge for experiment: {experiments_name}")
    print(f"Discovered {len(all_calibrations)} unique calibration methods.")

    with ThreadPoolExecutor(max_workers=num_cpus) as executor:
        futures = []

        for seed in range(seeds[0], seeds[1]):
            for calibration in all_calibrations:
                futures.append(
                    executor.submit(process_calibration, calibration, seed, experiments_name)
                )

        # Process with tqdm progress bar
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Merging CSVs"):
            try:
                result_df = future.result()
                if result_df is not None and not result_df.empty:
                    all_dfs.append(result_df)
            except Exception as e:
                print(f"Error resolving future: {e}")

    if len(all_dfs) == 0:
        raise Exception("No calibrations found. Check your file paths and seed ranges.")

    all_df = pd.concat(all_dfs, ignore_index=True)

    results_dir = get_merged_metric_calibration_result_path(experiments_name)
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "all_df.csv")

    all_df.to_csv(results_path, index=False)
    print(f"\nSuccessfully stored merged dataset at: {os.path.abspath(results_path)}")


def main():
    parser = argparse.ArgumentParser(description="Merge dynamic calibration results.")
    parser.add_argument('--seed-start', type=int, default=0)
    parser.add_argument('--seed-end', type=int, default=50)
    parser.add_argument('--data-type', type=str, default='real')
    parser.add_argument('--dataset-name', type=str, default='')
    parser.add_argument('--dataset-setup', type=str, default='')
    parser.add_argument('--budget-per-sample', type=float, default=40)
    parser.add_argument('--cal-size', type=int, default=4000)
    parser.add_argument('--tau-prior', type=float, default=0.56)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    # Reconstruct variables required for allocator initialization
    device = 'cuda:0' if torch.cuda.is_available() and 'cuda' in args.device else 'cpu'

    num_taus = 1000
    max_tau_exp = -0.01
    min_tau_exp = -3
    taus_range = torch.tensor(np.logspace(min_tau_exp, max_tau_exp, num_taus)).to(device)

    is_real = True if args.data_type.lower() == 'real' else False
    m_upper_bound = 200 if is_real else 20

    # We use your naming convention from the original merge script logic
    experiments_name = f"{args.dataset_name}_{args.dataset_setup}_{args.budget_per_sample}_safety_metrics"

    merge_results(
        experiments_name=experiments_name,
        seeds=(args.seed_start, args.seed_end),
        budget_per_sample=args.budget_per_sample,
        taus_range=taus_range,
        tau_prior=args.tau_prior,
        m_upper_bound=m_upper_bound,
        device=device
    )


if __name__ == '__main__':
    main()
