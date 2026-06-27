import os.path
import pandas as pd
import os

from src.safety_evaluation.calibration.survival_calibration_with_known_weights import get_gamma

import numpy as np
import matplotlib.pyplot as plt

from src.safety_evaluation.utils.utils import get_merged_calibration_result_path


def load_results(dataset_name, data_setup, budget_per_sample, cal_size, m_upper_bound, tau_prior):
    gamma = get_gamma(m_upper_bound, budget_per_sample)
    experiments_name = f"{dataset_name}_{data_setup}_{budget_per_sample}_{cal_size}_{tau_prior}_{gamma}"
    base_results_dir = get_merged_calibration_result_path(experiments_name, basedir='.')
    if not os.path.exists(base_results_dir):
        raise Exception(f"no results in path {os.path.abspath(base_results_dir)}")
    all_df = pd.read_csv(os.path.join(base_results_dir, "all_df.csv"))
    all_df['coverage_diff_target_90'] = (abs(all_df['coverage'] - 0.9)) * 100
    all_df['coverage'] = all_df['coverage'] * 100
    all_df['budget_used_per_sample'] = all_df['budget_used'] / cal_size
    return all_df


# 1. Create a helper function for directional standard deviation
def calculate_semi_stds(series):
    series = series.dropna()
    if series.empty:
        return 0.0, 0.0, 0.0

    mean_val = series.mean()

    # Separate values above and below the mean
    upper_vals = series[series > mean_val]
    lower_vals = series[series < mean_val]

    # Calculate semi-deviations relative to the overall mean
    # Using max(1, N-1) to avoid division by zero if there's only 1 or 0 points
    upper_std = np.sqrt(
        ((upper_vals - mean_val) ** 2).sum() / max(1, len(upper_vals) - 1)) if not upper_vals.empty else 0.0
    lower_std = np.sqrt(
        ((lower_vals - mean_val) ** 2).sum() / max(1, len(lower_vals) - 1)) if not lower_vals.empty else 0.0

    return mean_val, upper_std, lower_std


def plot_n1_ablation(figures_dir, dataset_name, metric_fns, metric_labels, titles, data_setup, budget_per_sample,
                     cal_size, m_upper_bound, tau_prior):
    all_df = load_results(dataset_name, data_setup, budget_per_sample, cal_size, m_upper_bound, tau_prior)
    plot_n1_ablation_aux(all_df, figures_dir, dataset_name, metric_fns, metric_labels, titles)


def plot_n1_ablation_aux(all_df, figures_dir, dataset_name, metric_fns, metric_labels, titles):
    for metric_fn, metric_label, title in zip(metric_fns, metric_labels, titles):
        df_90 = all_df[all_df['target_coverage'] == 0.9]

        # n1_values = [25, 50, 100, 150, 200, 250, 300, 400, 500, 750, 1000, 1250, 1500]
        n1_values = [25, 50, 100, 150, 200, 250, 300, 400]
        n1_values_used = []
        dapro_means = []
        dapro_upper_stds = []
        dapro_lower_stds = []
        baseline_res = df_90[df_90['calibration_name'] == 'calibration_optimized_allocation']
        baseline_mean = baseline_res[metric_fn].mean()
        baseline_std = baseline_res[metric_fn].std()
        if pd.isna(baseline_std): baseline_std = 0.0

        for n1 in n1_values:
            res = df_90[
                df_90['calibration_name'] == f'calibration_projected_optimization_platt_prob_n1_{n1}_allocation']
            res_series = res[metric_fn]
            if len(res_series) == 0:
                continue
            mean_val, upper_std, lower_std = calculate_semi_stds(res_series)

            dapro_means.append(mean_val)
            dapro_upper_stds.append(upper_std)
            dapro_lower_stds.append(lower_std)
            n1_values_used.append(n1)

        dapro_means = np.array(dapro_means)

        dapro_upper_stds = np.array(dapro_upper_stds)
        dapro_lower_stds = np.array(dapro_lower_stds)

        plt.figure(figsize=(8, 5))

        # Plot DAPRO with std
        plt.plot(n1_values_used, dapro_means, marker='o', label='DAPRO', color='tab:green', linewidth=2.5)
        plt.fill_between(n1_values_used,
                         dapro_means - dapro_lower_stds,
                         dapro_means + dapro_upper_stds, alpha=0.2, color='tab:green')

        # Plot Static Baseline with std
        plt.axhline(y=baseline_mean, color='tab:blue', linestyle='--', label='Static Baseline', linewidth=2.5)
        plt.axhspan(baseline_mean - baseline_std, baseline_mean + baseline_std, color='tab:blue', alpha=0.1, linewidth=2.5)

        plt.xlabel(r'$N_1$')
        plt.ylabel(metric_label)
        plt.title(title)
        if metric_fn == 'coverage':
            plt.legend()
        plt.grid(True)

        if figures_dir:
            os.makedirs(figures_dir, exist_ok=True)
            plt.savefig(os.path.join(figures_dir, f'{dataset_name}_{metric_fn}_n1_ablation.png'),
                        bbox_inches='tight')
        plt.show()


def plot_score_error_lambda_ablation(figures_dir, dataset_name, metric_fns, metric_labels, titles, data_setup,
                                     budget_per_sample, cal_size, m_upper_bound, tau_prior):
    all_df = load_results(dataset_name, data_setup, budget_per_sample, cal_size, m_upper_bound, tau_prior)
    plot_score_error_lambda_ablation_aux(all_df, figures_dir, dataset_name, metric_fns, metric_labels, titles)


def plot_score_error_lambda_ablation_aux(all_df, figures_dir, dataset_name, metric_fns, metric_labels, titles):
    for metric_fn, metric_label, title in zip(metric_fns, metric_labels, titles):
        df_90 = all_df[all_df['target_coverage'] == 0.9]

        lambda_values = list(np.arange(0, 1, 0.1)) + [0.95, 0.99]
        dapro_means = []
        dapro_upper_stds = []
        dapro_lower_stds = []
        baseline_res = df_90[df_90['calibration_name'] == 'calibration_optimized_allocation']
        baseline_mean = baseline_res[metric_fn].mean()
        baseline_std = baseline_res[metric_fn].std()
        if pd.isna(baseline_std): baseline_std = 0.0

        for score_error_lambda in lambda_values:
            score_error_lambda = np.round(score_error_lambda, 2)
            res = df_90[df_90[
                            'calibration_name'] == f'calibration_projected_optimization_platt_prob_lambda_{score_error_lambda}_allocation']
            res_series = res[metric_fn]
            mean_val, upper_std, lower_std = calculate_semi_stds(res_series)

            dapro_means.append(mean_val)
            dapro_upper_stds.append(upper_std)
            dapro_lower_stds.append(lower_std)


        dapro_means = np.array(dapro_means)
        dapro_upper_stds = np.array(dapro_upper_stds)
        dapro_lower_stds = np.array(dapro_lower_stds)

        plt.figure(figsize=(8, 5))

        # Plot DAPRO with std
        plt.plot(lambda_values, dapro_means, marker='o', label='DAPRO', color='tab:green', linewidth=2.5)
        # if 'mean_weight' not in metric_fn:
        plt.fill_between(lambda_values, dapro_means - dapro_lower_stds, dapro_means + dapro_upper_stds, alpha=0.2, color='tab:green', linewidth=2.5)
        if 'mean_weight' in metric_fn:
            plt.yscale('log')
        # Plot Static Baseline with std
        plt.axhline(y=baseline_mean, color='tab:blue', linestyle='--', label='Static Baseline', linewidth=2.5)
        plt.axhspan(baseline_mean - baseline_std, baseline_mean + baseline_std, color='tab:blue', alpha=0.1)

        plt.xlabel(r'$\lambda$')
        plt.ylabel(metric_label)
        plt.title(title)
        if metric_fn == 'coverage':
            plt.legend()
        plt.grid(True)

        if figures_dir:
            os.makedirs(figures_dir, exist_ok=True)
            plt.savefig(os.path.join(figures_dir, f'{dataset_name}_{metric_fn}_lambda_ablation.png'),
                        bbox_inches='tight')
        plt.show()


def plot_budget_per_sample_ablation(figures_dir, dataset_name, metric_fns, metric_labels, titles, data_setup, cal_size,
                                    m_upper_bound, tau_prior):
    plot_budget_per_sample_ablation_aux(figures_dir, dataset_name, metric_fns, metric_labels, titles, data_setup,
                                        cal_size, m_upper_bound, tau_prior)


def plot_budget_per_sample_ablation_aux(figures_dir, dataset_name, metric_fns, metric_labels, titles, data_setup,
                                        cal_size, m_upper_bound, tau_prior):
    budget_values = [5, 6,7,8,9, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    for metric_fn, metric_label, title in zip(metric_fns, metric_labels, titles):
        dapro_means = []
        baseline_means = []
        dapro_upper_stds = []
        dapro_lower_stds = []
        baseline_upper_stds = []
        baseline_lower_stds = []
        used_budget_values = []
        for budget in budget_values:
            budget = float(budget)
            all_df = load_results(dataset_name, data_setup, budget, cal_size, m_upper_bound, tau_prior)
            df_90 = all_df[all_df['target_coverage'] == 0.9]

            # Baseline calculation per budget
            baseline_res = df_90[df_90['calibration_name'] == 'calibration_optimized_allocation']
            res_series = baseline_res[metric_fn]
            if budget <= 10:
                res = df_90[df_90['calibration_name'] == f'calibration_projected_optimization_platt_prob_n1_25_allocation']
            elif budget == 15:
                res = df_90[df_90['calibration_name'] == f'calibration_projected_optimization_platt_prob_n1_50_allocation']
            else:
                res = df_90[df_90['calibration_name'] == 'calibration_projected_optimization_platt_prob_allocation']
            dapro_res_series = res[metric_fn]
            if len(dapro_res_series) == 0 or len(res_series) == 0:
                continue
            used_budget_values.append(budget)
            mean_val, upper_std, lower_std = calculate_semi_stds(res_series)
            baseline_means.append(mean_val)
            baseline_upper_stds.append(upper_std)
            baseline_lower_stds.append(lower_std)

            # DAPRO calculation per budget

            mean_val, upper_std, lower_std = calculate_semi_stds(dapro_res_series)

            dapro_means.append(mean_val)
            dapro_upper_stds.append(upper_std)
            dapro_lower_stds.append(lower_std)

        dapro_means = np.array(dapro_means)
        dapro_upper_stds = np.array(dapro_upper_stds)
        dapro_lower_stds = np.array(dapro_lower_stds)

        baseline_means = np.array(baseline_means)
        baseline_upper_stds = np.array(baseline_upper_stds)
        baseline_lower_stds = np.array(baseline_lower_stds)
        plt.figure(figsize=(8, 5))

        # DAPRO line with std
        plt.plot(used_budget_values, dapro_means, marker='o', label='DAPRO', color='tab:green', linewidth=2.5)
        plt.fill_between(used_budget_values, dapro_means - dapro_lower_stds, dapro_means + dapro_upper_stds, alpha=0.2, color='tab:green')

        # Static Baseline line with std (varying over budget)
        plt.plot(used_budget_values, baseline_means, marker='x', linestyle='--', color='tab:blue', label='Static Baseline', linewidth=2.5)
        plt.fill_between(used_budget_values, baseline_means - baseline_lower_stds, baseline_means + baseline_upper_stds, color='tab:blue',
                         alpha=0.1)

        plt.xlabel('Budget per Sample')
        plt.ylabel(metric_label)
        plt.title(title)
        if metric_fn == 'coverage':
            plt.legend()
        plt.grid(True)

        if figures_dir:
            os.makedirs(figures_dir, exist_ok=True)
            plt.savefig(os.path.join(figures_dir, f'{dataset_name}_{metric_fn}_budget_ablation.png'),
                        bbox_inches='tight')
        plt.show()


def main():
    dataset_name = 'dataset_toxicity'
    data_setup = 'attack_toxic_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_detoxify'
    # data_setup = 'attack_toxic_attack_qwen25_14b_instruct_lm_target_llama_31_8B_instruct_judge_detoxify'
    # data_setup = 'attack_toxic_attack_qwen25_14b_instruct_lm_target_mini_phi_4_instruct_judge_detoxify'
    # data_setup = 'attack_toxic_attack_qwen25_14b_instruct_lm_target_gemma3_4b_it_judge_detoxify'
    budget_per_sample = 20.
    cal_size = 3000
    m_upper_bound = 200
    tau_prior = 0.56
    plt.rcParams.update({'font.size': 24})
    """
    'seed', 'calibration_name', 'coverage_deviation',
       'prior_observed_jailbreaks', 'prior_observed_f_lower_c',
       'prior_observed_both', 'n_observed_events', 'n_achieved_q_prior1',
       'n_achieved_q_prior2', 'budget_used', 'mean_weight', 'max_weight',
       'val_obj', 'val_obj2', 'val_icw', 'test_icw', 'test_icw2', 'all_icw',
       'valid_budget', 'val_budget', 'coverage', 'size', 'target_coverage',
       'all_observed_jailbreaks', 'all_f_lower_c', 'all_observed_both',
       'alpha_hat_per_tau', 'coverage_diff_target_90'],
    """
    figures_dir = "./figures/ablation_results/"
    metric_fns = ['coverage_diff_target_90', 'n_observed_events', 'mean_weight', 'coverage', 'budget_used_per_sample']
    metric_labels = ['|Coverage - Target| (%)', 'Observed Events', 'Mean Weight', 'Coverage (%)', 'Budget Used']
    titles = ['Coverage Difference', '# Observed Events', 'Mean Weight', 'Coverage Rate', 'Budget Used per Sample']

    # n1 plots
    # titles_n1 = ['N1 Coverage Diff', 'N1 n_observed_events', 'N1 Mean Weight', 'N1 Coverage']
    plot_n1_ablation(figures_dir, dataset_name, metric_fns, metric_labels, titles, data_setup, budget_per_sample,
                     cal_size, m_upper_bound, tau_prior)

    # lambda plots
    # titles_lambda = ['Lambda Coverage Diff', 'Lambda n_observed_events', 'Lambda Mean Weight', 'Lambda Coverage']
    plot_score_error_lambda_ablation(figures_dir, dataset_name, metric_fns, metric_labels, titles, data_setup,
                                     budget_per_sample, cal_size, m_upper_bound, tau_prior)

    # budget plots
    plot_budget_per_sample_ablation(figures_dir, dataset_name, metric_fns, metric_labels, titles, data_setup,
                                    cal_size, m_upper_bound, tau_prior)

    print("Finished")


if __name__ == '__main__':
    main()
