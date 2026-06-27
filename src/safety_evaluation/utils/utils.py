import gc
import random
from typing import Union, List

import numpy as np
import json
import torch
from lifelines.utils import concordance_index
from scipy.stats import kstest
from lifelines.statistics import logrank_test  # New import
from sksurv.metrics import integrated_brier_score
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
)
from tqdm import tqdm

from src.train_model.acquisition_strategies.dummy_acquisition import DummyAcquisition
from src.train_model.active_learning import ActiveLearner
from src.dataset_utils.datasets import PartialSequenceDataset
from src.safety_evaluation.survival_utils.compute_mean_time_given_pmf import compute_mean_survival_time, \
    compute_quantile_survival_time
from src.safety_evaluation.survival_utils.conditional_pmf_utils import get_conditional_pmf
from src.safety_evaluation.survival_utils.quantiles import compute_conditional_quantiles_single_step
from src.train_model.models.transformer_survival_model import TransformerSurvivalModel, DiscreteSurvivalLoss
from src.utils.utils import set_seeds


def get_model(model_save_path, model_figure_save_path, is_real, x_train, t_tilde_train, device, features_size: int,
              max_time: int, dataset_name, data_setup, ):
    dropout = 0.1
    y_train = torch.zeros(x_train.shape[0], x_train.shape[1], device=device)
    y_train[list(range(len(x_train))), torch.minimum(torch.zeros_like(t_tilde_train.long()), t_tilde_train.long())] = 1
    training_data = PartialSequenceDataset(x_train, y_train, t_tilde_train, dataset_name=data_setup, initial_obs_len=1)

    dropout = 0.2
    model_class = lambda: TransformerSurvivalModel(x_train.shape[-1], x_train.shape[1], dropout)
    learner = ActiveLearner(model_class=model_class,
                            loss_fn=None,
                            dataset=training_data,
                            seed_indices=[],
                            val_indices=[],
                            pool_indices=[],
                            acquisition=DummyAcquisition(),
                            device=device,
                            retrain_from_scratch=True,
                            verbose=True)
    initial_total_budget = 10
    n_seed = int(len(x_train) * 0.9)
    save_path_suffix = os.path.join('transformer', data_setup, learner.acquisition.name,
                                    f'seed_{n_seed}_budget_{initial_total_budget}', f"seed={0}")
    saved_models_dir = os.path.join('./saved_models/al', save_path_suffix)
    loaded_model, last_round = learner.load_state(saved_models_dir, update_steps=False)
    loaded_model.eval()
    loaded_model = loaded_model.to(device)

    return loaded_model


def compute_probabilities_and_quantiles(x_cal, x_train, x_test, model_cal_test_preds_path, dataset_name, data_setup, p_cal, p_test, max_time,
            model_save_path, model_figure_save_path, is_real, t_tilde_train,
            taus_range, m_upper_bound, device):
    current_time = 0
    if 'synthetic' in dataset_name and 'oracle' in data_setup:
        probability_est = torch.cat([p_cal, p_test]).clone().to(device)
        noise_levels = 1 / (10 * (torch.arange(0, max_time, device=device) + 1))
        conditional_grid = get_conditional_pmf(probability_est, validate_logic=True)
        conditional_grid = conditional_grid * (1 - noise_levels) + torch.rand_like(conditional_grid) * noise_levels
        # conditional_grid[:, 0] = conditional_grid[:, 0] * (1-0.1) + torch.rand_like(conditional_grid[:, 0]) * 0.1
        quantile_est_cal_test = torch.zeros(conditional_grid.shape[0], len(taus_range), device=device)
        for i, tau in tqdm(enumerate(taus_range), desc="computing quantiles for all taus"):
            quantile_est_cal_test[:, i] = compute_conditional_quantiles_single_step(conditional_grid[:, current_time],
                                                                                    tau.item())
        quantile_est_cal_test = quantile_est_cal_test.clip(max=max_time)
    else:
        if os.path.exists(model_cal_test_preds_path):
            probability_est = torch.load(model_cal_test_preds_path).to(device)
            del x_cal
            del x_train
            del x_test
        else:
            loaded_model = get_model(model_save_path, model_figure_save_path, is_real, x_train, t_tilde_train, device,
                                     x_cal.shape[-1], max_time, dataset_name, data_setup)
            print("loaded model")
            x_cal_test = torch.cat([x_cal, x_test]).clone().to(device)
            loaded_model = loaded_model.to(device)
            batch_size = 1028
            probs_list = []
            # 2. Main Loop
            with torch.no_grad():
                total_samples = x_cal_test.shape[0]

                for i in range(0, total_samples, batch_size):
                    # Create a batch slice
                    end_idx = min(i + batch_size, total_samples)
                    batch = x_cal_test[i:end_idx]

                    # Move only this small batch to GPU
                    batch_gpu = batch.float().to(device)

                    # Predict
                    output = loaded_model.predict_proba(batch_gpu)

                    # Move result to CPU immediately to free GPU space
                    probs_list.append(output)

                    # Clean up batch variables explicitly
                    del batch_gpu
            del x_cal_test
            # 3. Delete original data as requested
            x_cal_test = None
            gc.collect()  # Force garbage collection
            torch.cuda.empty_cache()
            # 4. Recombine results
            probability_est = torch.cat(probs_list, dim=0)
            # probability_est = loaded_model.predict_proba(x_cal_test.float()).clone()
            torch.save(probability_est.cpu(), model_cal_test_preds_path)

        conditional_grid = probability_est
        quantile_est_cal_test = torch.zeros(conditional_grid.shape[0], len(taus_range), device=device)
        for i, tau in tqdm(enumerate(taus_range), desc="computing quantiles for all taus"):
            quantile_est_cal_test[:, i] = compute_quantile_survival_time(conditional_grid[:, current_time].unsqueeze(1),
                                                                         quantile=tau.item(),
                                                                         tail_distribution='geometric').squeeze()

        quantile_est_cal_test = quantile_est_cal_test.clamp(max=m_upper_bound)
    return quantile_est_cal_test, probability_est, conditional_grid


def split_data(seed, cal_size, test_size, x_cal_test, t_tilde_cal_test, probability_est, quantile_est_cal_test):
    set_seeds(seed)
    perm = np.random.permutation(cal_size + test_size)
    cal_idx = perm[:cal_size]
    test_idx = perm[cal_size:]
    if x_cal_test is not None:
        x_cal = x_cal_test[cal_idx].detach()
        x_test = x_cal_test[test_idx].detach()
    else:
        x_cal = None
        x_test = None

    t_tilde_cal = t_tilde_cal_test[cal_idx].detach()
    quantile_est_cal = quantile_est_cal_test[cal_idx].detach()
    probability_est_cal = probability_est[cal_idx].detach()

    t_tilde_test = t_tilde_cal_test[test_idx].detach()
    quantile_est_test = quantile_est_cal_test[test_idx].detach()
    probability_est_test = probability_est[test_idx].detach()

    return x_cal, x_test, t_tilde_cal, probability_est_cal, quantile_est_cal, t_tilde_test, quantile_est_test, \
           probability_est_test, cal_idx, test_idx


def geom_cdf_stable(p: float, k: np.ndarray):
    """
    Numerically stable CDF of Geometric(p) at k (support 1,2,3,...).
    """
    # computes 1 - (1-p)^k accuratly
    cdf = -np.expm1(k * np.log1p(-p))
    return cdf.clip(0, 1)


def inspect_hazards_and_errors(model, x_test, p_test, t_tilde_test, num_samples=20, outdir: str = None):
    """
    For a random subset of test samples:
      - Plot true hazard p_test[i,t] (dashed) vs. predicted p_hat[i,t] (solid)
      - Compute per-sample integrated absolute error across time
      - Print mean & std of these errors
      - Show a histogram of the error distribution
    """
    device = next(model.parameters()).device
    N, T = p_test.shape

    # 1) Select a random subset of indices
    idx = np.random.permutation(N)[:num_samples]

    # 2) Predict hazards on entire test set (so we can compute errors)
    x_all = torch.tensor(x_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        p_hat_all = model(x_all)  # shape (N, T)

    # 3) Compute integrated absolute error per sample
    errors = torch.abs(p_hat_all - p_test).sum(dim=1).detach().cpu().numpy()  # shape (N,)
    print(f'Integrated absolute error over {T} time‑steps:')
    print(f'Mean = {errors.mean():.4f},  Std = {errors.std():.4f}')

    # 4) Histogram of errors
    plt.figure(figsize=(6, 4))
    plt.hist(errors, bins=30, edgecolor='k', alpha=0.7)
    plt.xlabel('Integrated |p_hat - p_true|')
    plt.ylabel('Count')
    plt.title('Distribution of Per‑Sample Event')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 5) Plot true vs. predicted hazards for selected samples
    times = np.arange(1, T + 1)
    p_test = p_test.detach().cpu().numpy()
    p_hat_all = p_hat_all.detach().cpu().numpy()
    for i in idx:
        plt.figure(figsize=(6, 3))
        plt.plot(times, p_test[i], '--', label='True Event')
        plt.plot(times, p_hat_all[i], label='Predicted Event')
        plt.xlabel('Time step')
        plt.ylabel('Event probability')
        plt.title(f'Sample {i} (first success at t={t_tilde_test[i]})')
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def get_probs(budget_per_sample, prior_quantile_est, needed_prob=1):
    C_probs = budget_per_sample / prior_quantile_est
    above = C_probs > needed_prob
    below = C_probs < needed_prob
    while above.any() and below.any():
        leftover = ((C_probs[above] - needed_prob) * prior_quantile_est[above]).sum()
        C_probs[above] = needed_prob
        below = C_probs < needed_prob
        above = C_probs > needed_prob
        # Distribute the leftover budget to the bellow one, proportionally to their prior quantile estimate
        leaftover_per_sample = leftover / below.sum()
        C_probs[below] += leaftover_per_sample / prior_quantile_est[below]
        below = C_probs < needed_prob
        above = C_probs > needed_prob
    C_probs = np.minimum(C_probs, 1)
    return C_probs


def estimate_t_alpha_per_sample(probability_est: torch.Tensor,
                                alphas: Union[torch.Tensor, np.ndarray, float, List[float]],
                                max_time: Union[None, int, list, np.ndarray, torch.Tensor] = None,
                                is_oracle=False) -> torch.Tensor:
    probability_est = probability_est.clone()
    max_t = probability_est.shape[1]
    if not is_oracle and max_time is not None:
        if type(max_time) == int:
            if max_time < max_t:
                probability_est[:, max_time:] = probability_est[:, max_time].unsqueeze(1)
        else:
            max_time = torch.Tensor(max_time)
            for j in range(len(max_time)):
                probability_est[j, max_time[j]:] = probability_est[j, max_time[j]].unsqueeze(1)

    probability_est = probability_est.to(torch.float64)
    N, n = probability_est.shape
    if type(alphas) == np.ndarray or type(alphas) == List:
        alphas = torch.tensor(alphas, dtype=torch.float64, device=probability_est.device)
    if type(alphas) == float:
        alphas = torch.tensor([alphas], dtype=torch.float64, device=probability_est.device)
    t_alphas = torch.zeros((N, len(alphas)), dtype=torch.float64, device=probability_est.device)
    survival = torch.concatenate(
        [torch.ones(len(probability_est), 1).to(probability_est.device), torch.cumprod(1 - probability_est, dim=1)],
        dim=1)
    cdf = 1 - survival[:, 1:]

    N, n = cdf.shape
    G = torch.cat([torch.zeros(N, 1, device=cdf.device), cdf], dim=1)  # shape (N,n+1)

    for j, alpha in enumerate(alphas):
        # find first t in {1..n} such that F[:,t] >= alpha
        ge = (G >= alpha)  # shape (N,n)
        has_crossed = ge.any(dim=1)  # (N,) bool

        # initialize output
        t_mid = torch.empty(N, device=cdf.device, dtype=torch.long)

        # for those that do cross
        idx = torch.argmax(ge.float(), dim=1)  # candidate t*
        idx = torch.clamp(idx, 1, n + 1)
        idx_valid = idx[has_crossed]  # shape (~M,)
        L = G[has_crossed, idx_valid - 1]  # F(t*-1)
        U = G[has_crossed, idx_valid]  # F(t*)
        frac = (U - alpha) / (U - L)  # shape (~M,)

        u = torch.rand_like(frac)
        pick_lower = (u < frac)
        t_lower = idx_valid - 1
        t_upper = idx_valid
        t_mid_valid = t_lower  # torch.where(pick_lower, t_lower, t_upper) + 1

        t_mid[has_crossed] = t_mid_valid
        S_n = survival[~has_crossed, -1]  # shape (M,)
        r = (1 - alpha) / S_n  # shape (M,) = probability to choose t=n
        u = torch.rand_like(r)
        t_mid[~has_crossed] = max_t + 1  # torch.where(u < r, n, max_t + 1)
        t_alphas[:, j] = t_mid.clone()

    return t_alphas


# def clear():
#     import shutil
#     import os
#     if os.path.exists('model.pth'):
#         os.remove('model.pth')
#     shutil.rmtree('q_est', ignore_errors=True)
#     shutil.rmtree('../alg_playground_data', ignore_errors=True)
#     print("cleared")


from lifelines import KaplanMeierFitter


def evaluate_model_alignment(p, x_test, t_test, delta_test, device='cuda', outdir=None):
    """
    Comprehensive evaluation of Discrimination and Calibration.
    """

    # Compute CDF and Survival S(t)
    # S(t) = cumprod(1 - p)
    S = torch.cumprod(1 - p, dim=1).cpu().numpy()
    t_test = t_test.long().cpu().detach().numpy()
    delta_test = delta_test.bool().cpu().detach().numpy()
    # CDF(t) = 1 - S(t)
    CDF = 1 - S

    # --- Metric 1: D-Calibration (PIT Histogram) ---
    # We only look at Uncensored patients for this specific check
    uncensored_indices = np.where(delta_test == 1)[0]

    # Get the CDF value predicted by the model at the exact moment the event occurred
    # t_test is 1-based, so we use t-1 for 0-based indexing
    event_times_indices = t_test[uncensored_indices] - 1

    # Clip to max time index just in case
    max_idx = CDF.shape[1] - 1
    event_times_indices = np.clip(event_times_indices, 0, max_idx)

    # Gather the CDF values: "What probability did the model assign to time 't'?"
    pit_values = CDF[uncensored_indices, event_times_indices]

    # Plotting D-Calibration
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(pit_values, bins=10, range=(0, 1), density=True, alpha=0.7, color='purple', edgecolor='black')
    plt.axhline(y=1.0, color='r', linestyle='--', label='Perfect Calibration')
    plt.title("D-Calibration (PIT Histogram)\nShould be Flat/Uniform")
    plt.xlabel("Predicted CDF at Event Time")
    plt.ylabel("Frequency")
    plt.legend()

    # --- Metric 2: Visual Calibration at Median Time ---
    # We compare the "Average Predicted Curve" vs "Kaplan Meier" for the whole population

    plt.subplot(1, 2, 2)

    # A. Actual Data (Kaplan Meier)
    kmf = KaplanMeierFitter()
    kmf.fit(t_test, event_observed=delta_test, label="Actual (KM)")
    kmf.plot_survival_function(color='black', linewidth=2)

    # B. Model Average
    # We simply take the mean of all predicted S(t) curves
    avg_model_survival = np.mean(S, axis=0)
    plt.plot(avg_model_survival, color='blue', linestyle='--', linewidth=2, label="Model Average Prediction")

    plt.title("Population Calibration\nAverage Predicted S(t) vs Actual KM")
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    if outdir is not None:
        plt.savefig(os.path.join(outdir, "D-Calibration.png"), bbox_inches='tight', dpi=300)
    plt.show()

    # --- Metric 3: Point-Estimate Error (MAE) ---
    # Only for uncensored patients: comparisons of Median Predicted Time vs Actual Time
    # Get predicted median time (first time S(t) <= 0.5)
    # Using argmax on boolean gives first True index
    predicted_medians = (S <= 0.5).argmax(axis=1)
    # Handle cases where it never drops below 0.5 (set to max_time)
    unfinished_mask = (S[:, -1] > 0.5)
    predicted_medians[unfinished_mask] = S.shape[1]

    # Calculate MAE only for uncensored
    errors = np.abs(predicted_medians[uncensored_indices] - t_test[uncensored_indices])
    mae = np.mean(errors)

    print(f"Mean Absolute Error (Uncensored only): {mae:.2f} time steps")

    return pit_values, mae


def plot_survival_and_quantile(p, t_sample=None, delta_sample=None, alpha=0.5, outdir=None):
    """
    p: (1, T, F) tensor - probability of event for each time
    t_sample: scalar - actual time of event (optional)
    delta_sample: scalar - event indicator (optional)
    alpha: float - quantile level (e.g., 0.5 for median)
    """

    p = p.cpu()

    # 2. Compute Survival Curve S(t)
    # S(t) = cumprod(1 - p)
    # We prepend 1.0 at t=0
    S_curve = torch.cumprod(1 - p, dim=1).cpu().numpy().flatten()
    S_curve = np.insert(S_curve, 0, 1.0)  # S(0) = 1.0

    # 3. Use your function to estimate quantile
    # Note: We pass p, not S
    t_est = estimate_t_alpha_per_sample(p, alpha).item()

    # 4. Plotting
    times = np.arange(len(S_curve))

    plt.figure(figsize=(10, 6))

    # Plot S(t)
    plt.step(times, S_curve, where='post', label='Predicted S(t)', color='blue', linewidth=2)

    # Plot Threshold (1 - alpha)
    threshold = 1 - alpha
    plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold (1-alpha={round(threshold, 3)})')

    # Plot Estimated Quantile
    # Note: If your function returns 0-based index, we plot it directly.
    # If it corresponds to time "t", it usually aligns with the drop.
    plt.axvline(x=t_est, color='green', linestyle='-', linewidth=2, label=f'Estimated t_alpha: {t_est}')

    # Plot Actual Event (if provided)
    if t_sample is not None:
        color = 'black' if delta_sample == 1 else 'gray'
        label = f'Actual Event (T={t_sample})' if delta_sample == 1 else f'Censored Time (C={t_sample})'
        marker = 'x' if delta_sample == 1 else 'o'
        plt.scatter(t_sample, S_curve[int(t_sample)] if t_sample < len(S_curve) else 0,
                    color=color, s=100, zorder=5, label=label, marker=marker)

    plt.title(f"Survival Curve Analysis (Alpha={alpha})")
    plt.xlabel("Time Step")
    plt.ylabel("Survival Probability S(t)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if outdir is not None:
        plt.savefig(f"{outdir}/survival_curve.png", bbox_inches='tight', dpi=300)
    plt.show()

    # Numeric Debug
    print(f"Threshold: {threshold}")
    print(f"S(t_est): {S_curve[int(t_est)] if t_est < len(S_curve) else 'N/A'}")
    print(f"S(t_est+1): {S_curve[int(t_est) + 1] if t_est + 1 < len(S_curve) else 'N/A'}")


import torch
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_patient_trajectory(idx, estimated_mean_time, t_test, mask, output_dir):
    """
    Plots estimated mean time vs time steps for a single patient idx.
    If the mask is empty (no valid times), it plots all time steps and notes it in the title.
    """
    # 1. Prepare Data: Move to CPU and convert to Numpy
    y_values = estimated_mean_time[idx].detach().cpu().numpy()
    actual_event_time = t_test[idx].detach().cpu().item()
    patient_mask = mask[idx].detach().cpu().numpy().astype(bool)

    # Create Time Grid (1 to T)
    T = len(y_values)
    time_grid = np.arange(0, T)

    # 2. Apply Mask and Handle Empty Case
    valid_times = time_grid[patient_mask]
    valid_predictions = y_values[patient_mask]

    # Check if we have any valid times
    if len(valid_times) == 0:
        # Fallback: Use ALL times
        plot_times = time_grid
        plot_values = y_values
        title_note = "\n(Showing all times - no hazard event observed)"
        line_style = ':'  # Optional: differentiate visually
    else:
        # Standard: Use masked times
        plot_times = valid_times
        plot_values = valid_predictions
        title_note = ""
        line_style = '-'

    # 3. Plotting
    plt.figure(figsize=(10, 6))

    # Plot the trajectory
    plt.plot(plot_times, plot_values, marker='o', linestyle=line_style, label='Estimated Mean Time')

    plt.plot(time_grid, time_grid, color='green', linestyle='--', alpha=0.3, label='Reference (y=x)')

    # Plot the Actual Time (t_test)
    plt.axhline(y=actual_event_time, color='r', linestyle='--', label=f'Actual Time (t={actual_event_time:.0f})')

    # Mark the actual time on X-axis for reference
    plt.axvline(x=actual_event_time, color='gray', linestyle=':', alpha=0.5)

    plt.xlabel('Time Step (t)')
    plt.ylabel('Estimated Mean Time')
    plt.title(f'Trajectory for Sample Index {idx}{title_note}')
    plt.xlim(0 - 0.5, len(plot_values) + 1)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save plot
    save_path = os.path.join(output_dir, f'trajectory_sample_{idx}.png')
    plt.savefig(save_path)
    plt.close()
    # plt.show()


def visualize_metrics(estimated_mean_time, x_test, t_test, delta_test, output_dir):
    uncensored_idx = np.where(delta_test.cpu() == 1)[0]
    if len(uncensored_idx) == 0:
        return {}

    batch_size, T, _ = x_test.shape
    device = x_test.device

    # Create time grid [1, 2, ..., T]
    time_grid = torch.arange(1, T + 1, device=device).unsqueeze(0)  # (1, T)

    # Mask: 1 if this time step matters for this patient, 0 otherwise
    mask = (time_grid <= t_test.unsqueeze(1)).float()
    mask = mask * delta_test.unsqueeze(-1).repeat(1, mask.shape[1])
    mask = mask.bool()

    # --- NEW: Plot for a specific index ---
    idx1 = ((t_test < T) & (t_test > 10) & delta_test).float().argmax().item()
    idx2 = ((t_test < T) & (t_test >= 15) & delta_test).float().argmax().item()
    idx3 = ((t_test < T) & (t_test >= 20) & delta_test).float().argmax().item()
    idx4 = (~delta_test).float().argmax().item()
    for idx in [idx1, idx2, idx3, idx4]:
        plot_patient_trajectory(idx, estimated_mean_time, t_test, mask, output_dir)


def get_sklearn_compatible_data(t_true, e_true):
    # scikit-survival expects a structured array of boolean event + float time
    dtype = [('event', bool), ('time', float)]
    structured_data = np.array([(bool(e), float(t)) for e, t in zip(e_true, t_true)], dtype=dtype)
    return structured_data


def get_brier_score(test_pmf, train_time_tensor, train_event_tensor, test_time_tensor, test_event_tensor):
    # ==========================================
    # 1. Prepare Training Data (Required for IPCW)
    # ==========================================
    # IBS needs the training data to estimate the censoring distribution
    train_structured = get_sklearn_compatible_data(train_time_tensor.cpu().numpy(),
                                                   train_event_tensor.cpu().numpy())

    # ==========================================
    # 2. Prepare Test Data & Predictions
    # ==========================================
    test_structured = get_sklearn_compatible_data(test_time_tensor.cpu().numpy(),
                                                  test_event_tensor.cpu().numpy())

    # Survival Probability Matrix [N_test, N_times]
    # DeepHit outputs PMF. Convert PMF to Survival Function:
    # Survival = 1 - CDF (Cumulative Sum of PMF)
    pmf_numpy = test_pmf.cpu().detach().numpy()
    surv_probs = 1.0 - np.cumsum(pmf_numpy, axis=1)

    # ... (Previous data preparation steps are correct) ...

    # 1. Get the valid time range from the test set
    # The error message [1.0; 20.0[ implies times must be < max_test_time
    min_test_time = test_structured['time'].min()
    max_test_time = test_structured['time'].max()

    # 2. Define the full grid based on model output
    full_times_grid = np.arange(pmf_numpy.shape[1]).astype(float)

    # 3. Create a mask for valid evaluation times
    # We usually want times strictly less than the max to avoid boundary issues with IPCW
    mask_valid = (full_times_grid >= min_test_time) & (full_times_grid < max_test_time)

    valid_times_grid = full_times_grid[mask_valid]

    # 4. Slice the predictions to match these times
    # surv_probs shape is [N_samples, N_times]
    # We only keep the columns corresponding to valid_times_grid
    valid_surv_probs = surv_probs[:, mask_valid]

    # 5. Calculate IBS
    if len(valid_times_grid) > 0:
        ibs = integrated_brier_score(train_structured, test_structured, valid_surv_probs, valid_times_grid)
        print(f"Integrated Brier Score: {ibs:.4f}")
    else:
        ibs = 0.0
        print("Error: No valid time points found. Test data range is too small compared to model horizon.")

    return ibs


def get_dynamic_mean_time_trajectory(model, x, valid_lengths, max_time_horizon=None):
    """
    Calculates the expected time-to-event for every patient at every time step,
    updating the prediction as more history becomes available.

    Args:
        model: Trained DeepHit-style model. Must accept input 'x' and return (pmf, ...).
        x: Input features. Shape [Batch, Max_Seq_Len, Features].
        valid_lengths: The actual observed length for each patient (censor/event time).
                       Shape [Batch]. Used to mask output after patient is gone.
        max_time_horizon: (Optional) The size of the output distribution (number of time bins).
                          If None, inferred from model output.

    Returns:
        mean_trajectory: Shape [Batch, Max_Seq_Len].
                         entry [i, t] is the Expected Event Time for patient i,
                         estimated using ONLY history x[i, :t+1].
    """
    model.eval()
    device = x.device
    batch_size, max_seq_len, num_feats = x.shape

    # This will hold our results
    all_pmf = torch.zeros(batch_size, max_seq_len, max_seq_len, device=device)

    with torch.no_grad():
        # Loop through "Real Time" t (0-indexed)
        # At iteration t, we feed history x[:, 0:t+1]
        for t in range(max_seq_len):

            # 1. Prepare Input: Mask future data
            # We treat everything after 't' as unknown (zeros)
            current_x = x.clone()
            if t < max_seq_len - 1:
                current_x[:, (t + 1):, :] = 0.0

            # 2. Forward Pass
            # Some models take 'lengths' arg. If yours does, pass it.
            # Here we assume the model handles zero-padding or we pass just x.
            # If your model requires lengths: out = model(current_x, lengths=torch.full((batch_size,), t+1))
            out = model(current_x)

            # Unpack: DeepHit usually returns (pmf, x_pred, x_proj) or just pmf
            if isinstance(out, tuple):
                pmf = out[0]
            else:
                pmf = out
            all_pmf[:, :t] = pmf

    mean_trajectory = compute_mean_time(all_pmf, valid_lengths, device)

    return mean_trajectory


def generate_risk_surface(model, x, max_event_horizon=None):
    """
    Generates the full dynamic prediction surface.

    Args:
        model: Trained DeepHit model.
        x: Input features [Batch, Max_Seq_Len, Feat].
        max_event_horizon: Size of the output layer (T_max).
                           If None, inferred from first pass.

    Returns:
        risk_surface: Tensor of shape [Batch, Max_Obs_Len, Max_Event_Horizon].
                      Entry [i, t_obs, t_event] is the probability P(T=t_event)
                      predicted for patient i using ONLY history x[:, :t_obs+1].
    """
    model.eval()
    batch_size, max_obs_len, _ = x.shape
    device = x.device

    # We will populate this 3D tensor
    risk_surface = None

    with torch.no_grad():
        # Loop through every possible "Observation Time"
        for t_obs in range(max_obs_len):

            # --- 1. STRICT MASKING ---
            # Create a view of x that is ZEROED out for all times > t_obs.
            # This ensures no leakage from the future.
            x_masked = x.clone()
            if t_obs < max_obs_len - 1:
                x_masked[:, (t_obs + 1):, :] = 0.0

            # --- 2. Forward Pass ---
            # Model sees x_masked, so it thinks today is t_obs.
            out = model(x_masked)

            # Unpack (Handle tuple or tensor output)
            if isinstance(out, tuple):
                pmf = out[0]  # [Batch, Max_Event_Horizon]
            else:
                pmf = out

            # Initialize surface on first pass
            if risk_surface is None:
                max_event_horizon = pmf.shape[1]
                risk_surface = torch.zeros(
                    batch_size, max_obs_len, max_event_horizon,
                    device=device, dtype=torch.float32
                )

            # --- 3. Store Prediction ---
            risk_surface[:, t_obs, :] = pmf

    return risk_surface


def hazard_to_pmf(hazards):
    """
    Converts Discrete Hazards (Sigmoid outputs) to PMF.
    Args:
        hazards: [Batch, T_obs, T_event] (values in 0-1)
    Returns:
        pmf: [Batch, T_obs, T_event] (sums to ~1 along last dim)
    """
    # 1. Compute Survival Function: S(t) = Product(1 - h_k)
    # epsilon prevents log(0) issues if you were using logs,
    # but for simple multiplication it's fine.
    survival = torch.cumprod(1 - hazards, dim=-1)

    # 2. Compute PMF: P(T=t) = h_t * S_{t-1}
    # We need to shift survival by 1 to get S_{t-1}
    # For the first step (t=0), S_{-1} is implicitly 1.0

    # Create a tensor for S_{t-1}
    # Shape: [Batch, T_obs, T_event]
    s_prev = torch.ones_like(survival)

    # Fill s_prev[:, :, 1:] with survival[:, :, :-1]
    s_prev[..., 1:] = survival[..., :-1]

    # 3. Calculate PMF
    pmf = hazards * s_prev

    return pmf


class DynamicEvaluator:
    def __init__(self, risk_surface, t_true, e_true):
        """
        Args:
            risk_surface: [Batch, T_obs, T_event] (Output from generate_risk_surface)
            t_true: [Batch] True event/censor times.
            e_true: [Batch] Event indicators.
        """
        self.risk_surface = risk_surface
        self.t_true = t_true
        self.e_true = e_true
        self.device = risk_surface.device
        self.batch_size, self.max_obs, self.max_event = risk_surface.shape

    def get_valid_patients_at_t(self, t_obs):
        """
        Returns indices of patients who are still in the 'Risk Set' at t_obs.
        (i.e., they haven't died or been censored yet).
        """
        # Strictly speaking, if t_true == t_obs, they are effectively gone/event happens now.
        # We usually predict for those surviving PAST t_obs.
        mask_still_alive = self.t_true > t_obs
        return torch.where(mask_still_alive)[0]

    def _get_conditional_expectation(self, pmf, t_obs):
        """
        Calculates expected time E[T | T > t_obs] from a PMF.
        Used for C-Index ranking.
        """
        time_grid = torch.arange(self.max_event, device=self.device).float()
        mask = time_grid >= t_obs
        pmf_future = pmf * mask
        prob_survive_max = 1.0 - pmf.sum(dim=1)
        total_future_prob = pmf_future.sum(dim=1) + prob_survive_max
        denominator = total_future_prob.clamp(min=1e-8)
        expected_val_obs = (pmf_future * time_grid).sum(dim=1)

        # Contribution from survivors (using T_max as the conservative estimate)
        expected_val_tail = prob_survive_max * self.max_event

        # 5. Final Conditional Expectation
        return (expected_val_obs + expected_val_tail) / denominator

        mean_obs = (pmf * time_grid).sum(dim=1)

        # 3. Handle Censored Cases (Survivors)
        # Probability of surviving past the last step
        # P(T >= 20) = 1 - sum(P(T=t) for t in 0..19)
        prob_censored = 1.0 - torch.sum(pmf, dim=-1)

        # Contribution = 20 * P(surviving)
        mean_censored = self.max_event * prob_censored

        # Total Expected Time
        return mean_obs + mean_censored

        # 1. Mask past probabilities (0...t_obs) -> They are impossible now
        # # shape: [Batch, T_event]
        # time_grid = torch.arange(self.max_event, device=self.device).float() + 0.5
        # mask_future = (time_grid > t_obs).float()
        #
        # masked_pmf = pmf * mask_future
        #
        # # 2. Renormalize
        # denom = masked_pmf.sum(dim=1) + 1e-8
        # cond_pmf = masked_pmf / denom.unsqueeze(1)
        #
        # # 3. Expectation
        # return (cond_pmf * time_grid).sum(dim=1)

    def landmark_c_index(self, t_obs):
        """
        Calculates C-Index specifically for predictions made at time t_obs.
        """
        # 1. Who is valid?
        idx = self.get_valid_patients_at_t(t_obs)
        if len(idx) < 2: return 0.0  # Need pairs

        # 2. Get predictions made AT t_obs for these patients
        # Shape: [N_subset, T_event]
        current_pmfs = self.risk_surface[idx, t_obs, :]

        # 3. Calculate Risk Score (Predicted Expected Survival Time)
        # Higher Time = Lower Risk
        pred_times = self._get_conditional_expectation(current_pmfs, t_obs)
        risk_scores = pred_times

        # 4. Get Truth
        true_times = self.t_true[idx]
        true_events = self.e_true[idx]

        # 5. Calculate C-Index (Reuse standard logic)
        return self._calculate_concordance(risk_scores, true_times, true_events)

    def dynamic_brier_score(self, t_obs, delta_t):
        """
        Calculates accuracy of predicting survival for (t_obs + delta_t).
        "At month 5, what is the chance of surviving to month 10?"
        """
        target_time = t_obs + delta_t
        if target_time >= self.max_event: return 0.0

        idx = self.get_valid_patients_at_t(t_obs)
        if len(idx) == 0: return 0.0

        # 1. Get Predicted Prob of Surviving past target_time
        # Use PMF from t_obs
        pmfs = self.risk_surface[idx, t_obs, :]

        # Sum PMF from (target_time + 1) to End
        surv_prob = pmfs[:, (int(target_time) + 1):].sum(dim=1)

        # 2. Ground Truth
        # Alive if t_true > target_time
        # Dead if t_true <= target_time (and is_event=1)
        # Censored in between (t_obs < t < target) -> Exclude

        subset_t = self.t_true[idx]
        subset_e = self.e_true[idx]

        is_censored_early = (subset_t <= target_time) & (subset_e == 0)
        mask_eval = ~is_censored_early

        if mask_eval.sum() == 0: return 0.0

        true_status = (subset_t > target_time).float()

        # 3. MSE
        mse = (true_status[mask_eval] - surv_prob[mask_eval]) ** 2
        return mse.mean().item()

    def dynamic_coverage_probability(self, alpha=0.95):
        """
        Iterates over ALL valid (patient, t_obs) pairs and checks if
        T_true falls inside the predicted alpha-CI.
        """
        hits = 0
        total = 0

        # Loop over every patient
        for i in range(self.batch_size):
            # Only evaluate for times BEFORE the event/censoring
            # We can't predict at t=20 if they died at t=10
            valid_steps = int(self.t_true[i].item())

            # If uncensored, we check if true time is in interval
            if self.e_true[i] == 0: continue  # Skip coverage for censored (truth unknown)

            true_time = self.t_true[i]

            for t in range(valid_steps):
                # Prediction made at time t
                pmf = self.risk_surface[i, t, :]

                # Conditional PMF (renormalized for T > t)
                time_grid = torch.arange(self.max_event, device=self.device)
                mask_future = time_grid > t

                cond_pmf = pmf * mask_future.float()
                if cond_pmf.sum() == 0: continue
                cond_pmf /= cond_pmf.sum()

                # Calculate CI (CDF based)
                cdf = torch.cumsum(cond_pmf, dim=0)
                lower_idx = (cdf > (1 - alpha) / 2).int().argmax()
                upper_idx = (cdf > 1 - (1 - alpha) / 2).int().argmax()

                if lower_idx <= true_time <= upper_idx:
                    hits += 1
                total += 1

        return hits / total if total > 0 else 0.0

    def _calculate_concordance(self, risk, t, e):
        # ... (Same C-Index logic as provided in previous answer) ...
        # Simplified for brevity:
        # broadcasting: (N, 1) < (1, N) -> (N, N) matrix
        # t = t.view(-1, 1)
        # e = e.view(-1, 1)
        # r = risk.view(-1, 1)
        # time_comparison = t < t.T

        # broadcasting: (N, 1) & (N, N) -> (N, N) matrix
        # We only care if the 'earlier' person (row i) was an actual event
        # valid_pair_mask = time_comparison & e.bool()

        c_index = concordance_index(t.cpu().numpy(), risk.cpu().numpy(), event_observed=e.bool().cpu().numpy())

        # 2. Create the Validity Mask (Who can be compared?)
        # A pair (i, j) is comparable if:
        #    Patient i had an event (e[i] == 1)
        #    AND Patient i died before Patient j (t[i] < t[j])

        # Count total valid pairs
        # total_pairs = valid_pair_mask.sum()
        #
        # if total_pairs == 0:
        #     return 0.0
        #
        # # 3. Calculate Concordance
        # # For a valid pair (i, j) where t[i] < t[j]:
        # #    We want Risk[i] > Risk[j] (The one who died first should have higher risk)
        #
        # # broadcasting: (N, 1) - (1, N) -> (N, N) matrix
        # risk_diff = r - r.T
        #
        # # Concordant: risk[i] > risk[j] -> risk_diff > 0
        # # Note: We apply the mask *after* comparison to zero out invalid pairs
        # concordant_pairs = (risk_diff > 0).float() * valid_pair_mask.float()
        #
        # # Ties: risk[i] == risk[j] -> risk_diff == 0
        # tie_pairs = (risk_diff == 0).float() * valid_pair_mask.float()
        #
        # # 4. Final Calculation
        # c_index = (concordant_pairs.sum() + 0.5 * tie_pairs.sum()) / total_pairs

        return c_index.item()


def compute_mean_time(pmf, valid_lengths, device):
    """
    Computes Expected Time to Event (Restricted Mean Survival Time)
    for every time step t1 in the sequence.

    Args:
        pmf: (Batch, T, T) - Sigmoid outputs.
                 Dim 1 is observation time (t1), Dim 2 is prediction time (t2).
        valid_lengths: (Batch,) - The actual sequence length of the patient (t_obs).
        horizon_size: int - The max time T.
        device: torch.device

    Returns:
        mean_times: (Batch, T) - Estimated mean event time for each step t1.
    """
    batch_size, T_seq, T_pred = pmf.shape

    # 1. Prepare Grids for Masking
    # We need to enforce that for a given t1, the patient is ALIVE for all t2 <= t1.
    # This means Hazard(t2) = 0 for all t2 <= t1.

    # t1 indices: (1, T_seq, 1)
    t1_grid = torch.arange(1, T_seq + 1, device=device).view(1, -1, 1)
    # t2 indices: (1, 1, T_pred)
    t2_grid = torch.arange(1, T_pred + 1, device=device).view(1, 1, -1)

    # Mask: 1 if t2 is in the future relative to t1, 0 otherwise
    future_mask = (t2_grid > t1_grid).float()

    # 2. Mask Hazards
    # If t2 <= t1, hazard is 0 (probability of death is 0, because we know they are alive)
    # If t2 > t1, we keep the model's predicted hazard
    masked_hazards = pmf * future_mask

    # 3. Compute Survival Function S(t)
    # S(t) = Product(1 - h_k) from k=1 to t
    # For t2 <= t1, term is (1-0) = 1. So S(t) stays at 1.0 (Correct).
    survival_curve = torch.cumprod(1 - masked_hazards, dim=2)

    # 4. Compute Restricted Mean Survival Time (RMST)
    # Formula: E[T] = Sum_{t=0}^{T_max} S(t)
    # Our survival_curve tensor contains S(1), S(2)... S(T).
    # We implicitly know S(0) = 1 for everyone.
    # So Mean = 1.0 + Sum(S(1)...S(T))

    mean_times = 1.0 + survival_curve.sum(dim=2)

    # 5. (Optional) Mask padding
    # If the patient's data ended at step 5, predictions for t1=6,7,8 are garbage.
    # We can zero them out or leave them (caller usually handles masking).
    # Here is how to zero them out if desired:
    seq_mask = (t1_grid.squeeze(-1) <= valid_lengths.unsqueeze(1)).float()  # (Batch, T_seq)
    mean_times = mean_times * seq_mask

    return mean_times


def single_probability_compute_and_save_metrics(preds, t_train, delta_train, x_test, t_tilde_test, delta_test,
                                                output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Compute Standard Metrics ---
    # Note: In strict survival analysis, standard metrics on delta_test can be
    # biased due to censoring. If preds are conditioned on a specific time horizon,
    # these metrics treat delta_test as the ground truth class.

    metrics = {}
    preds = preds.cpu().detach().numpy()
    delta_test = delta_test.cpu().detach().numpy()
    # ROC AUC (Discrimination)
    try:
        metrics['roc_auc'] = roc_auc_score(delta_test, preds)
    except ValueError:
        metrics['roc_auc'] = None  # Handle cases with only one class

    # Average Precision (PR AUC)
    metrics['average_precision'] = average_precision_score(delta_test, preds)

    # Brier Score (Calibration/Accuracy)
    metrics['brier_score'] = brier_score_loss(delta_test, preds)

    # Log Loss (Cross Entropy)
    # Clip preds to avoid log(0)
    eps = 1e-15
    preds_clipped = np.clip(preds, eps, 1 - eps)
    metrics['log_loss'] = log_loss(delta_test, preds_clipped)

    # --- 2. Save Metrics to JSON ---
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    return metrics


def compute_brier_score(probs, true_times, event_indicators):
    """
        A stable PyTorch implementation of the Brier Score.
        """
    # probs: (N, T_max) - PMF predictions
    # true_times: (N,)
    # event_indicators: (N,)

    # 1. Get Survival Curves S(t)
    # S(t) = 1 - CDF(t)
    max_time = probs.shape[-1]
    cdf = torch.cumsum(probs, dim=1)
    surv_pred = 1.0 - cdf
    surv_pred = torch.clamp(surv_pred, min=0.0, max=1.0)  # Safety clamp

    # 2. Create Binary Target Matrix Y(t)
    # Y(i, t) = 1 if patient i survives past time t
    # Y(i, t) = 0 if patient i experiences event at or before t

    N = len(true_times)
    time_grid = torch.arange(max_time, device=probs.device).unsqueeze(0).expand(N, -1)  # (N, T)
    true_times_exp = true_times.unsqueeze(1).expand(N, max_time)

    # Who is known to be alive at time t?
    is_alive = time_grid < true_times_exp

    # Who had an event at or before time t?
    # Note: If censored at t, we don't know status.
    # Standard BS calculation often excludes censored subjects after their censoring time.
    # We will use a mask to ignore censored data points after they vanish.

    had_event = (time_grid >= true_times_exp) & (event_indicators.unsqueeze(1).bool())

    # Target: 1.0 if Alive, 0.0 if Event
    target = is_alive.float()

    # 3. Compute Error Squared: (Target - Predicted_Survival)^2
    squared_error = (target - surv_pred) ** 2

    # 4. Masking (Handling Censoring)
    # If a patient is censored at t_c, we stop counting them after t_c
    # because we don't know if they had the event or not.

    # Mask = 1 if (time < true_time) OR (event happened)
    # Effectively: valid if we know the status.
    valid_mask = (time_grid < true_times_exp) | (event_indicators.unsqueeze(1).bool())

    # Apply mask
    masked_sq_error = squared_error * valid_mask.float()

    # Average over valid patients per time step
    brier_scores = masked_sq_error.sum(dim=0) / valid_mask.sum(dim=0).clamp(min=1.0)

    # Integrated Brier Score (Mean over time)
    ibs = brier_scores.mean().item()

    return ibs


def multi_time_probabilities_compute_and_save_metrics(preds, t_tilde_test, delta_test,
                                                      output_dir):
    estimated_mean_time = compute_mean_survival_time(preds, tail_distribution="geometric")
    # estimated_median_time = compute_quantile_survival_time(preds, quantile=0.5, tail_distribution='geometric')
    # estimated_mean_time = compute_mean_time(preds, t_tilde_test, preds.device)

    metrics = mean_time_compute_metrics(estimated_mean_time, t_tilde_test, delta_test)

    metrics['mean_time_std'] = estimated_mean_time.std().item()
    survival_loss = DiscreteSurvivalLoss(censored_mode='full_survival')
    metrics['survival_loss'] = survival_loss.forward(preds, t_tilde_test, delta_test).item()
    # A. Landmark C-Index: "How good is the model at Month 0 vs Month 12?"
    # metrics['pmf_c_index_0'] = evaluator.landmark_c_index(t_obs=0)
    # metrics['pmf_c_index_5'] = evaluator.landmark_c_index(t_obs=5)
    # metrics['pmf_c_index_10'] = evaluator.landmark_c_index(t_obs=10)
    # metrics['pmf_c_index_15'] = evaluator.landmark_c_index(t_obs=15)

    # B. Dynamic Brier: "At Month 6, how good is the 12-month survival prediction?"
    # delta_t = 6 means predicting outcome at month 12 (6+6)
    # bs_landmark = evaluator.dynamic_brier_score(t_obs=0, delta_t=10)
    # metrics['pmf_bs_t0_10'] = bs_landmark
    # bs_landmark = evaluator.dynamic_brier_score(t_obs=5, delta_t=5)
    # metrics['pmf_bs_t5_10'] = bs_landmark
    # bs_landmark = evaluator.dynamic_brier_score(t_obs=5, delta_t=10)
    # metrics['pmf_bs_t5_15'] = bs_landmark
    # bs_landmark = evaluator.dynamic_brier_score(t_obs=15, delta_t=5)
    # metrics['pmf_bs_t15_20'] = bs_landmark

    q_025 = compute_quantile_survival_time(preds, quantile=0.025, tail_distribution='geometric')
    q_05 = compute_quantile_survival_time(preds, quantile=0.05, tail_distribution='geometric')
    q_1 = compute_quantile_survival_time(preds, quantile=0.1, tail_distribution='geometric')
    q_9 = compute_quantile_survival_time(preds, quantile=0.9, tail_distribution='geometric')
    q_95 = compute_quantile_survival_time(preds, quantile=0.95, tail_distribution='geometric')
    q_975 = compute_quantile_survival_time(preds, quantile=0.975, tail_distribution='geometric')
    for t in range(0, preds.shape[1], 5):
        relevant_idx = (t_tilde_test >= t)
        uncensored_relevant = relevant_idx & delta_test.bool()
        metrics[f'cov_t{t}_95'] = ((t_tilde_test >= q_025[:, t]) & (t_tilde_test <= q_975[:, t]))[
            uncensored_relevant].float().mean().item()
        metrics[f'cov_t{t}_90'] = ((t_tilde_test >= q_05[:, t]) & (t_tilde_test <= q_95[:, t]))[
            uncensored_relevant].float().mean().item()
        metrics[f'cov_t{t}_80'] = ((t_tilde_test >= q_1[:, t]) & (t_tilde_test <= q_9[:, t]))[
            uncensored_relevant].float().mean().item()

        metrics[f'brior_score_t{t}'] = compute_brier_score(preds[relevant_idx, t], t_tilde_test[relevant_idx],
                                                           delta_test[relevant_idx])

    save_path = os.path.join(output_dir, f"metrics.json")
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)


def compute_and_save_metrics(estimated_survival_curve, estimated_mean_time, t_train, delta_train, x_test, t_tilde_test,
                             delta_test, output_dir):
    pass


def ddh_compute_and_save_metrics(model, t_train, delta_train, x_test, t_tilde_test, delta_test, output_dir):
    estimated_mean_time = get_dynamic_mean_time_trajectory(model, x_test, t_tilde_test.long())

    metrics = mean_time_compute_metrics(estimated_mean_time, t_tilde_test, delta_test)

    test_pmdf = model(x_test)[0]
    brier_score = get_brier_score(test_pmdf, t_train, delta_train, t_tilde_test, delta_test)
    metrics['brier_score'] = brier_score
    risk_surface = generate_risk_surface(model, x_test)
    evaluator = DynamicEvaluator(risk_surface, t_tilde_test, delta_test)

    # A. Landmark C-Index: "How good is the model at Month 0 vs Month 12?"
    metrics['ddh_c_index_0'] = evaluator.landmark_c_index(t_obs=0)
    metrics['ddh_c_index_5'] = evaluator.landmark_c_index(t_obs=5)
    metrics['ddh_c_index_10'] = evaluator.landmark_c_index(t_obs=10)
    metrics['ddh_c_index_15'] = evaluator.landmark_c_index(t_obs=15)

    # B. Dynamic Brier: "At Month 6, how good is the 12-month survival prediction?"
    # delta_t = 6 means predicting outcome at month 12 (6+6)
    bs_landmark = evaluator.dynamic_brier_score(t_obs=0, delta_t=10)
    metrics['dynamic_bs_t0'] = bs_landmark

    cov = evaluator.dynamic_coverage_probability(0.95)
    metrics['cov_0_95'] = cov
    cov = evaluator.dynamic_coverage_probability(0.9)
    metrics['cov_0_9'] = cov
    cov = evaluator.dynamic_coverage_probability(0.8)
    metrics['cov_0_8'] = cov

    save_path = os.path.join(output_dir, f"metrics.json")
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)


def mean_time_compute_and_save_metrics(estimated_mean_time, t_test, delta_test, output_dir):
    metrics = mean_time_compute_metrics(estimated_mean_time, t_test, delta_test)
    save_path = os.path.join(output_dir, f"metrics.json")
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)


def mean_time_compute_metrics(estimated_mean_time, t_test, delta_test):
    uncensored_idx = torch.where(delta_test == 1)[0]
    if len(uncensored_idx) == 0:
        return {}

    batch_size, T = estimated_mean_time.shape
    device = estimated_mean_time.device

    # Create time grid [1, 2, ..., T]
    time_grid = torch.arange(1, T + 1, device=device).unsqueeze(0)  # (1, T)

    # Mask: 1 if this time step matters for this patient, 0 otherwise
    mask = (time_grid <= t_test.unsqueeze(1)).float()
    mask = mask * delta_test.unsqueeze(-1).repeat(1, mask.shape[1])
    mask = mask.bool()
    t_test_repeated = t_test.unsqueeze(1).repeat(1, estimated_mean_time.shape[1])
    all_errors = abs(estimated_mean_time - t_test_repeated)[mask]

    calibration_mean_error = abs(estimated_mean_time[mask].mean(dim=0) - \
                                 t_test_repeated[mask].float().mean(dim=0)) \
        .mean()
    calibration_median_error = abs(estimated_mean_time[mask].median(dim=0).values - \
                                   t_test_repeated[mask].float().median(
                                       dim=0).values) \
        .mean()
    medae_uncensored = torch.median(all_errors).item()
    mse = (all_errors ** 2).mean()
    mae = all_errors.mean()
    pred_static_t0 = estimated_mean_time[uncensored_idx, 0]
    pred_static_t10 = estimated_mean_time[uncensored_idx, 10]
    true_times = t_test[uncensored_idx]
    mape_uncensored_t0 = ((pred_static_t0 - true_times).abs() / (true_times + 1e-6)).mean().item()  # Percentage error
    # bias_uncensored_t0 = (pred_static_t0 - true_times).mean().item()  # Positive = Overestimation

    mape_uncensored_t10 = ((pred_static_t10 - true_times).abs() / (true_times + 1e-6)).mean().item()  # Percentage error
    # bias_uncensored_t10 = (pred_static_t10 - true_times).mean().item()  # Positive = Overestimation

    previous_event_time_mse = ((estimated_mean_time[uncensored_idx, t_test.long()[uncensored_idx] - 1] - t_test[
        uncensored_idx]) ** 2).mean()
    previous_event_time_mae = (
        (estimated_mean_time[uncensored_idx, t_test.long()[uncensored_idx] - 1] - t_test[uncensored_idx])).abs().mean()

    metrics = {
        "medae_uncensored": float(medae_uncensored),
        "mape_uncensored_t0": float(mape_uncensored_t0),
        # "bias_uncensored_t0": float(bias_uncensored_t0),
        "mape_uncensored_t10": float(mape_uncensored_t10),
        # "bias_uncensored_t10": float(bias_uncensored_t10),
        "mae_uncensored": float(mae),
        "mse_uncensored": float(mse),
        "previous_event_mse": float(previous_event_time_mse),
        "previous_event_mae": float(previous_event_time_mae),
        "n_test_samples": len(estimated_mean_time),
        "n_uncensored": len(uncensored_idx),
        "ratio_uncensored": len(uncensored_idx) / len(estimated_mean_time),
        "calibration_mean_error": float(calibration_mean_error),
        "calibration_median_error": float(calibration_median_error)
    }

    for t in range(0, estimated_mean_time.shape[1], 5):
        mse_at_time_t = ((estimated_mean_time[:, t] - t_test)[(t_test >= t) & delta_test.bool()] ** 2).mean().item()
        mae_at_time_t = ((estimated_mean_time[:, t] - t_test)[(t_test >= t) & delta_test.bool()]).abs().mean().item()
        bias_at_time_t = abs((estimated_mean_time[:, t][(t_test >= t) & delta_test.bool()].mean().item() - \
                              t_test[(t_test >= t) & delta_test.bool()].float().mean().item()))
        metrics[f'mse_at_time_{t}'] = float(mse_at_time_t)
        metrics[f'mae_at_time_{t}'] = float(mae_at_time_t)
        metrics[f'bias_at_time_{t}'] = float(bias_at_time_t)

    for t in range(0, estimated_mean_time.shape[1], 5):
        mse_event_time_t = ((estimated_mean_time - t_test_repeated)[mask & (t_test_repeated == t)] ** 2).mean().item()
        mae_event_time_t = ((estimated_mean_time - t_test_repeated)[mask & (t_test_repeated == t)]).abs().mean().item()
        metrics[f'mse_event_time_{t}'] = float(mse_event_time_t)
        metrics[f'mae_event_time_{t}'] = float(mae_event_time_t)

    for t in range(0, T, 5):

        # Who is still in the study at time t? (t_test > t)
        at_risk_mask = t_test > t

        if at_risk_mask.float().sum() > 10:  # Need enough samples to calc C-Index
            # Predictions made at time t for people alive at t
            # Note: We negate prediction because Higher Survival Time = Lower Risk
            # C-index usually expects "Risk Score". If passing "Survival Time", logic might flip depending on library.
            # Lifelines: (Time, Score, Event). If Score is Survival Time, higher is better.

            try:
                c_idx = concordance_index(
                    t_test[at_risk_mask].cpu().detach().numpy(),
                    estimated_mean_time[at_risk_mask, t].detach().cpu().numpy(),
                    delta_test.bool()[at_risk_mask].detach().cpu().numpy()
                )
                metrics[f'c_index_at_time_{t}'] = c_idx.item()
            except Exception as e:
                print(f"Skipping C-index at {t}: {e}")

    # Does the model successfully separate Short vs Long conversations?
    try:
        # Split data by the Median of the PREDICTIONS
        median_pred = estimated_mean_time[:, 0].median().item()
        high_risk_group = estimated_mean_time[:, 0] < median_pred  # Predicted to fail early
        low_risk_group = ~high_risk_group  # Predicted to survive long

        # Run Log-Rank test between these two groups based on ACTUAL outcomes
        if high_risk_group.float().sum() > 0 and low_risk_group.float().sum() > 0:
            results = logrank_test(
                t_test[high_risk_group].detach().cpu().numpy(), t_test[low_risk_group].detach().cpu().numpy(),
                event_observed_A=delta_test.bool()[high_risk_group].detach().cpu().numpy(),
                event_observed_B=delta_test.bool()[low_risk_group].detach().cpu().numpy()
            )
            # The test statistic (chi-squared). Higher = Better Separation.
            metrics["log_rank_statistic"] = float(results.test_statistic)
            metrics["log_rank_p_value"] = float(results.p_value)
    except Exception as e:
        print(f"Log-rank failed: {e}")

    return metrics


def hazard_probabilities_compute_and_save_metrics(p, x_test, t_test, delta_test, output_dir):
    """
    Computes standard survival metrics and saves them to a JSON file.

    Args:
        p: hazar probabilities.
        x_test: (N, T, F) input data.
        t_test: (N,) actual time of event/censoring.
        delta_test: (N,) event indicator (1=event, 0=censored).
        output_dir: Folder to save results.
        run_name: Filename identifier.
    """
    os.makedirs(output_dir, exist_ok=True)
    t_test, delta_test = t_test.long().cpu().detach().numpy(), delta_test.bool().cpu().detach().numpy()
    # 1. Generate Predictions
    with torch.no_grad():
        # Hazard probability p(t)
        # Survival S(t)
        S = torch.cumprod(1 - p, dim=1).cpu().detach().numpy()
        # CDF F(t)
        CDF = 1 - S

    # Max time horizon in test set
    T_max = S.shape[1]

    # --- Metric 1: Concordance Index (Discrimination) ---
    # We use (1 - S at event time) as the "risk score".
    # Note: For time-dependent C-index, we usually pick a fixed evaluation time
    # or integrate. Here we use the risk at the actual observed time t_i
    # (or the final time if t_i is out of bounds).

    # Simple proxy for risk: "Probability of dying before the end" (1 - S_final)
    # Or better: "Expected Survival Time" (area under S(t))
    expected_lifetime = np.sum(S, axis=1)
    # C-index: Higher expected lifetime should correspond to higher true time t
    c_index = concordance_index(t_test, expected_lifetime, delta_test)

    # --- Metric 2: Integrated Brier Score (IBS) (Calibration + Accuracy) ---
    # We calculate Brier score at every single time step and average it.
    brier_scores = []
    times_to_eval = np.arange(1, T_max)

    for t in times_to_eval:
        # Predicted probability of surviving past t: S(t)
        # In python index t corresponds to index t-1 (0-based) ??
        # Actually S array usually aligns 0..T-1.
        # Let's assume S[:, t-1] is prediction for time step t.
        if t - 1 >= S.shape[1]: break

        pred_surv_t = S[:, t - 1]

        # True status at time t:
        # Alive if (time_of_event > t)
        # Dead if (time_of_event <= t) AND (uncensored)
        # Censored before t? We exclude them (standard simple Brier) or weighting.
        # For simplicity on 100 samples: Exclude patients censored before t.

        # Mask: People who are either (Events observed) OR (Censored AFTER t)
        # i.e. we know their status at time t for sure.
        known_status_mask = (t_test > t) | ((t_test <= t) & (delta_test == 1))

        if known_status_mask.astype(float).sum().item() == 0:
            continue

        # Target: 1 if alive at t, 0 if dead at t
        true_status = (t_test[known_status_mask] > t).astype(float)
        preds = pred_surv_t[known_status_mask]

        bs = np.mean((true_status - preds) ** 2)
        brier_scores.append(bs)

    ibs = np.mean(brier_scores) if brier_scores else 0.0

    # --- Metric 3: MAE on Uncensored (Real-world Error) ---
    uncensored_idx = np.where(delta_test == 1)[0]
    if len(uncensored_idx) > 0:
        # Median predicted time: First time S(t) <= 0.5
        pred_medians = (S <= 0.5).argmax(axis=1)
        # Fix those that never drop (set to T_max)
        unfinished = (S[:, -1] > 0.5)
        pred_medians[unfinished] = T_max

        calibration_error = abs(pred_medians[uncensored_idx].mean() - t_test[uncensored_idx].mean())

        mae = np.mean(np.abs(pred_medians[uncensored_idx] - t_test[uncensored_idx]))
    else:
        mae = -1.0
        calibration_error = -1.

    # --- Metric 4: D-Calibration (KS Test) ---
    # Are the PIT values uniform?
    # Get CDF values at event times for uncensored
    event_indices = np.clip(t_test[uncensored_idx] - 1, 0, T_max - 1)
    pit_values = CDF[uncensored_idx, event_indices]

    # Kolmogorov-Smirnov test against Uniform Distribution
    # Null hypothesis: data is drawn from Uniform(0,1).
    # High p-value (>0.05) means "We cannot say it's NOT uniform" (Good).
    # Low p-value (<0.05) means "It is definitely not uniform" (Bad).
    ks_stat, ks_p_value = kstest(pit_values, 'uniform')

    # --- Pack and Save ---
    metrics = {
        "c_index": float(c_index),
        "ibs": float(ibs),
        "mae_uncensored": float(mae),
        "d_calib_ks_stat": float(ks_stat),
        "d_calib_p_value": float(ks_p_value),
        "n_test_samples": len(x_test),
        "n_uncensored": len(uncensored_idx),
        "calibration_error": calibration_error
    }

    save_path = os.path.join(output_dir, f"metrics.json")
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Results saved to: {save_path}")
    print("--- Summary ---")
    print(f"C-Index: {c_index:.3f} (Rank accuracy)")
    print(f"IBS:     {ibs:.3f} (Probabilistic error)")
    print(f"MAE:     {mae:.1f} steps (Point error)")
    print(f"D-Calib: p={ks_p_value:.3f} (Is it uniform? >0.05 is good)")

    return metrics




def get_tmp_calibration_result_path(experiments_name, calibration_name):
    return os.path.join("results", "tmp_calibration_results", experiments_name, calibration_name)

def get_tmp_upb_calibration_result_path(experiments_name, calibration_name):
    return os.path.join("results", "tmp_upb_calibration_results", experiments_name, calibration_name)

def get_tmp_metric_calibration_result_path(experiments_name, calibration_name):
    return os.path.join("results", "tmp_metric_calibration_results", experiments_name, calibration_name)


def get_merged_calibration_result_path(experiments_name, basedir='.'):
    return os.path.join(basedir, "results", "merged_calibration_dfs", experiments_name)

def get_merged_upb_calibration_result_path(experiments_name, basedir='.'):
    return os.path.join(basedir, "results", "merged_upb_calibration_dfs", experiments_name)

def get_merged_metric_calibration_result_path(experiments_name, basedir='.'):
    return os.path.join(basedir, "results", "merged_metric_calibration_dfs", experiments_name)

import os
from src.dataset_utils.data_utils import get_data

def setup_experiment_data(cal_size, is_real, device, dataset_name, data_setup, taus_range, m_upper_bound):
    model_preds_dir = f'alg_playground_model/is_real_{is_real}_dataset_{dataset_name}_dataset_{data_setup}'
    model_cal_test_preds_path = os.path.join(model_preds_dir, "probability_est_cal_test.pt")
    load_x = not os.path.exists(model_cal_test_preds_path)

    p_train, p_cal, p_test, x_train, x_cal, x_test, y_train, y_cal, y_test, t_tilde_train, t_tilde_cal, t_tilde_test, \
        e_train, e_cal, e_test, b_train, b_cal, b_test, n_samples_train, n_samples_cal, n_samples_test = get_data(
        is_real, device, dataset_name, data_setup, load_x=load_x)
    print("loaded data")

    model_save_dir = f'alg_playground_model/is_real_{is_real}_dataset_{dataset_name}_dataset_{data_setup}'
    model_save_path = os.path.join(model_save_dir, 'model.pt')
    model_figure_save_path = os.path.join(model_save_dir, 'history.png')
    os.makedirs(model_save_dir, exist_ok=True)

    max_time = y_train.shape[1]

    t_tilde_cal_test = torch.cat([t_tilde_cal, t_tilde_test]).clone()
    quantile_est_cal_test, probability_est, conditional_grid = compute_probabilities_and_quantiles(
        x_cal, x_train, x_test, model_cal_test_preds_path, dataset_name, data_setup, p_cal, p_test, max_time,
        model_save_path, model_figure_save_path, is_real, t_tilde_train, taus_range, m_upper_bound, device
    )

    test_size = len(quantile_est_cal_test) - cal_size
    return max_time, t_tilde_cal_test, quantile_est_cal_test, probability_est, conditional_grid, test_size
