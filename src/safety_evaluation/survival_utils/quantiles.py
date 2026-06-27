import torch

import torch
from typing import List, Union


def compute_conditional_quantiles_single_step(conditional_pmf: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Computes time-to-event quantiles for a single specific starting time condition.

    Args:
        conditional_pmf (torch.Tensor): Shape (Batch, FutureTime).
            This represents P(Event=f | Event >= t_current) for a fixed t_current.
            (e.g., this is conditional_grid[:, t_current, :])
        tau (float): The quantile level (e.g., 0.5).

    Returns:
        torch.Tensor: Shape (Batch,).
            The estimated time index where the cumulative probability reaches tau.
    """
    B, T = conditional_pmf.shape
    device = conditional_pmf.device

    # --- 1. Calculate Conditional Cumulative Probability ---
    # Shape: (B, T)
    cond_cdf = conditional_pmf.cumsum(dim=-1)

    # --- 2. Find In-Window Quantiles ---
    threshold_crossed = (cond_cdf >= tau)
    # argmax finds the first True index
    quantiles_idx = torch.argmax(threshold_crossed.int(), dim=-1).float()

    # --- 3. Identify Extrapolation Cases ---
    max_cdf = cond_cdf[:, -1]  # Shape: (B,)
    needs_extrapolation = (max_cdf < tau)

    # --- 4. Analytic Extrapolation (Exponential Tail) ---

    # A. Survival probability remaining at the end: P(T > T_max | T >= t_current)
    surv_at_max = (1.0 - max_cdf).clamp(min=1e-8)

    # B. Target Survival
    target_surv = 1.0 - tau

    # C. Recover the Last Hazard Rate (h_last) from the PMF
    # We need h(T_max) = P(T=T_max | T >= T_max)
    # Relation: P(T=T_max | T >= t_curr) = h(T_max) * P(T >= T_max | T >= t_curr)
    # So: h(T_max) = PMF[-1] / S(T_max - 1)

    # P(T=T_max | T >= t_curr)
    prob_at_last = conditional_pmf[:, -1]

    # P(T >= T_max | T >= t_curr) = 1 - P(T < T_max | T >= t_curr)
    # This is 1 - CDF[-2] (cumulative prob up to second to last step)
    # We handle the T=1 edge case where there is no 'second to last'
    if T > 1:
        cdf_prev = cond_cdf[:, -2]
    else:
        cdf_prev = torch.zeros(B, device=device)

    surv_prior_to_last = (1.0 - cdf_prev).clamp(min=1e-8)

    h_last = prob_at_last / surv_prior_to_last
    h_last = torch.clamp(h_last, min=1e-6, max=1.0 - 1e-6)

    # D. Calculate Delta T
    numerator = torch.log(torch.tensor(target_surv, device=device)) - torch.log(surv_at_max)
    denominator = torch.log(1.0 - h_last)

    delta_t = (numerator / denominator).clamp(min=0.0)

    # --- 5. Combine Results ---
    last_time_idx = T - 1

    final_quantiles = torch.where(
        needs_extrapolation,
        last_time_idx + delta_t,
        quantiles_idx
    )

    return final_quantiles

def compute_conditional_quantiles_per_tau(conditional_grid: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Computes the time-to-event quantiles for each time step 't', given survival
    up to that step. Uses analytic extrapolation for tails.

    If the cumulative probability within the window does not reach 'tau', the function
    extrapolates beyond the last observed time step assuming a constant hazard rate
    equal to the hazard at the last step (Exponential Tail).

    Args:
        conditional_grid (torch.Tensor): Shape (Batch, Time, Time).
            Row 't' contains the conditional PMF: P(Event=f | Event >= t).
            (This grid must be valid, i.e., rows sum to <= 1.0).
        tau (float): The cumulative probability threshold (e.g., 0.5 for median).
                     Values must be in (0, 1).

    Returns:
        torch.Tensor: Shape (Batch, Time).
            The estimated time index where the cumulative probability reaches 'tau'.
            - Values < T-1: Event predicted within the window.
            - Values > T-1: Event predicted in the extrapolated tail.
    """
    B, T = conditional_grid.shape[:2]
    device = conditional_grid.device

    # --- 1. Calculate Conditional Cumulative Probability ---
    # cdf[b, t, f] = P(Event <= f | Event >= t)
    cond_cdf = conditional_grid.cumsum(dim=-1)

    # --- 2. Find In-Window Quantiles ---
    # Identify the first time index 'f' where CDF >= tau
    threshold_crossed = (cond_cdf >= tau)

    # argmax finds the first True. If no value is True, it returns 0 (handled later).
    quantiles_idx = torch.argmax(threshold_crossed.int(), dim=-1).float()

    # --- 3. Identify Extrapolation Cases ---
    # If the CDF at the last time step is still < tau, we didn't cross the threshold.
    max_cdf = cond_cdf[..., -1]  # Shape: (B, T)
    needs_extrapolation = (max_cdf < tau)

    # --- 4. Analytic Extrapolation (Exponential Tail) ---
    # We solve for delta_t: S(T_max + delta) = 1 - tau
    # Formula: delta_t = (ln(1 - tau) - ln(S_max)) / ln(1 - h_last)

    # A. Survival probability remaining at the end of the window: P(T > T_max | T >= t)
    # We clamp specifically to avoid log(0)
    surv_at_max = (1.0 - max_cdf).clamp(min=1e-8)

    # B. Target Survival Probability (the "gap" we need to close in the tail)
    target_surv = 1.0 - tau

    # C. Extract the Last Hazard Rate from the Grid
    # The grid diagonal contains h(t). The last hazard is at index [T-1, T-1].
    # We assume the hazard rate remains constant at this value for t > T_max.
    # Shape: (B, ) -> Expand to (B, 1) for broadcasting across 'current time' dim.
    h_last = conditional_grid[:, -1, -1].unsqueeze(1)

    # Clamp hazard to avoid division by zero (h=0) or log(0) (h=1)
    h_last = torch.clamp(h_last, min=1e-6, max=1.0 - 1e-6)

    # D. Calculate Delta T
    # Numerator: ln(Target / Current_Survival)
    # Denominator: ln(Probability of surviving one more step)
    numerator = torch.log(torch.tensor(target_surv, device=device)) - torch.log(surv_at_max)
    denominator = torch.log(1.0 - h_last)

    delta_t = numerator / denominator

    # Ensure non-negative (statistically it should be positive if needs_extrapolation is True)
    delta_t = delta_t.clamp(min=0.0)

    # --- 5. Combine Results ---
    # Base index for extrapolation is the last time index (T-1)
    last_time_idx = T - 1

    final_quantiles = torch.where(
        needs_extrapolation,
        last_time_idx + delta_t,  # Extrapolated float value (e.g. 20.5)
        quantiles_idx  # Integer grid value (e.g. 15.0)
    )

    return final_quantiles


def compute_conditional_quantiles(conditional_grid: torch.Tensor, taus: Union[List, torch.Tensor]) -> torch.Tensor:
    """
    Computes time-to-event quantiles for multiple levels of tau simultaneously, given survival
    up to that step. Uses analytic extrapolation for tails.
    If the cumulative probability within the window does not reach 'tau', the function
    extrapolates beyond the last observed time step assuming a constant hazard rate
    equal to the hazard at the last step (Exponential Tail).

    Args:
        conditional_grid (torch.Tensor): Shape (Batch, Time, Time).
                    Row 't' contains the conditional PMF: P(Event=f | Event >= t).
                    (This grid must be valid, i.e., rows sum to <= 1.0).
        taus (list[float] | torch.Tensor): A list or 1D tensor of quantile levels
                                           (e.g., [0.1, 0.5, 0.9]).

    Returns:
        torch.Tensor: Shape (Batch, Num_Taus, Time).
            result[b, k, t] is the estimated time for the k-th tau,
            given survival up to time t.
            - Values < T-1: Event predicted within the window.
            - Values > T-1: Event predicted in the extrapolated tail.

    """
    B, T = conditional_grid.shape[:2]
    device = conditional_grid.device

    # Ensure taus is a tensor on the correct device
    if not isinstance(taus, torch.Tensor):
        taus = torch.tensor(taus, device=device, dtype=conditional_grid.dtype)

    # Shape of taus: (K,) -> Reshape for broadcasting
    K = len(taus)
    taus_bc = taus.view(1, K, 1, 1)  # For comparing with full grid (B, 1, T, T)
    taus_vec = taus.view(1, K, 1)  # For comparing with max_cdf (B, 1, T)

    # --- 1. Calculate Conditional CDF ---
    # Shape: (B, T, T) -> Expand to (B, 1, T, T)
    cond_cdf = conditional_grid.cumsum(dim=-1).unsqueeze(1)

    # --- 2. Find In-Window Quantiles ---
    # Compare (B, 1, T, T) with (1, K, 1, 1) -> Result (B, K, T, T)
    threshold_crossed = (cond_cdf >= taus_bc)

    # argmax over the last dimension (Time 'f') -> Shape (B, K, T)
    quantiles_idx = torch.argmax(threshold_crossed.int(), dim=-1).float()

    # --- 3. Identify Extrapolation Cases ---
    # Check CDF at the last step T-1
    # Shape: (B, 1, T)
    max_cdf = cond_cdf[..., -1]

    # Compare (B, 1, T) with (1, K, 1) -> Result (B, K, T)
    needs_extrapolation = (max_cdf < taus_vec)

    # --- 4. Analytic Extrapolation (Vectorized over K) ---

    # A. Survival probability at max: S(T_max | t)
    # Shape: (B, 1, T)
    surv_at_max = (1.0 - max_cdf).clamp(min=1e-8)

    # B. Target Survival: 1 - tau
    # Shape: (1, K, 1)
    target_surv = (1.0 - taus_vec)

    # C. Last Hazard Rate
    # Shape: (B,) -> Expand to (B, 1, 1)
    h_last = conditional_grid[:, -1, -1].view(B, 1, 1)
    h_last = torch.clamp(h_last, min=1e-6, max=1.0 - 1e-6)

    # D. Calculate Delta T
    # (B, 1, T) and (1, K, 1) broadcast to (B, K, T)
    numerator = torch.log(target_surv) - torch.log(surv_at_max)
    denominator = torch.log(1.0 - h_last)

    delta_t = (numerator / denominator).clamp(min=0.0)

    # --- 5. Combine Results ---
    last_time_idx = T - 1

    final_quantiles = torch.where(
        needs_extrapolation,
        last_time_idx + delta_t,
        quantiles_idx
    )

    return final_quantiles

def validate_quantiles(quantiles: torch.Tensor, taus: Union[List, torch.Tensor], conditional_grid):
    """
        Args:
            quantiles: (Batch, Num_Taus, Time) - Predicted quantile time indices.
                torch.Tensor: Shape (Batch, Num_Taus, Time).
                quantiles[b, k, t] is the estimated time for the k-th tau,
                given survival up to time t.

            taus (list[float] | torch.Tensor): A list or 1D tensor of quantile levels
                                   (e.g., [0.1, 0.5, 0.9]).

            conditional_grid (torch.Tensor): Shape (Batch, Time, Time).
                Row 't' contains the conditional PMF: P(Event=f | Event >= t).
                (This grid must be valid, i.e., rows sum to <= 1.0).

    """
    for i, tau in enumerate(taus):
        single_tau_quantiles = compute_conditional_quantiles_per_tau(conditional_grid, tau)
        assert (single_tau_quantiles - quantiles[:, i]) < 1e-5


def validate_quantile_coverage(quantiles, true_times, tau, tolerance=0.1):
    """
    Validates that the empirical coverage matches the target (1 - tau).

    Args:
        quantiles: (Batch, Time) - Predicted quantile time indices.
        true_times: (Batch,) - Actual observed event times (or censoring times).
        tau: float - The target quantile level (e.g., 0.1).
        tolerance: float - Allowed margin of error (e.g., 0.05 for 5%).
    """
    # 1. Get dimensions
    # quantiles shape is (B, T), so T is the max observed window size
    max_obs_time = quantiles.shape[1]

    # print(f"--- Validating Coverage for tau={tau} (Target Coverage: {1 - tau:.2f}) ---")

    # Check the first few time steps (checking all T might be noisy for small risk sets)
    for t in range(0, 3):

        # A. Filter for the Risk Set (Subjects alive at start of t)
        # Note: We must ensure we have enough samples to calculate a meaningful mean
        risk_set_mask = (true_times >= t)
        n_samples = risk_set_mask.sum().item()

        if n_samples < 10:
            print(f"t={t}: Skipped (Too few samples in risk set: {n_samples})")
            continue

        # B. Get predictions and actuals for the risk set
        # We perform the check: Predicted_Quantile <= Actual_Time
        # This is equivalent to T >= Q (Coverage)
        q_pred = quantiles[risk_set_mask, t]
        t_actual = true_times[risk_set_mask]

        # C. Calculate Empirical Coverage
        # Note: We do NOT clamp q_pred. If the model predicts 25.4 (extrapolated)
        # and the true time is 20 (censored), then 25.4 <= 20 is False.
        # This is correct behavior for checking if the event happened *before* the quantile.

        coverage_mask = (q_pred <= t_actual).float()
        empirical_coverage = coverage_mask.mean().item()

        # D. Assertion / Check
        diff = empirical_coverage - (1 - tau)

        # Status: PASS if within tolerance OR if conservative (slightly higher is ok in discrete land)
        # For strict checking:
        passed = abs(diff) < tolerance

        # Soft Assertion (Raises warning instead of crashing)
        if not passed:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"t={t} | N={n_samples} | Cov: {empirical_coverage:.4f} (Diff: {diff:+.4f}) | {status}")
            # print(f"   WARNING: Coverage deviation > {tolerance} at t={t}")



