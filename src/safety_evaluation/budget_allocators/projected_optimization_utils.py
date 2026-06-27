import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from torch import nn
from scipy.optimize import minimize


def sample_c(C_probs, prior_quantile_est):
    C = torch.where(torch.rand(len(prior_quantile_est), device=C_probs.device) < C_probs, prior_quantile_est,
                    0).int().reshape(
        -1, 1)
    return C


class BudgetProjectedLayer(nn.Module):
    def __init__(self, T, target_budget):
        super().__init__()
        self.T = T
        self.target_budget = target_budget

    def forward(self, Y_raw, mask):
        """
        Finds a scalar delta such that if we shift Y_raw everywhere by delta,
        the UNWEIGHTED cumulative budget is satisfied exactly.
        """
        device = Y_raw.device

        # 1. Calculate Raw Cumulative Probabilities
        # Mask Y_raw so invalid steps don't mess up the cumsum
        Y_masked = Y_raw * mask.float()

        # P_cum_raw[i, t] = exp( y_0 + ... + y_t )
        log_P_cum_raw = torch.cumsum(Y_masked, dim=1)
        P_cum_raw = torch.exp(log_P_cum_raw) * mask.float()

        # 2. Formulate the Root Finding Problem
        # We want to find a shift 'delta' such that Y_new = Y_raw + delta
        # P_cum_new(t) = exp( Sum(y_k + delta) )
        #              = exp( Sum(y_k) + (t+1)*delta )
        #              = P_cum_raw(t) * (e^delta)^(t+1)

        # Let x = e^delta. We want to find x.
        # Target = Sum_{i,t} [ P_cum_raw(i,t) * x^(t+1) ]  <-- Unweighted sum

        # Pre-calculate coefficients C_t = Sum_over_i( P_cum_raw(i, t) )
        C = torch.sum(P_cum_raw, dim=0)  # Shape (T,)

        # 3. Newton's Method to find x
        # f(x) = Sum( C_t * x^(t+1) ) - Budget = 0
        x = torch.tensor(1.0, device=device)  # Start guess (delta=0)
        powers = torch.arange(1, self.T + 1, device=device, dtype=torch.float32)

        for _ in range(12):  # 10 iterations is plenty for high precision
            # f(x)
            x_pow = torch.pow(x, powers)
            f_val = torch.dot(C, x_pow) - self.target_budget

            # f'(x) = Sum( C_t * (t+1) * x^t )
            # Efficient calc: Sum( term * (t+1) ) / x
            deriv_terms = C * powers * x_pow
            f_deriv = torch.sum(deriv_terms) / (x + 1e-8)

            # Update
            diff = f_val / (f_deriv + 1e-8)
            x = x - diff
            x = torch.clamp(x, 1e-6, 5.0)  # Clamp for safety

            if torch.abs(f_val) < 1e-3 * self.target_budget:
                break

        # 4. Apply Shift
        # shift = log(x) -> Y_new = Y_raw + log(x)
        shift = torch.log(x)
        Y_projected = Y_raw + shift
        Y_projected.clamp_(max=0.)
        return Y_projected


def solve_pytorch_stable(S, lengths, target_budget_per_sample=16, lr=0.001, max_iter=200):
    # --- 1. SETUP ---
    if not isinstance(S, torch.Tensor):
        S = torch.tensor(S, dtype=torch.float32)
    if not isinstance(lengths, torch.Tensor):
        lengths = torch.tensor(lengths, dtype=torch.float32)

    device = S.device
    N, T = S.shape

    # Masks and Weights
    range_t = torch.arange(T, device=device).unsqueeze(0)
    mask = range_t <= lengths.unsqueeze(1)

    # Weights for budget: [1, 2, 3, ..., T]
    weights = torch.arange(1, T + 1, device=device, dtype=torch.float32)
    TOTAL_BUDGET = N * target_budget_per_sample

    # --- 2. INITIALIZATION ---
    # We optimize Y = log(P).
    # Initialize close to 0 (P=1) so we start with a "cheap" objective
    # but put a small penalty at t=0 to give it room to move.
    Y = torch.zeros((N, T), device=device)
    Y[:, 0] = torch.log(1 / (((1 / target_budget_per_sample) * lengths.sqrt().mean()) * lengths.sqrt()))
    # Y[:, 0] = np.log(0.1)  # Start with P(t=0) approx 0.13
    Y[:, 1:] = np.log(1)  # Start with P(t=0) approx 0.13
    Y.clamp_(max=0.0)
    curr_budget = torch.sum(torch.exp(torch.cumsum(Y * mask.float(), dim=1)) * mask.float()).item()
    print(f"initial loss: {(1 / Y[:, 0].exp()).mean().item():2f} | initial budget: {curr_budget:2f}/{TOTAL_BUDGET:1f}")
    Y = torch.nn.Parameter(Y)

    # Use Adam. It handles the different scales of gradients better than SGD.
    optimizer = torch.optim.Adam([Y], lr=lr)

    iso_reg = IsotonicRegression(increasing=True, out_of_bounds='clip')

    # Pre-compute numpy indices for Isotonic Projection
    S_np = S.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()
    sort_indices = []

    for t in range(T):
        valid_idx = np.where(mask_np[:, t])[0]
        if len(valid_idx) > 0:
            s_col = S_np[valid_idx, t]
            idx = np.argsort(s_col)
            # Store (rows, sorted_rows)
            sort_indices.append((valid_idx, valid_idx[idx]))
        else:
            sort_indices.append(None)

    print(f"Starting Stable PyTorch Optimization...")
    projector = BudgetProjectedLayer(T, TOTAL_BUDGET)

    # --- 3. OPTIMIZATION LOOP ---
    for it in range(max_iter):
        optimizer.zero_grad()

        # A. Compute Log-Probability of the Sequence (R_i)
        # We assume independent probabilities for the sequence product:
        # log(Prod P) = Sum(log P)
        Y_valid = projector(Y, mask)
        # Mask Y so invalid steps are 0 (log(1)=0) in the sum
        Y_masked = Y_valid * mask.float()
        R = torch.sum(Y_masked, dim=1)  # Shape (N,)

        # B. STABLE OBJECTIVE: LogSumExp
        # We want to minimize Sum( exp(-R) )
        # Equivalent to minimizing Log( Sum( exp(-R) ) )
        # This is exactly torch.logsumexp(-R)
        objective_loss = torch.exp(-R).mean()
        # curr_budget = torch.sum(torch.exp(torch.cumsum(Y * mask.float(), dim=1)) * mask.float())
        # budget_loss = torch.relu(curr_budget - TOTAL_BUDGET)
        total_loss = objective_loss  # + 0.02*budget_loss

        # Backward
        total_loss.backward()
        optimizer.step()

        # --- D. Projections (Hard Constraints) ---
        with torch.no_grad():

            # 1. Monotonicity (Isotonic Regression)
            # Apply periodically (e.g., every 5 steps) to save CPU transfer time
            if it % 10 == 0:
                Y_data = Y.detach().cpu().numpy()
                did_change = False
                for t in range(T):
                    if sort_indices[t] is not None:
                        rows, sorted_rows = sort_indices[t]
                        y_col_sorted = Y_data[sorted_rows, t]
                        tensor_y_col_sorted = Y[sorted_rows, t]
                        if torch.any(tensor_y_col_sorted[:-1] > tensor_y_col_sorted[1:]):
                            # Violation found, fix it
                            y_new = iso_reg.fit_transform(np.arange(len(y_col_sorted)), y_col_sorted)
                            # Map back
                            Y_data[sorted_rows, t] = y_new
                            did_change = True

                if did_change:
                    Y.copy_(torch.from_numpy(Y_data).to(device))

            # 2. Upper Bound (Prob <= 1)
            Y.clamp_(max=0.0)

        # --- E. Logging ---
        if it % 200 == 0:
            # Calculate the REAL objective (Sum 1/P) for display
            # (Use standard sum for display, it might be large but useful to see)
            real_obj = torch.mean(torch.exp(-R)).item()
            # P_cum = torch.exp(torch.cumsum(Y_masked, dim=1)) * mask.float()
            used_budget = torch.sum(torch.exp(torch.cumsum(Y_masked, dim=1)) * mask.float()).item()
            print(
                f"Iter {it}: Log-Obj={total_loss.item():.4f} (Real={real_obj:.2e}) | Budget={used_budget:.1f}/{TOTAL_BUDGET}")

    # Final Return
    P_final = torch.exp(Y_valid).detach().cpu().numpy()
    P_final = P_final * mask_np
    print(P_final)
    return P_final


def project_to_test_ir(optimal_P, val_scores, test_scores, val_prior_q, t_val, T_max_curr, device):
    calibration_curves = []
    range_t = torch.arange(T_max_curr, device=device).unsqueeze(0)
    max_val = torch.minimum(val_prior_q, t_val)
    mask_val = range_t <= max_val.unsqueeze(1)
    for t in range(T_max_curr):
        # Initialize a regressor for this specific time step
        # We enforce that the curve must be increasing (y_min <= y_max)
        iso_reg = IsotonicRegression(increasing=True, out_of_bounds='clip')
        # Fit the curve: Learn mapping from Score -> Optimal Probability
        if len(val_scores[mask_val[:, t], t]) == 0:
            iso_reg.fit(val_scores[:, t].detach().cpu().numpy(), np.ones_like(val_scores[:, t].detach().cpu().numpy()))
        else:
            iso_reg.fit(val_scores[mask_val[:, t], t].detach().cpu().numpy(),
                        optimal_P[mask_val[:, t].detach().cpu().numpy(), t])

        calibration_curves.append(iso_reg)

    # --- INFERENCE PHASE ---
    # For a new test sample with scores 's_test' (size 200)
    p_test = torch.zeros(len(test_scores), T_max_curr, device=device)

    for t in range(T_max_curr):
        # Instant lookup: O(1) complexity
        p_test[:, t] = torch.Tensor(calibration_curves[t].predict(test_scores[:, t].detach().cpu().numpy())).to(device)

    return p_test


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def project_to_test_platt(optimal_P, val_scores, test_scores, val_prior_q, t_val, T_max_curr, device,
                          epsilon=1e-4):
    if torch.is_tensor(optimal_P) and isinstance(optimal_P, torch.Tensor):
        optimal_P = optimal_P.to(device)
    else:
        optimal_P = torch.Tensor(optimal_P).to(device)
    p_test = torch.zeros(len(test_scores), T_max_curr, device=device)

    range_t = torch.arange(T_max_curr, device=device).unsqueeze(0)
    max_val = torch.minimum(val_prior_q, t_val)
    mask_val = range_t <= max_val.unsqueeze(1)

    for t in range(T_max_curr):
        mask = mask_val[:, t]

        # Transfer data to CPU/NumPy
        x_train = val_scores[mask, t].detach().cpu().numpy()
        y_train = optimal_P[mask, t] if isinstance(optimal_P, torch.Tensor) else torch.tensor(optimal_P)[mask, t]
        y_train = y_train.detach().cpu().numpy()
        x_test = test_scores[:, t].detach().cpu().numpy()

        if len(x_train) == 0:
            p_test[:, t] = 1.0
            continue

        # Objective Function: MSE
        def objective(params):
            w, b = params
            preds = sigmoid(w * x_train + b)
            return np.mean((preds - y_train) ** 2)

        # Bounds: w must be >= 0 (monotonicity), b is unconstrained
        bounds = [(0, None), (None, None)]

        # Initial guess: w=1.0, b=0.0
        init_params = [1.0, 0.0]

        # SciPy L-BFGS-B exact optimization
        res = minimize(objective, np.array(init_params), method='L-BFGS-B', bounds=bounds)
        w_opt, b_opt = res.x

        # Predict on test data
        p_pred = sigmoid(w_opt * x_test + b_opt)

        # Clip to prevent the singularity (1/0) in your non-convex loss
        p_pred = np.clip(p_pred, epsilon, 1.0 - epsilon)

        # Transfer back to GPU
        p_test[:, t] = torch.tensor(p_pred, dtype=torch.float32, device=device)

    return p_test


def project_to_test_spline(optimal_P, val_scores, test_scores, val_prior_q, t_val, T_max_curr, device,
                           num_knots=7, l2_lambda=0.01, epsilon=1e-4):
    if torch.is_tensor(optimal_P) and isinstance(optimal_P, torch.Tensor):
        optimal_P = optimal_P.to(device)
    else:
        optimal_P = torch.Tensor(optimal_P).to(device)
    p_test = torch.zeros(len(test_scores), T_max_curr, device=device)

    range_t = torch.arange(T_max_curr, device=device).unsqueeze(0)
    max_val = torch.minimum(val_prior_q, t_val)
    mask_val = range_t <= max_val.unsqueeze(1)

    for t in range(T_max_curr):
        mask = mask_val[:, t]

        # Transfer data to CPU/NumPy
        x_train = val_scores[mask, t].detach().cpu().numpy()
        y_train = optimal_P[mask, t] if isinstance(optimal_P, torch.Tensor) else torch.tensor(optimal_P)[mask, t]
        y_train = y_train.detach().cpu().numpy()
        x_test = test_scores[:, t].detach().cpu().numpy()

        if len(x_train) == 0:
            p_test[:, t] = 1.0
            continue

        # Define fixed knots for stability
        x_min, x_max = x_train.min(), x_train.max()
        if x_min == x_max:
            x_max = x_min + 1e-5
        knots = np.linspace(x_min, x_max, num_knots)

        # Precompute basis matrices: max(0, X - knot)
        # shape: (N_samples, num_knots)
        basis_train = np.maximum(0, x_train[:, None] - knots[None, :])
        basis_test = np.maximum(0, x_test[:, None] - knots[None, :])

        # Objective Function: MSE + L2 Regularization (Crucial for bounding beta)
        def objective(params):
            w = params[:-1]  # First 'num_knots' params are weights
            b = params[-1]  # Last param is the bias

            # Spline output mapped through sigmoid
            spline_out = basis_train @ w + b
            preds = sigmoid(spline_out)

            mse = np.mean((preds - y_train) ** 2)
            l2_penalty = l2_lambda * np.sum(w ** 2)

            return mse + l2_penalty

        # Bounds: weights w must be >= 0 (to guarantee monotonicity)
        # Bias b is free (None, None)
        bounds = [(0, None)] * num_knots + [(None, None)]

        # Initialize weights to 1/num_knots, bias to 0
        init_params = np.ones(num_knots + 1)
        init_params[:-1] = 1.0 / num_knots
        init_params[-1] = 0.0

        # SciPy optimization
        res = minimize(objective, init_params, method='L-BFGS-B', bounds=bounds)
        w_opt = res.x[:-1]
        b_opt = res.x[-1]

        # Predict on test data
        spline_out_test = basis_test @ w_opt + b_opt
        p_pred = sigmoid(spline_out_test)

        # Clip to prevent the singularity (1/0)
        p_pred = np.clip(p_pred, epsilon, 1.0 - epsilon)

        # Transfer back to GPU
        p_test[:, t] = torch.tensor(p_pred, dtype=torch.float32, device=device)

    return p_test


def project_to_test_beta(optimal_P, val_scores, test_scores, val_prior_q, t_val, T_max_curr, device, epsilon=1e-4):
    if torch.is_tensor(optimal_P) and isinstance(optimal_P, torch.Tensor):
        optimal_P = optimal_P.to(device)
    else:
        optimal_P = torch.Tensor(optimal_P).to(device)
    p_test = torch.zeros(len(test_scores), T_max_curr, device=device)

    range_t = torch.arange(T_max_curr, device=device).unsqueeze(0)
    max_val = torch.minimum(val_prior_q, t_val)
    mask_val = range_t <= max_val.unsqueeze(1)

    for t in range(T_max_curr):
        mask = mask_val[:, t]

        # Transfer data to CPU/NumPy
        x_train = val_scores[mask, t].detach().cpu().numpy()
        y_train = optimal_P[mask, t] if isinstance(optimal_P, torch.Tensor) else torch.tensor(optimal_P)[mask, t]
        y_train = y_train.detach().cpu().numpy()
        x_test = test_scores[:, t].detach().cpu().numpy()

        if len(x_train) == 0:
            p_test[:, t] = 1.0
            continue

        # 1. Clip inputs to strictly prevent log(0) crashes
        x_train_safe = np.clip(x_train, epsilon, 1.0 - epsilon)
        x_test_safe = np.clip(x_test, epsilon, 1.0 - epsilon)

        # 2. Precompute the log terms so the optimizer doesn't recalculate them every step
        log_x_train = np.log(x_train_safe)
        log_1m_x_train = np.log(1.0 - x_train_safe)

        log_x_test = np.log(x_test_safe)
        log_1m_x_test = np.log(1.0 - x_test_safe)

        # 3. Objective Function: MSE
        def objective(params):
            c, d, b = params
            # Beta calibration formula: c*ln(x) - d*ln(1-x) + b
            logits = c * log_x_train - d * log_1m_x_train + b
            preds = sigmoid(logits)
            return np.mean((preds - y_train) ** 2)

        # 4. Bounds: c >= 0 and d >= 0 to strictly guarantee monotonicity. b is free.
        bounds = [(0, None), (0, None), (None, None)]

        # Initial guess: c=1.0, d=1.0, b=0.0 (Identity mapping on the log-odds scale)
        init_params = [1.0, 1.0, 0.0]

        # 5. SciPy L-BFGS-B exact optimization
        res = minimize(objective, np.array(init_params), method='L-BFGS-B', bounds=bounds)
        c_opt, d_opt, b_opt = res.x

        # 6. Predict on test data
        logits_test = c_opt * log_x_test - d_opt * log_1m_x_test + b_opt
        p_pred = sigmoid(logits_test)

        # 7. Clip the output to prevent the 1/0 singularity in your downstream non-convex loss
        p_pred = np.clip(p_pred, epsilon, 1.0 - epsilon)

        # 8. Transfer back to GPU
        p_test[:, t] = torch.tensor(p_pred, dtype=torch.float32, device=device)

    return p_test


def adaptive_budget_allocation(continuation_probabilities, prior_q, event_times, T_max_curr, device, reach_t_max_is_success=False):
    sim_N = prior_q.shape[0]
    sim_C = torch.zeros(sim_N, dtype=torch.long, device=device)
    sim_active = torch.ones(sim_N, dtype=torch.bool, device=device)
    sim_cum_prob = torch.ones(sim_N, dtype=torch.float32, device=device)
    total_cost = 0.0
    for t_curr in range(T_max_curr):
        # Update active mask based on events observed up to previous step
        event_seen_prev = event_times < t_curr
        sim_active = sim_active & (~event_seen_prev) & (t_curr <= prior_q)

        if not sim_active.any():
            break

        pi = continuation_probabilities[:, t_curr]

        # Random sampling for Test
        rand = torch.rand(sim_N, device=device)
        keep = (rand <= pi) & sim_active

        # Count actual advancements (transitions)
        step_cost = keep.sum().item()
        total_cost += step_cost

        # Update state
        sim_C[keep] += 1
        sim_cum_prob = torch.where(keep, sim_cum_prob * pi, sim_cum_prob)
        sim_active = sim_active & keep

    succeeded = (sim_C > prior_q) | (sim_C > event_times)
    if reach_t_max_is_success:
        succeeded = succeeded | (sim_C == T_max_curr)
    sim_C = torch.where(succeeded, prior_q + 1, 0)

    range_t = torch.arange(T_max_curr, device=device).unsqueeze(0)
    max_test = torch.minimum(prior_q, event_times)
    mask = range_t <= max_test.unsqueeze(1)
    masked_p_test = continuation_probabilities.clone()
    masked_p_test[~mask] = 1
    C_probs = masked_p_test.prod(dim=-1)  # should be equal to sim_cum_prob for samples with C > min(q,t)

    return sim_C, total_cost, C_probs


def construct_final_result(N, val_idxs, val_prior_q, test_idxs, test_prior_q, test_C, test_C_probs, device):
    prior_q = torch.empty(N, device=device)
    prior_q[val_idxs] = val_prior_q
    prior_q[test_idxs] = test_prior_q
    val_C = val_prior_q + 1  # torch.minimum(t[:val_size], val_prior_q)
    # For Validation set, we don't care about C_probs (set to 1.0 or dummy)
    val_size = len(val_prior_q)
    val_C_probs = torch.ones(val_size, device=device)

    # Concatenate
    final_C = torch.empty(N, device=device, dtype=torch.long)
    final_C[val_idxs] = val_C.to(torch.long)
    final_C[test_idxs] = test_C.to(torch.long)

    # final_C[final_C > t] = torch.max(prior_q[final_C > t], final_C[final_C > t])
    final_C_probs = torch.empty(N, device=device, dtype=test_C_probs.dtype)
    final_C_probs[val_idxs] = val_C_probs.to(final_C_probs.dtype)
    final_C_probs[test_idxs] = test_C_probs.to(final_C_probs.dtype)

    return final_C, final_C_probs


def split_to_two_sets(conditional_grid, prior_q, censored_event_time, scores, budget_per_sample, val_size=100,
                      reach_t_max_is_success=False):
    N = len(conditional_grid)
    total_budget = budget_per_sample * N
    perm = np.random.permutation(N)
    val_idxs = perm[:val_size]
    test_idxs = perm[val_size:]
    # --- Data Splitting ---
    # Validation Set: Used to learn the optimal policy parameters (lambda)
    val_grid = conditional_grid[val_idxs]
    val_prior_q = prior_q[val_idxs]
    t_val = censored_event_time[val_idxs]
    val_budget_used = torch.minimum(t_val + 1, val_prior_q + 1).sum().item()
    if total_budget < val_budget_used:
        raise ValueError("Total budget is too small")

    test_grid = conditional_grid[test_idxs]

    test_prior_q = prior_q[test_idxs]
    t_test = censored_event_time[test_idxs]

    target_budget_avg = (budget_per_sample * N - val_budget_used) / (N - val_size)

    val_scores = scores[val_idxs]
    test_scores = scores[test_idxs]
    val_max_steps = torch.minimum(val_prior_q, t_val)

    # if reach_t_max_is_success:
    #     val_max_steps[val_max_steps == conditional_grid.shape[1]] = conditional_grid.shape[1]+1

    return val_idxs, test_idxs, val_grid, val_prior_q, t_val, val_scores, test_grid, test_prior_q, t_test, test_scores, val_max_steps, target_budget_avg, val_budget_used
