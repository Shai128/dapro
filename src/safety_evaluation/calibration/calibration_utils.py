import numpy as np
import torch
from scipy.optimize import bisect


def get_prior(q: torch.Tensor, taus_range: torch.Tensor, tau_prior: float):
    idx = abs(taus_range - tau_prior).argmin()
    return q[..., idx]

def sample_calibration_set(
        prior_quantile_est,
        C_probs,
        num_attempts
):
    # Determine the number of samples per instance.
    # Note: C is computed as a (n, 1) array. The generic runner accepts per-prompt max attempts.
    C = torch.where(torch.rand(len(prior_quantile_est), device=C_probs.device) < C_probs, prior_quantile_est,
                    0).int().reshape(
        -1, 1)
    num_attempts = num_attempts.reshape(-1, 1)
    T_tilde = torch.minimum(num_attempts, C)
    return T_tilde, C


def sample_calibration_set_new(
        prior_quantile_est,
        C_probs,
        num_attempts
):
    # Determine the number of samples per instance.
    # Note: C is computed as a (n, 1) array. The generic runner accepts per-prompt max attempts.
    C = torch.where(torch.rand(len(prior_quantile_est), device=C_probs.device) < C_probs, prior_quantile_est,
                    0).int().reshape(
        -1, 1)
    num_attempts = num_attempts.reshape(-1, 1)
    T_tilde = torch.minimum(num_attempts, C)
    return T_tilde, C


def constraint_violation(lambda_val, w, b):
    """
    Computes the difference between the left-hand side of the constraint
    sum(w_i * p_i) and b, where p_i = min{1, 1/sqrt(lambda*w_i)}.
    """
    p = np.minimum(1, 1 / np.sqrt(lambda_val * w))
    return np.sum(w * p) - b


def solve_optimization(w, b, tol=1e-8):
    """
    Solves the optimization problem:
         min   sum(1/p_i)
         s.t.  sum(w_i * p_i) = b,   0 < p_i <= 1,
    by finding the Lagrange multiplier lambda such that the constraint holds.

    Parameters:
        w   : array-like, weights (all positive)
        b   : positive scalar, right-hand side of the constraint
        tol : tolerance for the bisection algorithm

    Returns:
        p   : optimal vector p*, where each p_i = min{1, 1/sqrt(lambda*w_i)}
        lambda_star : the Lagrange multiplier found
    """
    w = np.array(w, dtype=float)

    # Check feasibility: b must be no more than sum(w) since p_i <= 1.
    if b > np.sum(w):
        return np.ones_like(w), None

    # # Set a lower bound for lambda.
    # lambda_low = 1e-12
    # # Set an initial upper bound: choose lambda_high so that for every i, 1/sqrt(lambda_high*w_i) < 1.
    # # A sufficient condition is lambda_high > max(1/w).
    # lambda_high = max(1 / w) * 10.0
    #
    # # Ensure that our bounds bracket a zero of the function.
    # # When lambda is very small, p_i = 1 for each i, so the constraint is sum(w) - b (which is >= 0).
    # # We need constraint_violation(lambda_low, w, b) >= 0 and constraint_violation(lambda_high, w, b) <= 0.
    # f_low = constraint_violation(lambda_low, w, b)
    # f_high = constraint_violation(lambda_high, w, b)
    #
    # # Increase lambda_high until f_high is negative.
    # while f_high > 0:
    #     lambda_high *= 2
    #     f_high = constraint_violation(lambda_high, w, b)
    #
    # # Use bisection to find lambda such that the constraint is met.
    # lambda_star = bisect(constraint_violation, lambda_low, lambda_high, args=(w, b), xtol=tol)

    # With lambda_star in hand, compute the optimal p.
    lambda_star = (( (1/(b/len(w))) * np.sqrt(w).mean())) ** 2
    assert abs(lambda_star - (( (1 / b) * np.sqrt(w).sum() ) ** 2)) < 1e-5
    p_opt = np.minimum(1,1/ (( (1/(b/len(w))) * np.sqrt(w).mean()) * np.sqrt(w)))
    # p_opt = np.minimum(1, 1 / np.sqrt(lambda_star * w))
    return p_opt, lambda_star




