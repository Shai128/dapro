"""
solve_exact_fast.py
────────────────────────────────────────────────────────────────────────────────
Algorithm: Dual decomposition (bisection on λ) + Gauss-Seidel BCD inner loop.

Key optimisations over previous versions
─────────────────────────────────────────
1. EXACT PER-COORDINATE MINIMISER  (derivation below)
   For fixed λ and all y[i,j] j≠t, the Lagrangian in u = y[i,t] is:
     L_i(u) = A_i·exp(-u) + B_i·exp(u)   (A_i, B_i > 0, independent of u)
   Minimiser: u* = 0.5·log(A_i / B_i)
   After substitution of current cumY values:
     u*[i] = y[i,t] − 0.5·(log λ + log N + log suffix_t[i] + cumY[i, C_i−1])
   No step size, no iteration — one formula per (i,t).

2. INCREMENTAL cumY UPDATE  (avoids full O(NL) recompute per column)
   After updating y[:,t], cumY[:,s] shifts by Δ for all s≥t.
   Done in one numpy broadcast: cumY[:,t:] += delta[:,None]

3. SKIP PAV WHEN NOT NEEDED
   PAV violations drop 88 → 52 → 14 columns per pass.
   Check for violations before calling isotonic_regression.
   Later passes and warm-started outer iterations spend almost no time on PAV.

4. TWO-SIDED WARM START
   Maintain separate Y_lo / Y_hi for the two bisection sides.
   Warm-starting from the closest-budget solution halves the required
   inner passes after the first few outer iterations.

Derivation of per-coordinate minimiser
───────────────────────────────────────
Define (for fixed t and all other variables):
  z_t⁻  = cumY[i, t−1]   (prefix sum, doesn't include y[i,t])
  R_i   = cumY[i, C_i−1] − cumY[i, t]   (suffix sum beyond t)
  W_i   = Σ_{s≥t} exp(cumY[i,s] − cumY[i,t]) · mask[i,s]
        = N · suffix_t[i] · exp(−cumY[i,t])

Then: A_i = (1/N)·exp(−z_t⁻ − R_i)
      B_i = (λ/N)·W_i·exp(z_t⁻)

u* = 0.5·log(A_i/B_i)
   = 0.5·(−z_t⁻ − R_i − log λ − log W_i − z_t⁻)
   = y[i,t] − 0.5·(log λ + log N + log suffix_t[i] + cumY[i, C_i−1])
"""

from __future__ import annotations
import numpy as np
import torch
from sklearn.isotonic import isotonic_regression


def solve_exact_fast(
        S: torch.Tensor,
        C: torch.Tensor,
        B_bar: float,
        max_outer: int = 60,
        max_inner: int = 10,
        tol: float = 1e-9,
        verbose: bool = True,
) -> np.ndarray:
    S = S.detach().cpu().numpy()
    C = C.detach().cpu().numpy()
    """
    Parameters
    ----------
    S     : (N, L) float64 scores
    C     : (N,)   int     sequence lengths  (C[i] active steps for sample i)
    B_bar : float  per-sample budget
    Returns
    -------
    P_final : (N, L) float64,  P[i,t] = 0 for t >= C[i]
    """
    S = np.asarray(S, dtype=np.float64)
    C = np.asarray(C, dtype=np.int64)
    N, L = S.shape

    t_range = np.arange(L)
    mask = (t_range[None, :] < C[:, None]).astype(np.float64)
    arange_N = np.arange(N)
    end_idx = np.maximum(C - 1, 0)  # index of last active step per sample

    # Per-column sort indices (computed once)
    sort_indices = []
    for t in range(L):
        valid = np.where(mask[:, t] > 0)[0]
        if len(valid) >= 2:
            order = np.argsort(S[valid, t], kind="stable")
            sort_indices.append((valid, valid[order]))
        else:
            sort_indices.append(None)

    # Init: p/(1−p) = B_bar → p = B_bar/(1+B_bar)  (geometric series approx)
    p_init = float(np.clip(B_bar / (1.0 + B_bar), 1e-4, 1.0 - 1e-4))
    Y0 = np.full((N, L), np.log(p_init), dtype=np.float64)
    Y0[mask == 0] = -1e9

    # ── Helpers ───────────────────────────────────────────────────────────────

    def compute_budget(Y):
        return float(np.mean(np.sum(np.exp(np.cumsum(Y, axis=1)) * mask, axis=1)))

    def apply_isotonic_all(Y):
        for t in range(L):
            si = sort_indices[t]
            if si is None:
                continue
            _, sr = si
            y_iso = isotonic_regression(Y[sr, t], increasing=True)
            Y[sr, t] = np.minimum(y_iso, 0.0)
        return Y

    def apply_budget_bisection(Y, target):
        """Uniform log-shift preserves all isotonic orderings."""

        def _b(d):
            Ys = np.where(mask > 0, np.minimum(Y + d, 0.0), -1e9)
            return float(np.mean(np.sum(np.exp(np.cumsum(Ys, axis=1)) * mask, axis=1)))

        if _b(0.0) <= target + tol:
            return Y
        lo, hi = -60.0, 0.0
        for _ in range(100):
            mid = (lo + hi) * 0.5
            if _b(mid) > target:
                hi = mid
            else:
                lo = mid
        return np.where(mask > 0, np.minimum(Y + (lo + hi) * 0.5, 0.0), -1e9)

    # ── Gauss-Seidel BCD inner solver ─────────────────────────────────────────

    def inner_solve(lam, Y_init, n_iter):
        Y = Y_init.copy()
        log_lam = np.log(max(lam, 1e-300))
        log_N = np.log(N)

        for _ in range(n_iter):
            Y_prev = Y.copy()
            cumY = np.cumsum(Y, axis=1)  # recomputed once per pass

            for t in range(L):
                # suffix_t[i] = (1/N)·Σ_{s≥t} exp(cumY[i,s])·mask[i,s]
                exp_slice = np.exp(np.clip(cumY[:, t:], -700, 0)) * mask[:, t:]
                suffix_t = np.sum(exp_slice, axis=1) / N  # (N,)

                # Exact coordinate minimiser (see module docstring)
                log_suf = np.log(np.maximum(suffix_t, 1e-300))
                u_star = (Y[:, t]
                          - 0.5 * (log_lam + log_N + log_suf
                                   + cumY[arange_N, end_idx]))

                old_yt = Y[:, t].copy()
                Y[:, t] = np.where(mask[:, t] > 0,
                                   np.minimum(u_star, 0.0), -1e9)

                # ── PAV only if violations exist (key speedup) ────────────────
                si = sort_indices[t]
                if si is not None:
                    _, sr = si
                    if np.any(Y[sr[:-1], t] > Y[sr[1:], t] + 1e-12):
                        y_iso = isotonic_regression(Y[sr, t], increasing=True)
                        Y[sr, t] = np.minimum(y_iso, 0.0)

                # Incremental cumY update — O(N) instead of O(NL) per column
                delta = Y[:, t] - old_yt
                cumY[:, t:] += delta[:, None]

            diff = np.max(np.abs((Y - Y_prev)[mask > 0]))
            if diff < tol:
                break

        return Y

    # ── Outer bisection on λ ──────────────────────────────────────────────────

    lam_lo, lam_hi = 1e-8, 1e14
    Y_lo = Y0.copy()  # warm-start for λ_lo side (budget > B_bar)
    Y_hi = Y0.copy()  # warm-start for λ_hi side (budget < B_bar)
    b_lo = compute_budget(Y0)
    b_hi = compute_budget(Y0)
    Y_best = Y0.copy()
    prev_lam = None

    if verbose:
        print(f"  p_init={p_init:.4f}  initial_budget={b_lo:.4f}  target={B_bar}")

    for outer in range(max_outer):
        lam_mid = float(np.sqrt(lam_lo * lam_hi))  # geometric midpoint (λ lives on log-scale)

        if prev_lam is not None and abs(lam_mid - prev_lam) / lam_mid < tol:
            break
        prev_lam = lam_mid

        # Warm-start from whichever side's budget was closer to B_bar
        Y_warm = Y_lo if abs(b_lo - B_bar) < abs(b_hi - B_bar) else Y_hi
        Y_mid = inner_solve(lam_mid, Y_warm, max_inner)
        b_mid = compute_budget(Y_mid)

        if b_mid > B_bar:
            lam_lo, Y_lo, b_lo = lam_mid, Y_mid.copy(), b_mid
        else:
            lam_hi, Y_hi, b_hi = lam_mid, Y_mid.copy(), b_mid
            Y_best = Y_mid.copy()

        if verbose:
            print(f"  outer {outer + 1:3d}  λ={lam_mid:.4e}  budget={b_mid:.5f}/{B_bar}")

        if abs(b_mid - B_bar) / max(abs(B_bar), 1e-12) < tol:
            if verbose:
                print(f"  → converged at outer iteration {outer + 1}")
            break

    # ── Final hard enforcement ────────────────────────────────────────────────
    Y_best = apply_isotonic_all(Y_best)
    Y_best = apply_budget_bisection(Y_best, B_bar)
    Y_best = apply_isotonic_all(Y_best)  # re-check (usually no-op)

    # ── Verify ────────────────────────────────────────────────────────────────
    P_final = np.exp(Y_best) * mask
    budget_final = compute_budget(Y_best)
    mono_ok = all(
        not np.any(P_final[sr, t][:-1] > P_final[sr, t][1:] + 1e-9)
        for t, si in enumerate(sort_indices)
        if si is not None
        for _, sr in [si]
    )
    Y_masked = np.where(mask > 0, Y_best, 0.0)
    obj = float(np.mean(np.exp(-np.sum(Y_masked, axis=1))))
    if verbose:
        print(f"\nBudget:       {budget_final:.10f} <= {B_bar}  "
              f"{'✓' if budget_final <= B_bar + 1e-9 else '✗ VIOLATED'}")
        print(f"Monotonicity: {'✓ exact' if mono_ok else '✗ VIOLATED'}")
        print(f"Objective:    {obj:.6f}")
    assert budget_final < B_bar + 1e-8 and mono_ok
    return P_final


def fit_isotonic_maps(S_cal, P_opt, C_cal):
    """Fit one isotonic map per time step in log-space for generalisation."""
    from sklearn.isotonic import IsotonicRegression as _IR
    S_cal = np.asarray(S_cal, dtype=np.float64)
    Y_opt = np.log(np.clip(np.asarray(P_opt, dtype=np.float64), 1e-15, 1.0))
    C_cal = np.asarray(C_cal, dtype=np.int64)
    N, L = S_cal.shape
    mask_np = np.arange(L)[None, :] < C_cal[:, None]
    models = []
    for t in range(L):
        valid = np.where(mask_np[:, t])[0]
        if len(valid) >= 2:
            ir = _IR(increasing=True, out_of_bounds="clip")
            ir.fit(S_cal[valid, t], Y_opt[valid, t])
            models.append(ir)
        else:
            models.append(None)
    return models


def predict_isotonic_maps(S_test, C_test, models):
    """Apply fitted M_t maps to new scores. Returns P_test (N_test, L)."""
    S_test = np.asarray(S_test, dtype=np.float64)
    C_test = np.asarray(C_test, dtype=np.int64)
    N, L = S_test.shape
    mask_np = np.arange(L)[None, :] < C_test[:, None]
    Y_out = np.full((N, L), -1e9, dtype=np.float64)
    for t, ir in enumerate(models):
        if ir is None:
            continue
        valid = np.where(mask_np[:, t])[0]
        if len(valid) > 0:
            Y_out[valid, t] = np.minimum(ir.predict(S_test[valid, t]), 0.0)
    return np.exp(Y_out) * mask_np


if __name__ == "__main__":
    import time

    rng = np.random.default_rng(42)
    N, L = 100, 200
    S = rng.standard_normal((N, L))
    C = rng.integers(L // 2, L + 1, size=N)
    B_bar = 8.0
    print(f"Problem: N={N}, L={L}, B_bar={B_bar}\n")
    t0 = time.perf_counter()
    P = solve_exact_fast(S, C, B_bar, verbose=True)
    print(f"\nWall-clock: {time.perf_counter() - t0:.3f}s")
