import numpy as np
import torch

from src.safety_evaluation.budget_allocators.budget_allocator import BudgetAllocator, BudgetAllocationResult
from src.safety_evaluation.calibration.calibration_utils import get_prior


class AdaptiveOptimizedBudgetAllocator(BudgetAllocator):
    """
    Implements a Lagrangian Shadow Price method for optimal budget allocation.

    Mechanism:
    1. Uses the fully observed validation set (first 100 samples) to calibrate a global
       threshold 'lambda' (shadow price). This lambda balances the trade-off between
       budget cost and event discovery probability.
    2. Applies this threshold to the test set in a decentralized, parallel manner.
    3. Ensures a minimum exploration probability to maintain valid IPW properties.
    """

    def __init__(self, conditional_grid, budget_per_sample, taus_range, tau_prior, m_upper_bound, reach_t_max_is_success=False):
        super().__init__(budget_per_sample, taus_range, tau_prior)
        self.conditional_grid = conditional_grid
        self.min_pi = 0.005
        self.reach_t_max_is_success = reach_t_max_is_success
        # self.budget_per_sample *= 500

    @property
    def name(self) -> str:
        return "adaptive_optimized"

    def allocate_budget(self, probability_est: torch.Tensor, x: torch.Tensor, t: torch.Tensor,
                        quantile_est: torch.Tensor) -> BudgetAllocationResult:
        device = self.conditional_grid.device
        N, T_max_curr, T_max_future = self.conditional_grid.shape
        total_budget = self.budget_per_sample * N
        val_size = 100
        perm = np.random.permutation(N)
        val_idxs = perm[:val_size]
        test_idxs = perm[val_size:]
        # --- Data Splitting ---
        # Validation Set: Used to learn the optimal policy parameters (lambda)
        val_grid = self.conditional_grid[val_idxs]

        val_prior_q = get_prior(quantile_est[val_idxs], self.taus_range, self.tau_prior)
        t_val = t[val_idxs]
        val_budget_used = torch.minimum(t_val + 1, val_prior_q + 1).sum().item()
        if total_budget < val_budget_used:
            raise ValueError("Total budget is too small")

        # Test Set: The data we need to mine
        test_grid = self.conditional_grid[test_idxs]

        # val_grid, test_grid = cal(val_grid, test_grid, t_val)

        test_prior_q = get_prior(quantile_est[test_idxs], self.taus_range, self.tau_prior)
        t_test = t[test_idxs]

        # Global Target Budget for the test set
        # We assume the budget density (budget per sample) should be consistent
        target_budget_avg = (self.budget_per_sample * N - val_budget_used) / (N - val_size)
        # target_budget_avg = (target_budget_avg  * val_size - val_prior_q.max().item()) / (val_size + 1)

        # --- Phase 1: Calibration (Finding Lambda) ---
        # We search for a risk threshold (lambda) that results in the target budget usage.
        # We simulate the policy on the Validation set.
        # ==============================================================================
        # CORE POLICY FUNCTION
        # ==============================================================================
        def get_mpc_decision(belief_matrix, t_curr, horizons, prior_q, true_t, active_mask, lam, alpha,
                             current_cum_prob):
            """
            Performs a virtual backward pass on the CURRENT belief to find the optimal
            action for the immediate next step.

            Args:
                belief_matrix: (N, L) Tensor. grid[:, t_curr, t_curr+1:]
                               Prob of event at t+1+k given survival to t_curr.
                horizons: (N,) Tensor. Remaining steps allowed for each sample.
                active_mask: (N,) Boolean mask of currently active samples.
                lam: Scalar. Shadow price of budget.

            Returns:
                pi_immediate: (N,) Tensor. Probability to continue to next step.
            """
            """
                        Calculates pi_t based on Generalized Cost Sensitivity:
                        Target_Prob ~ 1 / (Expected_Remaining_Cost ** alpha)
                        """

            # Estimate Future:
            # E[Remaining] = 1.0 (current step) + P(No Event) * (Rest of Horizon)
            # This explicitly handles the "Stop at Event" condition.

            remaining_steps_max = torch.clamp(prior_q - t_curr + 1, min=0, max=belief_matrix.shape[1]).float()
            mask = torch.arange(0, belief_matrix.shape[1], device=belief_matrix.device).unsqueeze(
                0) <= remaining_steps_max.unsqueeze(1)
            masked_belief_matrix = belief_matrix * mask
            expected_remaining = (masked_belief_matrix * torch.arange(1, belief_matrix.shape[1] + 1,
                                                                      device=belief_matrix.device).unsqueeze(0)).sum(
                dim=-1)
            expected_remaining += remaining_steps_max * (belief_matrix * (~mask)).sum(dim=-1)
            # expected_remaining = 1.0 + (1.0 - hazard_now[:, 0]) * (remaining_steps_max - 1.0)
            # expected_remaining = torch.clamp(expected_remaining, min=1.0)

            # Total Cost = Sunk + Remaining
            # This ensures we penalize samples that have become expensive overall.
            total_est_cost = expected_remaining

            # 3. KKT Condition Target
            # pi* = 1 / sqrt(lambda * Total_Cost)
            target_pi_total = 1.0 / torch.sqrt(lam * total_est_cost + 1e-12)  # TODO: should add * len(total_est_cost) to lambda
            target_pi_total = torch.clamp(target_pi_total, max=1.0)

            # 4. Sequential Steering
            # We need: accum_prev * rho = target_pi
            safe_accum = torch.clamp(current_cum_prob, min=1e-9)
            rho = target_pi_total / safe_accum

            # rho_floor = self.min_pi / safe_accum
            # rho = torch.maximum(rho, rho_floor)
            rho = torch.clamp(rho, max=1.0)
            return rho

            # horizons = horizons.clip(min=0)
            # reward = belief_matrix.cpu()[list(range(len(horizons))), horizons.long().cpu()]
            # reward = reward.to(horizons.device)
            # cost = horizons
            # score = reward / cost
            # future_times = torch.arange(0, belief_matrix.shape[1], device=device) + 1
            # # score = (belief_matrix * (1/future_times)).sum(dim=-1)
            # safe_cum_prob = torch.clamp(current_cum_prob, min=1e-9)
            # min_p_dynamic = self.min_pi / safe_cum_prob
            # min_p_dynamic = torch.clamp(min_p_dynamic, max=1.0)
            #
            # # Optimization: We only need to iterate up to the max horizon in this batch
            # pi_immediate = torch.sigmoid(self.temperature * (score - lam))
            # pi_immediate = torch.maximum(pi_immediate, min_p_dynamic)
            #
            #
            # return pi_immediate

        def simulate_process(grid, prior_q, true_t, lam, alpha, stochastic=False):
            """
            Simulates the process for the whole batch over time.
            stochastic=True -> Samples decisions (for Test)
            stochastic=False -> Uses expected cost (for Validation/Calibration)

            Returns:
                sim_C: Number of steps taken
                sim_reach_prior_prob: P(prior_q <= final_C) after considering early stops
                total_cost: Total budget used
            """
            sim_N = grid.shape[0]
            sim_C = torch.zeros(sim_N, dtype=torch.long, device=device)
            sim_active = torch.ones(sim_N, dtype=torch.bool, device=device)
            sim_active_and_alive = torch.ones(sim_N, dtype=torch.bool, device=device)
            sim_cum_prob = torch.ones(sim_N, dtype=torch.float32, device=device)

            # Track probability of reaching prior_q (or stopping early)

            total_cost = 0.0

            for t_curr in range(T_max_curr):
                # Update active mask based on events observed up to previous step
                event_seen_prev = true_t < t_curr
                sim_active = sim_active & (~event_seen_prev) & (t_curr <= prior_q)

                if not sim_active.any():
                    break

                # Prepare Future Belief for MPC
                belief = grid[:, t_curr, t_curr:]

                # Call MPC Policy
                pi = get_mpc_decision(
                    belief,
                    t_curr,
                    prior_q - (t_curr + 1),
                    prior_q,
                    true_t,
                    sim_active,
                    lam,
                    alpha,
                    sim_cum_prob
                )

                if stochastic:
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

                else:
                    pi_active = pi * sim_active.float()

                    # Expected number of samples that advance from t to t+1
                    expected_step_cost = (sim_cum_prob * pi_active).sum().item()
                    total_cost += expected_step_cost

                    sim_cum_prob = torch.where(sim_active, sim_cum_prob * pi, sim_cum_prob)
            if stochastic:
                succeeded = (sim_C > prior_q) | (sim_C > true_t)
                succeeded = succeeded | (sim_C == T_max_curr) if self.reach_t_max_is_success else succeeded
                sim_reach_prior_prob = torch.where(
                    succeeded,
                    sim_cum_prob,  # Keep accumulated probability
                    torch.ones_like(sim_cum_prob) * self.min_pi  # Floor for failures
                )
                sim_C = torch.where(succeeded, prior_q+1, sim_C)
            else:
                sim_reach_prior_prob = sim_cum_prob


            return sim_C, sim_reach_prior_prob, total_cost

        # Grid of Alpha values to test.
        # 0.5 is theoretical optimum. We search around it.
        lam_low, lam_high = 0, 256.0
        for _ in range(25):
            mid = (lam_low + lam_high) / 2
            _, C_probs, val_expected_cost = simulate_process(val_grid, val_prior_q, t_val, mid, None, stochastic=False)
            assert val_expected_cost <= val_budget_used
            avg_cost = val_expected_cost / val_size

            if abs(avg_cost - target_budget_avg) < 1e-10:  # Tolerance
                break

            if avg_cost > target_budget_avg:
                lam_low = mid  # Need to be more expensive (higher lambda)
            else:
                lam_high = mid

        best_lambda = (lam_low + lam_high) / 2
        mean_val_weight = (1/C_probs).mean().item()
        max_val_weight = (1/C_probs).max().item()

        best_alpha = None
        test_C, test_C_probs, test_total_used = simulate_process(
            test_grid, test_prior_q, t_test, best_lambda, best_alpha, stochastic=True
        )
        test_avg_cost = test_total_used / len(test_C)
        # print(f"err: {val_avg_cost - test_avg_cost}")
        total_budget_used = test_total_used + val_budget_used

        # Reconstruct full-size tensors
        # Val C: Standard logic (stop at event or horizon)
        prior_q = torch.empty(N, device=device)
        prior_q[val_idxs] = val_prior_q
        prior_q[test_idxs] = test_prior_q
        val_C = val_prior_q + 1  # torch.minimum(t[:val_size], val_prior_q)
        # For Validation set, we don't care about C_probs (set to 1.0 or dummy)
        val_C_probs = torch.ones(val_size, device=device)

        # Concatenate
        final_C = torch.empty(N, device=device)
        final_C[val_idxs] = val_C
        final_C[test_idxs] = test_C

        # final_C[final_C > t] = torch.max(prior_q[final_C > t], final_C[final_C > t])
        final_C_probs = torch.empty(N, device=device, dtype=test_C_probs.dtype)
        final_C_probs[val_idxs] = val_C_probs.to(final_C_probs.dtype)
        final_C_probs[test_idxs] = test_C_probs.to(final_C_probs.dtype)

        # Enforce the user's explicit safety clamp for early stopping
        # This ensures P(V <= C) is floored at 0.01 even if dropped early.
        # final_C_probs[final_C < prior_q] = self.min_pi
        # target_budget_avg = (self.budget_per_sample * N - val_budget_used) / (N - val_size)
        # expected_test_budget = val_expected_cost / val_size * (N - val_size)
        # total_budget_used = int(target_budget_avg * (N - val_size) + val_budget_used)

        print(f"adaptive weights {mean_val_weight} | lambda {best_lambda} | total_budget_used {total_budget_used} "
              f"| total_budget {total_budget} | # observed: {(final_C > t).float().sum().item()}"
              f"| achieved prior: {(final_C.squeeze() >= prior_q).float().sum().item()}")
        return BudgetAllocationResult(quantile_est, final_C, final_C_probs, total_budget_used, mean_weight=mean_val_weight,
                                      max_weight=max_val_weight)
