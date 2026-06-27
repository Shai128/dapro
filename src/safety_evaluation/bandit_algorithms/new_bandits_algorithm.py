import numpy as np
import torch

from src.safety_evaluation.bandit_algorithms.abstract_bandit_algorithm import BanditAlgorithm
from src.safety_evaluation.reward_functions.abstract_reward import BanditsRewardFunction


class NewBanditsAlgorithm(BanditAlgorithm):

    def __init__(self, reward_function:BanditsRewardFunction, device: torch.device, k: int, n_monte_carlo:int= 1):
        super().__init__(reward_function, device)
        self.k = k
        self.n_monte_carlo = n_monte_carlo

    def run(self, total_budget: int, n_arms: int, max_draws_per_arm: torch.Tensor, max_time: int, **kwargs) -> torch.Tensor:
        montes = self.n_monte_carlo
        temp_c = torch.zeros(montes, n_arms).long().to(self.device)
        device = self.device
        k = self.k
        N = n_arms
        # event_observed_all = num_attempts <= temp_c
        max_c = max_draws_per_arm
        idx_arange = torch.arange(N, device=device).long()
        montes_arange = torch.arange(montes, device=device).long()
        relevant_samples_all = (temp_c < max_draws_per_arm)
        max_temp_c = temp_c.max(dim=0).values
        initial_curr_max_c = torch.full(
            max_temp_c.shape,
            -99,
            device=max_temp_c.device,
            dtype=temp_c.dtype
        )
        maximal_c_as_idx = torch.ones_like(temp_c) * (max_time)-1
        remaining_budget = total_budget
        while remaining_budget > 1e-10:
            if not relevant_samples_all.any():
                break
            with torch.no_grad():

                curr_probabilities = self.reward_function.get_rewards(idx_arange, torch.minimum(temp_c,maximal_c_as_idx), **kwargs) * relevant_samples_all
                # curr_probabilities = probability_est[idx_arange, torch.minimum(temp_c,maximal_c_as_idx) ] * relevant_samples_all
                topk_vals, topk_idx = torch.topk(curr_probabilities, k=min(k, len(curr_probabilities)), dim=-1)
                probs = topk_vals / topk_vals.sum(dim=-1).unsqueeze(-1)
                sample_idx = topk_idx[montes_arange, torch.multinomial(probs, 1).squeeze()]

                curr_selected_c = temp_c[montes_arange, sample_idx]

                curr_max_c = initial_curr_max_c.scatter_reduce(
                    0,
                    sample_idx,
                    curr_selected_c,
                    reduce="amax",
                    include_self=False
                )
                unique_idxs = torch.unique(sample_idx)
                new_max_idx = curr_max_c[unique_idxs] >= max_temp_c[unique_idxs]

                budget_used = new_max_idx.sum()
                if budget_used >= remaining_budget:
                    sample_idx = sample_idx[:int(remaining_budget)]
                    temp_c[montes_arange[:int(remaining_budget)], sample_idx] += 1
                    remaining_budget = 0
                else:
                    temp_c[montes_arange, sample_idx] += 1
                    remaining_budget -= budget_used
                    relevant_samples_all[montes_arange, sample_idx] = temp_c[montes_arange, sample_idx] < max_c[sample_idx]
                    max_temp_c[unique_idxs] += new_max_idx

        m = np.random.randint(0, montes, size=1)
        C = temp_c[m].clone().squeeze()
        return C

    @property
    def name(self) -> str:
        return "new"