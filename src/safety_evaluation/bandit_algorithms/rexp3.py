import numpy as np
import torch

from src.safety_evaluation.bandit_algorithms.abstract_bandit_algorithm import BanditAlgorithm
from src.safety_evaluation.reward_functions.abstract_reward import BanditsRewardFunction


def rexp3_algorithm(delta_t_percentage: float, gamma: float, budget: int, N: int, reward_function: BanditsRewardFunction,
                   max_draws_per_arm: torch.Tensor, device, **kwargs):
    delta_t = int(delta_t_percentage * N)
    j = 1
    budget_used = 0
    activations_per_arm = torch.zeros(N, device=device, dtype=torch.long)
    relevant_arms = torch.ones(N, device=device, dtype=torch.bool)
    while j <= np.ceil(budget / delta_t):
        if not relevant_arms.any() or budget_used > budget:
            break
        tau = (j - 1) * delta_t
        weights = torch.ones(N, device=device)
        for _ in range(tau, min(budget, tau + delta_t)):
            if not relevant_arms.any() or budget_used > budget:
                break
            weights = weights
            drawing_probs = (1 - gamma) * (weights / weights.sum()) + gamma / N
            drawing_probs[~relevant_arms] = 0
            sample_idx = torch.multinomial(drawing_probs, 1).squeeze().item()
            reward = reward_function.get_reward(sample_idx, activations_per_arm[sample_idx].item()+1, **kwargs)
            rewards = torch.zeros(N, device=device)
            rewards[sample_idx] = reward / drawing_probs[sample_idx] # TODO: check this
            weights = weights * torch.exp(gamma * rewards / N)
            activations_per_arm[sample_idx] += 1
            relevant_arms[sample_idx] = activations_per_arm[sample_idx] < max_draws_per_arm[sample_idx]
            budget_used += 1
            # curr_probabilities = probability_est[idx_arange, temp_c] * relevant_samples_all

        j += 1
    if budget_used != budget:
        print(f"Warning! budget_used: {budget_used} first_step_budget: {budget}")

    return activations_per_arm

class Rexp3(BanditAlgorithm):


    def __init__(self, reward_function:BanditsRewardFunction, device: torch.device, gamma: float = 0.1, delta_t: float = 0.1):
        super().__init__(reward_function, device)
        self.gamma = gamma
        self.delta_t = delta_t

    def run(self, total_budget: int, n_arms: int, max_draws_per_arm: torch.Tensor, max_time: int, **kwargs) -> torch.Tensor:
        return rexp3_algorithm(self.delta_t, self.gamma,total_budget, n_arms, self.reward_function, max_draws_per_arm, self.device, **kwargs)


    @property
    def name(self) -> str:
        return f"rexp3_gamma={self.gamma}_delta={self.delta_t}"