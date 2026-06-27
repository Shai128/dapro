import torch

from src.safety_evaluation.bandit_algorithms.abstract_bandit_algorithm import BanditAlgorithm
from src.safety_evaluation.reward_functions.abstract_reward import BanditsRewardFunction


def discounted_ucb(budget: int, n_arms: int, reward_function: BanditsRewardFunction, gamma: float, xi: float, max_draws_per_arm: torch.Tensor, device, **kwargs):
    # from On Upper-Confidence Bound Policies for Switching Bandit Problems
    # xi should be between 1/2 and 1
    # gamma should be between 1/2 and 1

    reward_sum_per_arm_mult_gamma = torch.zeros(n_arms, device=device)
    gamma_sum_history_per_arm = torch.zeros(n_arms, device=device)
    arms_activations = torch.zeros(n_arms, device=device, dtype=torch.int32)
    rewards_bound = reward_function.get_reward_upper_bound()
    for t in range(0, budget):
        if t < n_arms:
            arms_activations[t] += 1
            reward = reward_function.get_reward(t, arms_activations[t].item(), **kwargs)
            reward_sum_per_arm_mult_gamma[t] = gamma * reward_sum_per_arm_mult_gamma[t] + reward
            gamma_sum_history_per_arm[t] = gamma * gamma_sum_history_per_arm[t] + gamma
        else:
            estimated_rewards = reward_sum_per_arm_mult_gamma / gamma_sum_history_per_arm
            n_t_gamma = gamma_sum_history_per_arm.sum()
            c_t = 2 * rewards_bound * torch.sqrt(xi * torch.log(n_t_gamma / gamma_sum_history_per_arm))
            value_for_arg_max = estimated_rewards + c_t
            value_for_arg_max[arms_activations >= max_draws_per_arm] = 0
            best_arm = torch.argmax(value_for_arg_max).item()
            gamma_sum_history_per_arm[best_arm] = gamma * gamma_sum_history_per_arm[best_arm] + gamma

            arms_activations[best_arm] += 1
            if arms_activations[best_arm].item() < max_draws_per_arm[best_arm].item():
                new_reward = reward_function.get_reward(best_arm, arms_activations[best_arm].item())
            else:
                new_reward = 0

            reward_sum_per_arm_mult_gamma[best_arm] = gamma * reward_sum_per_arm_mult_gamma[best_arm] + new_reward

    return arms_activations

class DiscountedUCB(BanditAlgorithm):


    def __init__(self, reward_function:BanditsRewardFunction, device: torch.device, gamma: float = 0.1, xi: float = 0.1):
        super().__init__(reward_function, device)
        self.gamma = gamma
        self.xi = xi

    def run(self, total_budget: int, n_arms: int, max_draws_per_arm: torch.Tensor, max_time: int, **kwargs) -> torch.Tensor:
        return discounted_ucb(total_budget, n_arms, self.reward_function, self.gamma, self.xi, max_draws_per_arm, self.device, **kwargs)


    @property
    def name(self) -> str:
        return f"ducb_gamma={self.gamma}_xi={self.xi}"