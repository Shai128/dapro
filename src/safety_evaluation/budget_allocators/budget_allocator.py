import abc
import dataclasses

import torch

@dataclasses.dataclass
class BudgetAllocationResult:
    f: torch.Tensor
    C: torch.Tensor
    C_probs: torch.Tensor
    total_budget_used: int = None
    mean_weight: float = None
    max_weight: float = None
    additional_metrics: dict = None

class BudgetAllocator(abc.ABC):
    def __init__(self, budget_per_sample: float, taus_range: torch.Tensor, tau_prior: float):
        self.budget_per_sample = budget_per_sample
        self.taus_range = taus_range
        self.tau_prior = tau_prior

    @abc.abstractmethod
    def allocate_budget(self, probability_est: torch.Tensor, x: torch.Tensor, t: torch.Tensor,
                        quantile_est: torch.Tensor) -> BudgetAllocationResult:
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass
