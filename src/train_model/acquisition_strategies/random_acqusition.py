import numpy as np

from src.train_model.acquisition_strategies.acquisition_strategy import AcquisitionStrategy


class RandomAcquisition(AcquisitionStrategy):
    def score(self, model, dataset, pool_indices, batch_size=64):
        rng = np.random.RandomState()
        permutation = list(pool_indices)
        rng.shuffle(permutation)
        return {idx: float(i) for i, idx in enumerate(permutation)}
