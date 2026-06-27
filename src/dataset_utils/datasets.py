import torch
from torch.utils.data import Dataset
import os
import math
import copy
import json
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Callable, Optional, Dict, Any, Union

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset, Dataset
from tqdm import trange, tqdm
import matplotlib.pyplot as plt


# Dataset for discrete-time survival analysis
class SurvivalDataset(Dataset):
    def __init__(self, x, y, t, dataset_name):
        # x: numpy array (N, features, time_steps)
        # t_tilde: numpy array (N,), event times (1..T or T for censored)
        self.x = x.detach().to(torch.float32)
        self.y = y.detach().to(torch.long)  # (N,)
        # t = torch.argmax(y.float(), dim=1)
        # t[y.float().max(dim=1).values == 0] = 200
        self.t = t.detach()
        # event indicator: 1 if event observed (t<max), else 0
        self.max_t = self.x.size(1)
        self.delta = (t < self.max_t).long()
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.delta[idx]

    @property
    def name(self):
        return self.dataset_name


class PartialSequenceDataset(SurvivalDataset):
    """
    Partial-observation sequence dataset.

    - X: Tensor [N, T, D]
    - Y: Tensor [N, T] with binary indicators (0/1). The first 1 is the event; none -> censored.
    - obs_lens[i]: number of X time points currently revealed (so X_prefix = X[i, :L, :])
    - label_known[i, t]: whether Y[i, t] has been revealed to the learner.

    initialize_seed(seed_indices) will mark those indices as fully observed (obs_len=T, all labels known).
    observe_next_step(indices) reveals Y[i, L-1] (the label at the currently last observed time)
        and then increments obs_lens[i] by 1 (revealing the next X point).
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor, t, dataset_name: str , initial_obs_len: int = 1):
        super().__init__(x, y, t, dataset_name)
        assert x.ndim == 3 and y.ndim == 2
        assert x.shape[0] == y.shape[0] and x.shape[1] == y.shape[1]
        self.N, self.T, self.D = x.shape
        device = y.device
        # observed prefix length per sample (number of X timepoints we currently have)
        # default: many non-seed samples start with only X_0 observed (initial_obs_len = 1)
        self.obs_lens = torch.Tensor([initial_obs_len] * self.N).to(device)

        # label_known mask: which Y[t] values have been revealed
        self.label_known = torch.zeros((self.N, self.T), dtype=torch.bool, device=device)

        # true event times (None if no event within T)
        self.true_event_time = []
        for i in range(self.N):
            ys = y[i]
            ones = (ys == 1).nonzero(as_tuple=False)
            if len(ones) == 0:
                self.true_event_time.append(None)
            else:
                self.true_event_time.append(int(ones[0].item()))
        self._init_precomputed_label_info()

        # If you want, you can initialize label_known for the first timepoint's X (but per your protocol
        # Y_0 is initially missing for non-seed samples). We therefore leave label_known false until observed.

    def __len__(self):
        return self.N

    def initialize_seed(self, seed_indices: Union[List[int], np.ndarray]):
        self.set_fully_observed(seed_indices)
        self._init_precomputed_label_info()

    def set_fully_observed(self, observed_indices: Union[List[int], np.ndarray]):
        """Mark seed indices as fully observed (obs_lens = T) and all labels known."""
        for i in observed_indices:
            self.obs_lens[i] = self.T
            self.label_known[i, :] = True
        self._init_precomputed_label_info()

    def observe_next_step(self, indices: List[int]) -> List[bool]:
        """
        For each index:
          - reveal Y[i, L-1] (if in range and not already known)
          - if Y == 1: mark event_known/event_time and set censor_time = event_time
          - else: increment obs_lens (if possible) and update censor_time
        Returns list of booleans indicating whether an event (y==1) was revealed for each index.
        """
        if len(indices) == 0:
            return []

        device = self.y.device
        idx = torch.tensor(indices, dtype=torch.long, device=device)
        event_flags: List[bool] = []

        # Fast skip checks (vectorized)
        already_event = self.event_known[idx]  # bool[len(indices)]
        all_labels_known = self.label_known[idx].all(dim=1)  # bool[len(indices)]
        at_T = (self.obs_lens[idx] == self.T)  # bool[len(indices)]
        skip_mask = already_event | (at_T & all_labels_known)  # bool[len(indices)]

        # Loop only to perform the small in-place updates per sample.
        for k, i in enumerate(idx):
            if skip_mask[k]:
                # Nothing to reveal
                event_flags.append(False)
                continue

            L = int(self.obs_lens[i].item())
            pos = L - 1  # index of last currently observed label

            # Reveal label at pos (if pos in range)
            if 0 <= pos < self.T:
                # mark label_known
                self.label_known[i, pos] = True
                revealed_y = bool(int(self.y[i, pos].item()))
            else:
                # out of range (shouldn't normally happen)
                event_flags.append(False)
                continue

            if revealed_y:
                # Event observed at time pos (1-based pos+1)
                self.event_known[i] = True
                self.event_time[i] = pos + 1
                self.censor_time[i] = self.event_time[i]
                event_flags.append(True)
                # do NOT increment obs_lens (we stop revealing future X)
                continue
            else:
                # revealed 0 -> reveal the next X by incrementing obs_lens if possible
                if self.obs_lens[i] < self.T:
                    self.obs_lens[i] += 1
                # update censor_time to reflect new observed length (unless an event was already known)
                if not self.event_known[i]:
                    self.censor_time[i] = int(self.obs_lens[i].item())
                event_flags.append(False)
                continue

        return event_flags


    def _init_precomputed_label_info(self):
        N, T = self.y.shape
        # mask of revealed event positions
        positive_revealed = (self.y == 1) & self.label_known  # [N, T] bool
        any_event = positive_revealed.any(dim=1)  # [N] bool

        first_idx = positive_revealed.float().argmax(dim=1)  # for rows with no True, gives 0 (ignored)
        # event_time uses 1-based indexing per your original code; default = T (i.e. no event revealed)
        self.event_known = any_event.clone()  # bool[N]
        self.event_time = torch.full((N,), T, dtype=torch.long, device=self.y.device)
        self.event_time[any_event] = (first_idx[any_event] + 1).to(self.event_time.dtype)
        # censor_time = event_time if event_known else obs_lens
        self.censor_time = torch.where(self.event_known, self.event_time,
                                       self.obs_lens.clone().to(self.event_time.dtype))

    def __getitem__(self, idx):
        """
        Return:
           x_prefix: Tensor [L, D] (where L = obs_lens[idx])  <-- returns a view/slice
           label_obj: dict with keys:
               - 'is_event': bool
               - 'event_time': int
               - 'censor_time': int
           idx: original index
        All values taken from precomputed fields updated by observe_next_step / init.
        """
        # small, cheap operations only
        if torch.is_tensor(self.obs_lens):
            L = int(self.obs_lens[idx].item())
        elif isinstance(self.obs_lens, np.ndarray):
            L = int(self.obs_lens[idx].item())
        elif  isinstance(self.obs_lens, list):
            L = int(self.obs_lens[idx])
        else:
            raise Exception(f"cannot handle type {type(self.obs_lens)} of self.obs_lens")
        x_prefix = self.x[idx, ].clone()  # view; avoids clone and zeroing
        x_prefix[L: ] = 0

        is_event = bool(self.event_known[idx].item())
        event_time = int(self.event_time[idx].item())  # T if no revealed event (matches previous semantics)
        censor_time = int(self.censor_time[idx].item())

        label_obj = {
            'is_event': is_event,
            'event_time': event_time,
            'censor_time': censor_time,
            'obs_len': L
        }
        return x_prefix, label_obj, idx

