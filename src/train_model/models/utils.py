import abc
import dataclasses

import torch
from matplotlib import pyplot as plt
from torch import nn


@dataclasses.dataclass
class ModelPrediction(abc.ABC):
    pass


@dataclasses.dataclass
class SurvivalModelPrediction(ModelPrediction):
    quantile_est: torch.Tensor
    probability_est: torch.Tensor


def survival_loss(input, t_tilde, delta):
    """Custom survival_loss function.

    Args:
        input ([torch.tensor]): model prediction for probability of survival event at any given time
        meta ([tuple]): tuple of tensors including t_tilde, the censored time and delta, the event indicator

    Returns:
        [torch.tensor]: model loss, negative log likelihood
    """

    # Computr the loss using binary cross entropy
    prop_success = delta / t_tilde
    import torch.nn.functional as F
    return F.binary_cross_entropy_with_logits(input[:, 1], prop_success, weight=t_tilde)


# Negative log-likelihood loss for discrete-time survival
def discrete_survival_nll(p, t, delta):
    # p: (batch, T) hazard probabilities
    # t: (batch,) event/censoring times in [1, T]
    # delta: (batch,) event indicators
    # Compute survival S(t-1) and density
    delta = delta.float()
    eps = 1e-8
    batch, T = p.shape
    # survival up to t-1: S(t-1) = prod_{s< t}(1-p_s)
    one_minus_p = 1 - p
    cumprod = torch.cumprod(one_minus_p, dim=1)
    # S at t-1: shift cumprod right, S(0)=1
    S_tm1 = torch.ones_like(p)
    S_tm1[:, 1:] = cumprod[:, :-1]
    # discrete density f(t) = S(t-1)*p(t)
    f = S_tm1 * p
    # Gather likelihood for each sample
    # index for time-steps is zero-based
    idx = (t - 1).unsqueeze(1)
    # probability of observed event or censoring
    p_event = f.gather(1, idx).squeeze() + eps
    p_censor = cumprod[:, -1] + eps
    # log-likelihood
    ll = delta * torch.log(p_event) + (1 - delta) * torch.log(p_censor)
    return -ll.mean()


def binary_survival_loss(p, t, delta):
    # p: (Batch, T)
    # t: (Batch,)
    # delta: (Batch,)

    # Create a mask of valid times (Where step j <= t_i)
    batch_size, T = p.shape
    device = p.device

    # Create time grid [1, 2, ..., T]
    time_grid = torch.arange(1, T + 1, device=device).unsqueeze(0)  # (1, T)

    # Mask: 1 if this time step matters for this patient, 0 otherwise
    mask = (time_grid <= t.unsqueeze(1)).float()

    # Targets:
    # Everything is 0 (survived) EXCEPT the exact time of event, if delta=1
    target = torch.zeros_like(p)

    # Set the target to 1 only at the index of the event
    # We use scatter_ to put a 1 at the location t-1 for rows where delta=1
    event_indices = (t - 1).long().unsqueeze(1)
    target.scatter_(1, event_indices, delta.unsqueeze(1).float())

    # Binary Cross Entropy
    # We only care about the loss at valid time steps (masked)
    bce_loss = nn.functional.binary_cross_entropy(p, target, reduction='none')

    masked_loss = bce_loss * mask

    # Normalize by number of valid steps or batch size?
    # Usually sum over time, mean over batch.
    # TODO: maybe return this loss: return (masked_loss.sum(dim=1) / t).mean()
    return masked_loss.sum(dim=1).mean()


def batch_binary_survival_loss(p, t, delta):
    """
    p: (Batch, T, T)
       - Dim 1 (T): The time at which prediction is made (t1)
       - Dim 2 (T): The time being predicted (t2)
       - Values: Conditional Hazard P(T=t2 | T>=t2) via Sigmoid
    t: (Batch,) - Time of event/censoring
    delta: (Batch,) - Event indicator (1=event, 0=censored)
    """
    batch_size, T_seq, T_pred = p.shape
    device = p.device

    # --- 1. PREPARE GRIDS ---
    # Time steps when the prediction is made (t1)
    # Shape: (1, T, 1) to broadcast against Batch and Pred-Time
    seq_grid = torch.arange(1, T_seq + 1, device=device).view(1, -1, 1)

    # Time steps being predicted (t2)
    # Shape: (1, 1, T)
    pred_grid = torch.arange(1, T_pred + 1, device=device).view(1, 1, -1)

    # Expand t and delta for broadcasting
    t_bc = t.view(-1, 1, 1)  # (Batch, 1, 1)
    delta_bc = delta.view(-1, 1, 1)  # (Batch, 1, 1)

    # --- 2. CREATE MASKS ---

    # Mask A: Valid Sequence Steps (Stop training after patient is gone)
    # We only compute loss for LSTM steps t1 <= true_time
    seq_mask = (seq_grid <= t_bc).float()

    # Mask B: Valid Prediction Horizon (Standard Survival Mask)
    # We only compute loss for prediction times t2 <= true_time
    # (We don't know what happens after censoring)
    horizon_mask = (pred_grid <= t_bc).float()

    # Mask C: Future Only (Dynamic Prediction)
    # We only compute loss for predictions where t2 > t1
    future_mask = (pred_grid > seq_grid).float()

    # COMBINE MASKS
    # Loss is valid if: Patient is present AND Target is observed AND Prediction is in future
    full_mask = seq_mask * horizon_mask * future_mask

    # --- 3. CREATE TARGETS ---
    # Standard Logistic Hazard Target:
    # 0 everywhere, 1 at the specific index of event (if uncensored)

    # Get indices where event occurred
    # We want to place a 1 at index (i, :, t_event-1) IF delta=1
    # We only care about t_event, but we broadcast it across the sequence dimension (dim 1)
    event_idx = (t - 1).long().view(-1, 1, 1)

    # Create a scatter mask for the event location
    # This creates a tensor of 1s at the event time t2, for all t1
    event_target = torch.zeros_like(p)
    event_target.scatter_(2, event_idx.expand(-1, T_seq, 1), 1.0)

    # Apply delta: The target is 1 only if it was an actual event (delta=1)
    target = event_target * delta_bc

    # --- 4. COMPUTE LOSS ---
    bce_loss = nn.functional.binary_cross_entropy(p, target, reduction='none')

    masked_loss = bce_loss * full_mask

    # Normalize by the number of valid entries to keep loss scale consistent
    # Avoid division by zero with epsilon
    return masked_loss.sum() / (full_mask.sum() + 1e-7)

def binary_classification_loss(p, t, delta):

    bce_loss = nn.functional.binary_cross_entropy(p, delta.float(), reduction='mean')

    return bce_loss


def mse_loss(mean_prediction, t, delta):
    # p: (Batch, T)
    # t: (Batch,)
    # delta: (Batch,)

    # We only care about the loss at valid time steps (masked)

    batch_size, T = mean_prediction.shape
    device = mean_prediction.device

    # Create time grid [1, 2, ..., T]
    time_grid = torch.arange(1, T + 1, device=device).unsqueeze(0)  # (1, T)

    # Mask: 1 if this time step matters for this patient, 0 otherwise
    mask = (time_grid <= t.unsqueeze(1)).float()
    mask = mask * delta.unsqueeze(-1).repeat(1, mask.shape[1])

    mse_loss = (mean_prediction - t.unsqueeze(-1)) ** 2

    mse_loss = (mse_loss[mask.bool()]).mean(dim=-1)

    return mse_loss.mean()


def discrete_survival_nll(p, t, delta):
    """
    Negative Log Likelihood for discrete-time survival.
    Used by: Dynamic-DeepHit (often combined with a ranking loss).

    Args:
        p: (Batch, T) - Predicted hazard probabilities h(t)
        t: (Batch,)   - Index of event/censoring time (1-based, 1..T)
        delta: (Batch,) - Indicator: 1 if event occurred, 0 if censored
    """
    # 1. Clip probabilities to avoid log(0)
    eps = 1e-7
    p = torch.clamp(p, eps, 1.0 - eps)

    # 2. Compute Survival S(t) = cumprod(1 - p)
    # S(t-1) is the probability of surviving up to step t-1
    one_minus_p = 1.0 - p
    S = torch.cumprod(one_minus_p, dim=1)

    # 3. Handle Indices
    # t is 1-based (time 1..T). We need 0-based indices for gathering.
    idx = (t - 1).long().unsqueeze(1)  # shape (batch, 1)

    # 4. Compute Likelihood components
    # We need S(t-1). We pad S with 1.0 at the beginning (time 0).
    batch_size = p.shape[0]
    S_padded = torch.cat([torch.ones(batch_size, 1, device=p.device), S], dim=1)

    # S_prev = S(t-1)
    S_prev = S_padded.gather(1, idx).squeeze()

    # h_t = p(t) at the event time
    p_at_t = p.gather(1, idx).squeeze()

    # S_curr = S(t) (Used for censoring)
    S_curr = S.gather(1, idx).squeeze()

    # 5. Select Likelihood based on Delta
    # If Event (delta=1): Likelihood is f(t) = S(t-1) * h(t)
    L_uncensored = S_prev * p_at_t

    # If Censored (delta=0): Likelihood is S(t) (survived at least until t)
    L_censored = S_curr

    # Combine: L = L_uncensored if delta=1 else L_censored
    likelihood = delta * L_uncensored + (1 - delta) * L_censored

    # 6. Negative Log
    nll = -torch.log(likelihood + eps)

    return nll.mean()


# Plot training history
def plot_history(history, save_path):
    plt.figure()
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
