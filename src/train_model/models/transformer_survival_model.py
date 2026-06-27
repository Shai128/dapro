import math

import torch
import torch.nn as nn


class DiscreteSurvivalLoss(nn.Module):
    def __init__(self, epsilon=1e-7, censored_mode='full_survival'):
        """
        Args:
            censored_mode:
                'full_survival' - Approach 1 (theoretically correct)
                'survival_class' - Approach 2 (easier to optimize)
                'hybrid' - Compromise (recommended)
        """
        super().__init__()
        self.epsilon = epsilon
        self.censored_mode = censored_mode

    def forward(self, probs, true_times, event_indicators):
        B, T_curr, T_future = probs.shape
        device = probs.device

        # Valid mask
        t_curr_grid = torch.arange(T_curr, device=device).unsqueeze(0)
        true_times_expanded = true_times.unsqueeze(1)
        valid_mask = t_curr_grid <= true_times_expanded

        # Target indices
        target_indices = true_times_expanded.clamp(max=T_future - 1).long()
        target_indices_exp = target_indices.expand(-1, T_curr)

        # UNCENSORED: P(T = true_time)
        prob_event = probs.gather(-1, target_indices_exp.unsqueeze(-1)).squeeze(-1)

        # CENSORED: Different approaches
        if self.censored_mode == 'survival_class':
            # Approach 2: Target the survival class directly
            # Simple and effective for optimization
            survival_class_idx = T_future - 1
            prob_survival = probs[:, :, survival_class_idx]  # (B, T_curr)

        elif self.censored_mode == 'full_survival':
            # Approach 1: Sum all probabilities > true_time
            # Theoretically correct but harder to optimize
            time_grid = torch.arange(T_future, device=device).reshape(1, 1, -1)
            survival_mask = time_grid >= target_indices.unsqueeze(-1)  # Strictly > TODO: notice this... >
            prob_survival = (probs * survival_mask.float()).sum(dim=-1)

        elif self.censored_mode == 'hybrid':
            # Hybrid: Weighted combination
            # Encourages survival class but doesn't ignore other future times

            # Get survival class probability
            survival_class_prob = probs[:, :, -1]

            # Get sum of probabilities > true_time (excluding survival class)
            time_grid = torch.arange(T_future - 1, device=device).reshape(1, 1, -1)
            future_mask = time_grid > target_indices.unsqueeze(-1)
            future_prob = (probs[:, :, :-1] * future_mask.float()).sum(dim=-1)

            # Weighted combination (emphasize survival class)
            # This gives 70% weight to survival class, 30% to other future times
            prob_survival = 0.7 * survival_class_prob + 0.3 * future_prob

        else:
            raise ValueError(f"Unknown censored_mode: {self.censored_mode}")

        # Combine based on event indicator
        event_indicators_exp = event_indicators.unsqueeze(1).expand(-1, T_curr)
        likelihood = torch.where(
            event_indicators_exp.bool(),
            prob_event,
            prob_survival
        )

        # NLL with stability
        loss = -torch.log(likelihood.clamp(min=self.epsilon))

        # Apply valid mask
        masked_loss = loss * valid_mask.float()
        total_loss = masked_loss.sum() / valid_mask.sum().clamp(min=1.0)

        return total_loss

# Transformer-based survival model
"""
        dropout=0.1,  # Default dropout
        max_time=100, 
        num_transformer_layers=3,  # Recommended: 2-4
        n_attention_heads=4,  # Recommended: 4-8
        projection_dim=None,  # Auto-determined
        dim_feedforward=None,  # FFN dimension in transformer
"""
class TransformerSurvivalModel(nn.Module):
    def __init__(self, input_size, max_time,
                 dropout=0.2,  # Default dropout
                 num_transformer_layers=3,  # Recommended: 2-4
                 n_attention_heads=6,  # Recommended: 4-8
                 projection_dim=None,  # Auto-determined
                 dim_feedforward=None,  #
                 ):
        super().__init__()
        # self.lstm = TCNEncoder(input_size, [32, 64, 32], dropout=dropout)

        if projection_dim is None:
            if input_size >= 1024:
                projection_dim = 256
            elif input_size >= 512:
                projection_dim = 128
            elif input_size >= 256:
                projection_dim = 64
            else:
                projection_dim = 64
        self.projection_dim = projection_dim
        projection_dim = self._make_divisible(projection_dim, n_attention_heads)
        self.projection_dim = projection_dim

        if input_size != projection_dim:
            # 1. Add Projection Layer
            self.projection = nn.Sequential(
                nn.Linear(input_size, projection_dim),
                nn.ReLU(),  # Optional: adds non-linearity to the compression
                nn.Dropout(dropout)
            )
        else:
            projection_dim = input_size
            self.projection = nn.Identity()
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_time, projection_dim))
        if dim_feedforward is None:
            dim_feedforward = projection_dim * 4

        self.temperature = nn.Parameter(torch.zeros(max_time+1))

        encoder_layer = nn.TransformerEncoderLayer(d_model=projection_dim, nhead=n_attention_heads,
                                                   dropout=dropout, batch_first=True,
                                                   dim_feedforward=dim_feedforward,
                                                   activation='gelu',  # GELU often works better than ReLU
                                                   norm_first=True  # Pre-norm architecture (more stable)
                                                   )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        # Hazard head: one weight per time step
        self.max_time = max_time

        self.head = nn.Sequential(
            nn.Linear(projection_dim * 2, projection_dim ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim , projection_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim // 2, max_time+1)
        )
        self.projection_dim = projection_dim

    def _make_divisible(self, dim, divisor):
        """Round dim to nearest number divisible by divisor."""
        remainder = dim % divisor
        if remainder == 0:
            return dim

        # Try rounding up first (better to have slightly larger model)
        rounded_up = dim + (divisor - remainder)
        # Alternatively, round down
        rounded_down = dim - remainder

        # Choose the closest one
        if abs(rounded_up - dim) <= abs(rounded_down - dim):
            return rounded_up
        else:
            return max(rounded_down, divisor)  # Ensure at least divisor size

    def _generate_causal_mask(self, sz, dz=None):
        if dz is None:
            dz = sz
        """Generates an upper-triangular mask to prevent attending to future tokens."""
        mask = (torch.triu(torch.ones(dz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        # x: (Batch, T, D)
        logits_masked = self.get_logits_for_calibration(x)
        # Softmax along the last dimension (t_future axis)
        # Result: P(Event at t_future | Survival up to t_curr)
        # Sums to 1 for valid futures.
        probs = torch.softmax(logits_masked, dim=-1)
        return probs

    @staticmethod
    def get_masked_probs(logits, log_temperature):

        # CRITICAL: Apply temperature BEFORE masking
        scaled_logits = logits / torch.exp(log_temperature).clamp(min=0.1, max=10.0)

        # NOW apply causal masking
        B, T_curr, T_future = scaled_logits.shape
        device = scaled_logits.device

        t_curr_grid = torch.arange(T_curr, device=device).unsqueeze(1)
        t_future_grid = torch.arange(T_future, device=device).unsqueeze(0)
        future_mask = t_future_grid >= t_curr_grid
        future_mask = future_mask.unsqueeze(0).expand(B, -1, -1)

        logits_masked = scaled_logits.masked_fill(~future_mask, float('-inf'))

        # Softmax
        probs = torch.softmax(logits_masked, dim=-1)
        return probs

    def calibrate(self, x_val, t_val, delta_val, lr=0.01, max_iter=50, device='cpu'):
        """
        Tunes a temperature parameter to minimize NLL on the validation set.
        """
        x_val = x_val.to(device)
        t_val = t_val.to(device)
        delta_val = delta_val.to(device)
        self.eval()  # Ensure model is in eval mode (dropout off)

        # 1. Initialize Temperature (Parameter to be optimized)
        # We use log_temp to ensure temperature is always positive (exp(log_temp))

        # Pre-compute logits to avoid running the full transformer every iteration
        # We only need to scale logits, not re-run the encoder
        with torch.no_grad():
            # Get logits BEFORE softmax
            # We need to modify forward() slightly to return logits if requested,
            # or just replicate the forward pass here without the final softmax.

            # Assuming you split forward into get_logits() and softmax:
            logits = self.get_logits_for_calibration(x_val, mask=False)

        log_temperature = nn.Parameter(torch.zeros(logits.shape[2]).to(x_val.device))
        optimizer = torch.optim.LBFGS([log_temperature], lr=lr, max_iter=max_iter)

        criterion = DiscreteSurvivalLoss(censored_mode='full_survival')  # Use consistent mode


        def closure():
            optimizer.zero_grad()

            probs = TransformerSurvivalModel.get_masked_probs(logits, log_temperature)

            # Compute loss
            loss = criterion(probs, t_val, delta_val)
            loss.backward()

            return loss

        optimizer.step(closure)
        # Save optimal temperature
        self.temperature = log_temperature #torch.exp(log_temperature).clamp(min=0.01, max=10.0)

        # Evaluate calibration improvement
        with torch.no_grad():
            # Before calibration
            probs_before = TransformerSurvivalModel.get_masked_probs(logits, torch.zeros_like(self.temperature))
            loss_before = criterion(probs_before, t_val, delta_val)

            # After calibration
            probs_after = TransformerSurvivalModel.get_masked_probs(logits, self.temperature)
            loss_after = criterion(probs_after, t_val, delta_val)

        print(f"Calibration complete:")
        print(f"  Optimal Temperature: {self.temperature}")
        print(f"  NLL before: {loss_before:.4f}")
        print(f"  NLL after:  {loss_after:.4f}")
        print(f"  Improvement: {loss_before - loss_after:.4f}")

    def get_logits_for_calibration(self, x, mask=True):
        """Helper to get raw logits (shared with forward)"""
        B, T, D = x.shape

        # 1. Embed and Add Position
        # Scale inputs by sqrt(d_model) is standard for Transformers
        x = self.projection(x) * math.sqrt(self.projection_dim)

        # Add positional encoding (broadcasting over batch)
        # Ensure x length matches max_time or slice pos_encoder if T < max_time
        x = x + self.pos_encoder[:, :T, :]

        # 2. Causal Masking
        # Mask shape: (T, T) - preventing position t from seeing t+1...
        src_mask = self._generate_causal_mask(T).to(x.device)

        # 3. Pass through Transformer
        # output: (Batch, T, d_model)
        h_encoded = self.transformer_encoder(x, mask=src_mask)
        h_concatenated = torch.cat([h_encoded, x], dim=-1)
        # 4. Prediction
        # logits: (Batch, T, T_max_time) -> Represents raw scores for t_future
        logits = self.head(h_concatenated)

        # 5. Masking Invalid Futures & Softmax
        # For a given t_curr (row index), we only want to predict t_future >= t_curr
        # We create a mask for the OUTPUT logits

        # Create a mask where (t_curr, t_future) is valid only if t_future >= t_curr
        # Shape (T, T)
        _, T_curr, T_future = logits.shape
        future_mask = torch.triu(torch.ones(T_curr, T_future, device=x.device))
        if mask:
            # Set invalid locations (past) to -inf so Softmax makes them 0
            logits_masked = logits.masked_fill(future_mask == 0, float('-inf'))
        else:
            logits_masked = logits

        return logits_masked

    def predict_proba(self, x):
        """Inference method applying the calibrated temperature"""
        with torch.no_grad():
            logits = self.get_logits_for_calibration(x, mask=False)
        probs = TransformerSurvivalModel.get_masked_probs(logits, self.temperature)
        return probs


import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


def plot_survival_calibration(probs, true_times, event_indicators, t_curr, t_interest, n_bins=10):
    """
    Checks calibration for a specific time horizon 't_interest'.
    We look at P(T <= t_interest).
    """
    # 1. Get predicted risk of event occurring by time t_interest
    # Sum probabilities from current t (0) up to t_interest
    # Assuming probs is (N, T_max) and we start at t=0
    risk_scores = probs[:, t_curr, t_curr:t_interest + 1].sum(dim=1).detach().cpu().numpy()

    # 2. Define binary target: Did event happen by t_interest?
    # Target is 1 if (true_time <= t_interest) AND (event_indicator == 1)
    # If censored before t_interest, we usually exclude them or use IPCW (simplified here)

    # Simple approach: Drop patients censored before t_interest
    valid_idx = (true_times >= t_interest) | (event_indicators == 1)

    y_true = (true_times[valid_idx] <= t_interest).float().cpu().numpy()
    y_prob = risk_scores[valid_idx]

    # 3. Calculate curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    # 4. Plot
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.xlabel(f'Predicted Probability of Event by t={t_interest}')
    plt.ylabel(f'Actual Fraction of Events')
    plt.title(f'Calibration Plot at t={t_interest}')
    plt.legend()
    plt.show()

