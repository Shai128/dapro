import numpy as np
import torch
from torch import nn



# LSTM-based survival model
class LSTMSurvivalModel(nn.Module):
    def __init__(self, input_size, hidden_layers, dropout, max_time, batch_size=128):
        super().__init__()
        # self.lstm = TCNEncoder(input_size, [32, 64, 32], dropout=dropout)

        if input_size >= 1024:
            # 1. Add Projection Layer
            projection_dim = 256
            self.projection = nn.Sequential(
                nn.Linear(input_size, projection_dim),
                nn.ReLU(),  # Optional: adds non-linearity to the compression
                nn.Dropout(dropout)
            )
        else:
            projection_dim = input_size
            self.projection = None
        self.lstm = nn.LSTM(projection_dim, hidden_layers[0], 2, batch_first=True, bidirectional=False, dropout=dropout)
        # Hazard head: one weight per time step
        self.max_time = max_time
        self.batch_size = batch_size

        layers = []
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

        self.hazard = nn.Linear(hidden_layers[-1], 1)

    # def forward(self, x):
    #     # x: (batch, time_steps, features)
    #     B, T, _ = x.shape
    #     h_seq = self.lstm(x)       # → (B, T + extra, C)
    #     h_seq = h_seq[:, :T, :]   # crop back to 500
    #     flat  = h_seq.reshape(-1, h_seq.size(2))
    #     logits = self.hazard(flat).view(B, T)
    #     return torch.sigmoid(logits)
    def forward(self, x):
        # x: (batch, time_steps, features)
        batch_size = self.batch_size
        result = []
        for i in range(0, x.shape[0], batch_size):
            x_batch = x[i:min(i + batch_size, x.shape[0])]
            curr_batch_size, T, _ = x_batch.shape
            if self.projection is not None:
                projected_x_batch = self.projection(x_batch)
            else:
                projected_x_batch = x_batch
            h_seq, _ = self.lstm(projected_x_batch)  # (batch, T, hidden)
            h_flat = h_seq.contiguous().view(-1, h_seq.size(2))  # (batch*T, hidden)
            out = self.mlp(h_flat)
            logits = self.hazard(out).view(curr_batch_size, T)  # (batch, T)
            # hazard probability per time step
            p = torch.sigmoid(logits)
            result.append(p)
        result = torch.cat(result, dim=0)
        return result


# LSTM-based survival model
class LSTMMeanModel(nn.Module):
    def __init__(self, input_size, hidden_layers, dropout, max_time, batch_size=128):
        super().__init__()
        # self.lstm = TCNEncoder(input_size, [32, 64, 32], dropout=dropout)

        if input_size >= 1024:
            # 1. Add Projection Layer
            projection_dim = 256
            self.projection = nn.Sequential(
                nn.Linear(input_size, projection_dim),
                nn.ReLU(),  # Optional: adds non-linearity to the compression
                nn.Dropout(dropout)
            )
        else:
            projection_dim = input_size
            self.projection = None
        self.lstm = nn.LSTM(projection_dim, hidden_layers[0], 2, batch_first=True, bidirectional=False, dropout=dropout)
        # Hazard head: one weight per time step
        self.max_time = max_time

        layers = []
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

        self.hazard = nn.Linear(hidden_layers[-1], 1)
        self.batch_size = batch_size

    def forward(self, x):
        # x: (batch, time_steps, features)
        batch_size = self.batch_size
        result = []
        for i in range(0, x.shape[0], batch_size):
            x_batch = x[i:min(i + batch_size, x.shape[0])]
            curr_batch_size, T, _ = x_batch.shape
            if self.projection is not None:
                projected_x_batch = self.projection(x_batch)
            else:
                projected_x_batch = x_batch
            h_seq, _ = self.lstm(projected_x_batch)  # (batch, T, hidden)
            h_flat = h_seq.contiguous().view(-1, h_seq.size(2))  # (batch*T, hidden)
            out = self.mlp(h_flat)
            preds = self.hazard(out).view(curr_batch_size, T)  # (batch, T)
            result.append(preds)
        result = torch.cat(result, dim=0)
        return result


class ConstantHazardBaseline:
    def __init__(self, hazard_rate, max_time):
        self.hazard = hazard_rate
        self.max_time = max_time

    def predict_survival(self, N):
        # returns an (N, max_time) array of S(t) = (1-hazard)^t
        times = np.arange(1, self.max_time + 1)
        S = np.power(1 - self.hazard, times)
        return np.repeat(S[np.newaxis, :], N, axis=0)



# Training routine
# def train_model(x_train, y_train, t_train, hidden_layers=[32, 32],
#                 batch_size=64, lr=5e-4, epochs=20, val_split=0.2,
#                 save_path=None, device='cpu', dropout=0.1):
#     # Prepare dataset and loaders
#     dataset = SurvivalDataset(x_train, t_train)
#     N = len(dataset)
#     n_val = int(val_split * N)
#     n_train = N - n_val
#     train_ds, val_ds = random_split(dataset, [n_train, n_val])
#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_ds, batch_size=batch_size)
#
#     # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using {device} device.")
#     model = LSTMSurvivalModel(input_size=x_train.shape[2],
#                               hidden_layers=hidden_layers,
#                               max_time=x_train.shape[1], dropout=dropout).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     history = {'train_loss': [], 'val_loss': []}
#     best_model = None
#     best_loss = None
#     for epoch in range(1, epochs + 1):
#         model.train()
#         train_loss = 0.0
#         for x_batch, t_batch, delta_batch in train_loader:
#             x_batch = x_batch.to(device)
#             t_batch = t_batch.to(device)
#             delta_batch = delta_batch.to(device)
#             optimizer.zero_grad()
#             p = model(x_batch)
#             loss = discrete_survival_nll(p, t_batch, delta_batch)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item() * x_batch.size(0)
#         train_loss /= n_train
#
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for x_batch, t_batch, delta_batch in val_loader:
#                 x_batch = x_batch.to(device)
#                 t_batch = t_batch.to(device)
#                 delta_batch = delta_batch.to(device)
#                 p = model(x_batch)
#                 loss = discrete_survival_nll(p, t_batch, delta_batch)
#                 val_loss += loss.item() * x_batch.size(0)
#         val_loss /= n_val
#         if best_loss is None or best_model is None or val_loss < best_loss:
#             best_model = copy.deepcopy(model)
#             best_loss = val_loss
#
#         history['train_loss'].append(train_loss)
#         history['val_loss'].append(val_loss)
#         print(f'Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}')
#
#     if save_path is not None:
#         torch.save(best_model.cpu().state_dict(), save_path)
#         print(f"Model state_dict saved to {save_path}")
#
#     return best_model, history
#
#
# def evaluate(model_or_baseline, x_test, t_test, delta_test, horizons=[5, 10, 20, 50, 100, 200, 300, 400, 500]):
#     N, T, feats = x_test.shape
#     # get survival matrix
#     if isinstance(model_or_baseline, torch.nn.Module):
#         device = next(model_or_baseline.parameters()).device
#         x_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
#         p = model_or_baseline(x_tensor).detach()
#         S = torch.cumprod(1 - p, dim=1).detach().cpu().numpy()
#     else:
#         S = model_or_baseline.predict_survival(N)
#
#     # median time estimate
#     median_time = (S <= 0.5).argmax(axis=1) + 1
#     # concordance
#
#     c_idx = None
#     # brier scores
#     brier_scores = {}
#     for h in horizons:
#         idx = min(h, T) - 1
#         pred_prob = 1 - S[:, idx]
#         obs = (t_test <= h).astype(float)
#         brier_scores[h] = np.mean((obs - pred_prob) ** 2)
#     # nll
#     if isinstance(model_or_baseline, torch.nn.Module):
#         with torch.no_grad():
#             t_tensor = torch.tensor(t_test, dtype=torch.long).to(device)
#             delta_tensor = torch.tensor(delta_test, dtype=torch.long).to(device)
#             test_loss = discrete_survival_nll(p, t_tensor, delta_tensor).item()
#     else:
#         # baseline nll
#         eps = 1e-8
#         h = model_or_baseline.hazard
#         p_flat = np.full((N, T), h)
#         one_minus = 1 - p_flat
#         cumprod = np.cumprod(one_minus, axis=1)
#         S_tm1 = np.ones_like(p_flat)
#         S_tm1[:, 1:] = cumprod[:, :-1]
#         f = S_tm1 * p_flat
#         p_event = f[np.arange(N), t_test - 1] + eps
#         p_censor = cumprod[:, -1] + eps
#         ll = delta_test * np.log(p_event) + (1 - delta_test) * np.log(p_censor)
#         test_loss = -np.mean(ll)
#     # Report
#     print("--- Evaluation Results ---")
#     if c_idx is not None:
#         print(f"Concordance Index: {c_idx:.4f}")
#     print(f"Test NLL: {test_loss:.4f}")
#     for h, b in brier_scores.items():
#         print(f"Brier Score @ {h}: {b:.4f}")
#     return {'c_index': c_idx, 'nll': test_loss, 'brier': brier_scores}
