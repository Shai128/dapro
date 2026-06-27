import torch
from torch import nn



# LSTM-based survival model
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, hidden_layers, dropout, max_time, batch_size=128, project=True):
        super().__init__()
        # self.lstm = TCNEncoder(input_size, [32, 64, 32], dropout=dropout)

        if input_size >= 1024 and project:
            # 1. Add Projection Layer
            projection_dim = 256
            self.projection = nn.Sequential(
                nn.Linear(input_size, projection_dim),
                nn.ReLU(),  # Optional: adds non-linearity to the compression
                nn.Dropout(dropout)
            )
        else:
            projection_dim = input_size
            self.projection = nn.Identity()

        self.lstm = nn.LSTM(projection_dim, hidden_layers[0], 2, batch_first=True, bidirectional=False, dropout=dropout)

        self.max_time = max_time
        self.batch_size = batch_size
        layers = []
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

        self.hazard = nn.Linear(hidden_layers[-1], 1)

    def forward(self, x):
        # x: (batch, time_steps, features)
        batch_size = self.batch_size
        result = []
        for i in range(0, x.shape[0], batch_size):
            x_batch = x[i:min(i + batch_size, x.shape[0]), : self.max_time]
            # x_batch = x_batch.flatten(start_dim=1, end_dim=2)
            curr_batch_size = x_batch.shape[0]
            projected_x_batch = self.projection(x_batch)
            # times= torch.arange(0, projected_x_batch.shape[1], 1).unsqueeze(0).repeat(len(projected_x_batch), 1)
            # projected_x_batch = torch.cat([projected_x_batch, times], dim=-1)
            h_seq, _ = self.lstm(projected_x_batch)  # (batch, T, hidden)
            # h_flat = h_seq.contiguous().view(-1, h_seq.size(2))  # (batch*T, hidden)
            out = self.mlp(h_seq[:, -1])
            logits = self.hazard(out).view(curr_batch_size)  # (batch, T)
            # hazard probability per time step
            p = torch.sigmoid(logits)
            result.append(p)
        result = torch.cat(result, dim=0)
        return result

