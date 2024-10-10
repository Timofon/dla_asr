import torch
from torch import nn


class DeepSpeechBlock(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
            bidirectional=True,
            batch_first=False,
        )
        self.batch_norm = nn.BatchNorm1d(num_features=hidden_size)

    def forward(self, x, hidden_state=None):
        x, hidden_state = self.gru(x, hidden_state)

        # for bidirectionality
        x = x.reshape(x.shape[0], x.shape[1], 2, -1).sum(2)

        t_dim, n_dim = x.shape[0], x.shape[1]
        x = x.reshape(t_dim * n_dim, -1)
        x = self.batch_norm(x)
        x = x.reshape(t_dim, n_dim, -1).contiguous()
        return x, hidden_state
