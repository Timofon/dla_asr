import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DeepSpeechBlock(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()

        self.batch_norm = nn.BatchNorm1d(num_features=input_size)

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
            bidirectional=True,
            batch_first=False,
        )

    def forward(self, x, lengths, hidden_state=None):
        spec_len, batch_size, hidden_dim = x.shape

        x = x.reshape(-1, hidden_dim)
        x = self.batch_norm(x)
        x = x.reshape(spec_len, batch_size, hidden_dim)

        packed_x = pack_padded_sequence(x, lengths.cpu(), enforce_sorted=False)
        packed_out, hidden_state = self.gru(packed_x, hidden_state)
        x, _ = pad_packed_sequence(packed_out)

        # for bidirectionality
        x = x.view(x.shape[0], x.shape[1], 2, -1).sum(2)

        return x, hidden_state
