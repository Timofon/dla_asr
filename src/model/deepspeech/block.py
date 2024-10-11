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
        x = x.permute(1, 2, 0)
        x = self.batch_norm(x)
        x = x.permute(2, 0, 1)

        packed_x = pack_padded_sequence(x, lengths.cpu(), enforce_sorted=False)
        packed_out, hidden_state = self.gru(packed_x, hidden_state)
        x, _ = pad_packed_sequence(packed_out)

        x = x.view(x.shape[0], x.shape[1], 2, -1).sum(2)

        return x, hidden_state
