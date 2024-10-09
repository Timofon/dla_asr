import torch
from torch import nn

from .block import DeepSpeechBlock
from .convolution import ConvolutionModule


class DeepSpeech2(nn.Module):
    def __init__(self, n_feats, rnn_layers, hidden_size, dropout, n_tokens):
        super().__init__()

        self.conv_module = ConvolutionModule()

        rnn_input_size = self._count_rnn_input_size(n_feats, 20, 41, 2)
        rnn_input_size = self._count_rnn_input_size(rnn_input_size, 10, 21, 2)
        rnn_input_size = self._count_rnn_input_size(rnn_input_size, 10, 21, 2)
        rnn_input_size *= 96

        self.deep_speech_blocks = nn.ModuleList(
            [
                DeepSpeechBlock(
                    input_size=rnn_input_size, hidden_size=hidden_size, dropout=dropout
                ),
                *[
                    DeepSpeechBlock(
                        input_size=hidden_size, hidden_size=hidden_size, dropout=dropout
                    )
                    for _ in range(rnn_layers - 1)
                ],
            ]
        )

        self.head = nn.Linear(in_features=hidden_size, out_features=n_tokens)

    def forward(self, spectrogram, spectrogram_length, **batch):
        x = spectrogram.unsqueeze(1)
        x = self.conv_module(x)

        N, C, F, T = x.shape
        x = x.reshape(N, C * F, T)

        x = x.permute(2, 0, 1).contiguous()

        h = None
        for block in self.deep_speech_blocks:
            x, h = block(x, h)

        T, N, H = x.shape
        x = x.reshape(T * N, H)
        x = self.head(x)
        output = x.reshape(T, N, -1).permute(1, 0, 2).contiguous()

        log_probs = nn.functional.log_softmax(output, dim=2)
        log_probs_length = self.transform_input_lengths(spectrogram_length)

        return {"log_probs": log_probs, "log_probs_length": log_probs_length}

    def _count_rnn_input_size(self, in_features, padding, kernel_size, stride):
        return (in_features + 2 * padding - kernel_size) // stride + 1

    def transform_input_lengths(self, input_lengths):
        t_dim = input_lengths.max()

        t_dim = (t_dim + 2 * 5 - 11) // 2 + 1
        t_dim = (t_dim + 2 * 5 - 11) // 2 + 1
        t_dim = (t_dim + 2 * 5 - 11) + 1

        return torch.zeros_like(input_lengths).fill_(t_dim)
