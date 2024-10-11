import torch
from torch import nn

from .block import DeepSpeechBlock
from .convolution import ConvolutionModule


class DeepSpeech2(nn.Module):
    def __init__(self, n_feats, rnn_layers, hidden_size, dropout, n_tokens):
        super().__init__()

        self.conv_module = ConvolutionModule()

        rnn_input_size = self.calculate_rnn_input_size(n_feats)

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
        spectrogram = spectrogram.unsqueeze(1)
        spectrogram, length = self.conv_module(spectrogram, spectrogram_length)

        batch_size, num_channels, n_feats, spec_len = spectrogram.shape
        spectrogram = spectrogram.reshape(batch_size, num_channels * n_feats, spec_len)

        spectrogram = spectrogram.permute(2, 0, 1).contiguous()

        for block in self.deep_speech_blocks:
            spectrogram, _ = block(spectrogram, length)

        spec_len, batch_size, hidden_size = spectrogram.shape
        spectrogram = self.head(spectrogram.view(spec_len * batch_size, hidden_size))
        output = (
            spectrogram.view(spec_len, batch_size, -1).permute(1, 0, 2).contiguous()
        )

        log_probs = nn.functional.log_softmax(output, dim=2)
        log_probs_length = length

        return {"log_probs": log_probs, "log_probs_length": log_probs_length}

    def calculate_rnn_input_size(self, n_feats):
        for conv in [
            self.conv_module.conv1,
            self.conv_module.conv2,
            self.conv_module.conv3,
        ]:
            n_feats = (
                n_feats
                + 2 * conv.padding[0]
                - conv.dilation[0] * (conv.kernel_size[0] - 1)
                - 1
            ) // conv.stride[0] + 1
        rnn_input_size = n_feats * self.conv_module.conv3.out_channels
        return rnn_input_size
