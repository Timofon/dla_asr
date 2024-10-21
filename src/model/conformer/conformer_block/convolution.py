import torch
from torch import nn


class ConvolutionModule(nn.Module):
    def __init__(self, inner_dim) -> None:
        super().__init__()

        self.inner_dim = inner_dim
        self.expansion_factor = 2

        self.layer_norm = nn.LayerNorm(self.inner_dim)
        self.batch_norm = nn.BatchNorm1d(num_features=inner_dim)
        self.dropout = nn.Dropout(p=0.1)

        self.pointwise_conv1 = nn.Conv1d(
            in_channels=inner_dim,
            out_channels=inner_dim * self.expansion_factor,
            kernel_size=1,
        )
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=inner_dim,
            out_channels=inner_dim,
            kernel_size=1,
        )
        self.depthwise_conv = nn.Conv1d(
            in_channels=inner_dim,
            out_channels=inner_dim,
            kernel_size=32,
            groups=inner_dim,
            padding="same",
        )

        self.glu = nn.GLU(dim=1)
        self.swish = nn.SiLU()

    def forward(
        self, spectrogram: torch.Tensor
    ) -> torch.Tensor:  # [batch_size, spectrogram_length, n_feats]
        spectrogram = self.layer_norm(spectrogram.permute(0, 2, 1)).permute(0, 2, 1)

        spectrogram = self.pointwise_conv1(spectrogram)
        spectrogram = self.glu(spectrogram)
        spectrogram = self.depthwise_conv(spectrogram)
        spectrogram = self.batch_norm(spectrogram)
        spectrogram = self.swish(spectrogram)
        spectrogram = self.pointwise_conv2(spectrogram)

        spectrogram = self.dropout(spectrogram)

        return spectrogram
