import torch
from torch import nn


class FeedForwardModule(nn.Module):
    def __init__(self, inner_dim) -> None:
        super().__init__()

        self.inner_dim = inner_dim
        self.expansion_factor = 4

        self.layer_norm = nn.LayerNorm(self.inner_dim)
        self.linear_expanding = nn.Linear(
            self.inner_dim, self.inner_dim * self.expansion_factor
        )
        self.linear_compressing = nn.Linear(
            self.inner_dim * self.expansion_factor, self.inner_dim
        )
        self.dropout = nn.Dropout(p=0.1)

        self.swish = nn.SiLU()

    def forward(
        self, spectrogram: torch.Tensor
    ) -> torch.Tensor:  # [batch_size, inner_dim, spectrogram_length
        spectrogram = self.layer_norm(
            spectrogram
        )  # in layer norm [batch_size, spectrogram_length, inner_dim]

        spectrogram = self.linear_expanding(spectrogram)
        spectrogram = self.swish(spectrogram)
        spectrogram = self.dropout(spectrogram)
        spectrogram = self.linear_compressing(spectrogram)

        spectrogram = self.dropout(spectrogram)

        return spectrogram
