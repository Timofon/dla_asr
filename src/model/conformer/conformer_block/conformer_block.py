import torch
from torch import nn

from .convolution import ConvolutionModule
from .feed_forward import FeedForwardModule
from .mhsa.mhsa_module import MultiHeadSelfAttentionModule


class ConformerBlock(nn.Module):
    def __init__(self, inner_dim, mhsa_n_heads) -> None:
        super().__init__()

        self.feed_forward1 = FeedForwardModule(inner_dim=inner_dim)
        self.feed_forward2 = FeedForwardModule(inner_dim=inner_dim)
        self.mhsa = MultiHeadSelfAttentionModule(
            inner_dim=inner_dim, num_heads=mhsa_n_heads
        )
        self.conv = ConvolutionModule(inner_dim=inner_dim)

        self.layer_norm = nn.LayerNorm(inner_dim)

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        spectrogram = 0.5 * self.feed_forward1(spectrogram) + spectrogram
        spectrogram = self.mhsa(spectrogram) + spectrogram
        spectrogram = self.conv(spectrogram) + spectrogram
        spectrogram = self.layer_norm(
            spectrogram + 0.5 * self.feed_forward2(spectrogram)
        )

        return spectrogram
