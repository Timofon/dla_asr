import torch
from torch import nn

from .mhsa import MultiHeadSelfAttention, RelativePositionalEncoder


class MultiHeadSelfAttentionModule(nn.Module):
    def __init__(self, inner_dim, num_heads) -> None:
        super().__init__()

        self.layer_norm = nn.LayerNorm(inner_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.mhsa = MultiHeadSelfAttention(d_model=inner_dim, num_heads=num_heads)
        self.pos_encoding = RelativePositionalEncoder(inner_dim)

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        spectrogram = self.layer_norm(spectrogram)

        pos_embedding = self.pos_encoding(spectrogram)
        spectrogram = self.mhsa(spectrogram, pos_embedding)

        spectrogram = self.dropout(spectrogram)

        return spectrogram
