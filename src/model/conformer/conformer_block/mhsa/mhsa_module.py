import torch
from torch import nn

from .mhsa import MultiHeadSelfAttention


class MultiHeadSelfAttentionModule(nn.Module):
    def __init__(self, inner_dim, num_heads) -> None:
        super().__init__()

        self.layer_norm = nn.LayerNorm(inner_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.mhsa = MultiHeadSelfAttention(embed_dim=inner_dim, num_heads=num_heads, dropout=0.1)

    def forward(self, spectrogram: torch.Tensor, spectrogram_length: torch.Tensor) -> torch.Tensor:
        batch_size, max_len = spectrogram.size(0), spectrogram.size(2)
        mask = (torch.arange(max_len, device=spectrogram.device).expand(batch_size, max_len) >= spectrogram_length.unsqueeze(1)).to(torch.bool)
        attention_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        spectrogram = self.layer_norm(spectrogram.permute(0, 2, 1))

        spectrogram, _ = self.mhsa(query=spectrogram, key=spectrogram, value=spectrogram, mask=attention_mask)

        spectrogram = self.dropout(spectrogram).permute(0, 2, 1)

        return spectrogram
