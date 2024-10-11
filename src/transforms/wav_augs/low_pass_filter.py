import torch
import torch_audiomentations
from torch import nn


class LowPassFilter(nn.Module):
    def __init__(self, prob, min_cutoff_freq, max_cutoff_freq, sample_rate):
        super().__init__()
        self.augmentation = torch_audiomentations.LowPassFilter(
            p=prob,
            min_cutoff_freq=min_cutoff_freq,
            max_cutoff_freq=max_cutoff_freq,
            sample_rate=sample_rate,
        )
        self.prob = prob

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < self.prob:
            return self.augmentation(spectrogram)

        return spectrogram
