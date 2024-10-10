import torch
from torch import nn


class SubsamplingModule(nn.Module):
    def __init__(self, n_channels) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=n_channels, kernel_size=2, stride=2
        )
        self.conv2 = nn.Conv2d(
            in_channels=n_channels, out_channels=n_channels, kernel_size=2, stride=2
        )

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        spectrogram = self.conv1(spectrogram.unsqueeze(1))
        spectrogram = self.relu1(spectrogram)
        spectrogram = self.conv2(spectrogram)
        spectrogram = self.relu2(spectrogram)

        batch_size, n_channels, new_n_feats, new_seq_len = spectrogram.shape
        spectrogram = spectrogram.permute(0, 3, 1, 2)
        spectrogram = spectrogram.reshape(batch_size, new_seq_len, -1)

        return spectrogram


# class SubsamplingModule(nn.Module):
#     def __init__(self, out_channels):
#         super().__init__()
#         self.subsampling = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=3, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2),
#             nn.ReLU()
#         )

#     def forward(self, x):
#         # x - [bs, freq, time]
#         x = self.subsampling(x.permute(0, 2, 1).unsqueeze(1))  # [bs, out_channels, new_time, new_freq]
#         x = x.permute(0, 2, 1, 3)  # [bs, new_time, out_channels, new_freq]
#         return x.contiguous().view(x.size(0), x.size(1), -1)  # [bs, new_time, out_channels * new_freq]
