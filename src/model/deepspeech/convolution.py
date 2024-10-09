import torch
from torch import nn


class ConvolutionModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            padding=(20, 5),
            kernel_size=(41, 11),
            stride=(2, 2),
        )
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.activation1 = nn.Hardtanh(0, 20)

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            padding=(10, 5),
            kernel_size=(21, 11),
            stride=(2, 1),
        )  # TODO
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.activation2 = nn.Hardtanh(0, 20)

        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=96,
            padding=(10, 5),
            kernel_size=(21, 11),
            stride=(2, 1),
        )
        self.batch_norm3 = nn.BatchNorm2d(96)
        self.activation3 = nn.Hardtanh(0, 20)

    def forward(self, spectrogram):
        spectrogram = self.activation1(self.batch_norm1(self.conv1(spectrogram)))
        spectrogram = self.activation2(self.batch_norm2(self.conv2(spectrogram)))
        spectrogram = self.activation3(self.batch_norm3(self.conv3(spectrogram)))

        return spectrogram
