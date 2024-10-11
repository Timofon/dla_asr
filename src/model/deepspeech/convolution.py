import torch
from torch import Tensor, nn


class ConvolutionModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=(41, 11),
            stride=(2, 2),
            padding=(20, 5),
        )
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.activation1 = nn.Hardtanh(0, 20)

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(21, 11),
            stride=(2, 1),
            padding=(10, 5),
        )
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.activation2 = nn.Hardtanh(0, 20)

        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=96,
            kernel_size=(21, 11),
            stride=(2, 1),
            padding=(10, 5),
        )
        self.batch_norm3 = nn.BatchNorm2d(96)
        self.activation3 = nn.Hardtanh(0, 20)

    def forward(self, spectrogram: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        spectrogram = self.conv1(spectrogram)
        lengths = self.update_lengths(lengths, self.conv1)
        spectrogram = self.batch_norm1(spectrogram)
        spectrogram = self.activation1(spectrogram)
        spectrogram = self.apply_mask(spectrogram, lengths)

        spectrogram = self.conv2(spectrogram)
        lengths = self.update_lengths(lengths, self.conv2)
        spectrogram = self.batch_norm2(spectrogram)
        spectrogram = self.activation2(spectrogram)
        spectrogram = self.apply_mask(spectrogram, lengths)

        spectrogram = self.conv3(spectrogram)
        lengths = self.update_lengths(lengths, self.conv3)
        spectrogram = self.batch_norm3(spectrogram)
        spectrogram = self.activation3(spectrogram)
        spectrogram = self.apply_mask(spectrogram, lengths)

        return spectrogram, lengths

    def update_lengths(self, lengths: Tensor, conv: nn.Conv2d) -> Tensor:
        kernel_size = conv.kernel_size[1]
        stride = conv.stride[1]
        padding = conv.padding[1]
        dilation = conv.dilation[1]

        return (lengths + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    def apply_mask(self, x: Tensor, lengths: Tensor) -> Tensor:
        batch_size, channels, height, max_len = x.size()

        mask = torch.arange(max_len, device=x.device).expand(
            batch_size, max_len
        ) >= lengths.cuda().unsqueeze(1)
        mask = mask[:, None, None, :]

        return x.masked_fill(mask, 0)
