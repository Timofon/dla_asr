import torch
from torch import nn

from .conformer_block.conformer_block import ConformerBlock
from .subsampling import SubsamplingModule


class Conformer(nn.Module):
    def __init__(
        self,
        n_feats,
        inner_dim,
        mhsa_n_heads,
        n_blocks,
        decoder_dim,
        decoder_layers,
        n_tokens,
    ) -> None:
        super().__init__()

        self.subsampling = SubsamplingModule(n_channels=inner_dim)
        self.linear_expanding = nn.Linear(
            in_features=inner_dim * (n_feats // 4), out_features=inner_dim
        )

        self.dropout = nn.Dropout(p=0.1)
        self.conformer_blocks = nn.ModuleList(
            [
                ConformerBlock(inner_dim=inner_dim, mhsa_n_heads=mhsa_n_heads)
                for _ in range(n_blocks)
            ]
        )
        self.lstm = nn.LSTM(
            input_size=inner_dim,
            hidden_size=decoder_dim,
            num_layers=decoder_layers,
            dropout=0.1,
            batch_first=True,
        )
        self.head = nn.Linear(in_features=decoder_dim, out_features=n_tokens)

    def forward(
        self, spectrogram, spectrogram_length, **batch
    ) -> torch.Tensor:  # [batch_size, inner_dim, spectrogram_length]
        spectrogram = self.subsampling(spectrogram)

        spectrogram = self.linear_expanding(spectrogram)
        spectrogram = self.dropout(spectrogram)

        for block in self.conformer_blocks:
            spectrogram = block(spectrogram)

        output = self.head(self.lstm(spectrogram)[0])

        log_probs = nn.functional.log_softmax(output, dim=-1)
        log_probs_length = self.transform_input_lengths(spectrogram_length)
        return {"log_probs": log_probs, "log_probs_length": log_probs_length}

    def transform_input_lengths(self, input_lengths):
        """
        As the network may compress the Time dimension, we need to know
        what are the new temporal lengths after compression.

        Args:
            input_lengths (Tensor): old input lengths
        Returns:
            output_lengths (Tensor): new temporal lengths
        """
        return input_lengths // 4

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
