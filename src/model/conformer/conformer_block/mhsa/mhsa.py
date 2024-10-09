import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelativePositionalEncoder(nn.Module):
    def __init__(self, model_dim, max_len=5000):
        super().__init__()
        self.model_dim = model_dim

        positional_encodings = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, model_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / model_dim)
        )

        positional_encodings[:, 0::2] = torch.sin(position * div_term)
        positional_encodings[:, 1::2] = torch.cos(position * div_term)

        positional_encodings = positional_encodings.unsqueeze(0)
        self.register_buffer("positional_encodings", positional_encodings)

    def forward(self, x):
        return x + self.positional_encodings[:, : x.size(1)]


# https://github.com/kimiyoung/transformer-xl/tree/master - official impl. of relative attention in Transformer-XL
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout_p=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_head = d_model // num_heads
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(d_model)

        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout_p)

        self.u = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))

        self.layer_norm = nn.LayerNorm(d_model)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, pos_embedding):
        batch_size = x.size(0)

        query = self.Q(x).view(batch_size, -1, self.num_heads, self.d_head)
        key = (
            self.K(x)
            .view(batch_size, -1, self.num_heads, self.d_head)
            .permute(0, 2, 1, 3)
        )
        value = (
            self.V(x)
            .view(batch_size, -1, self.num_heads, self.d_head)
            .permute(0, 2, 1, 3)
        )
        pos_embedding = self.proj(pos_embedding).view(
            batch_size, -1, self.num_heads, self.d_head
        )

        content_score = torch.matmul(
            (query + self.u).transpose(1, 2), key.transpose(2, 3)
        )
        pos_score = torch.matmul(
            (query + self.v).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1)
        )
        pos_score = self._relative_positional_encoding(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim

        attn = F.softmax(score, -1)
        result = torch.matmul(attn, value).transpose(1, 2)
        result = result.contiguous().view(batch_size, -1, self.d_model)

        x = self.out_proj(result)
        return x

    def _relative_positional_encoding(self, pos_score):
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(
            batch_size, num_heads, seq_length2 + 1, seq_length1
        )
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)

        return pos_score
