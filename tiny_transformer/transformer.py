# %%
# Imports
import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import softmax
from config import NUM_DECODER_LAYERS, VOCAB_SIZE, NUM_HEADS, D_MODEL, MAX_SEQ_LENGTH
from attention import multi_head_attention
from positional_encoding import positional_encoding


# %%
# Transformer class
class DecoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super(DecoderBlock, self).__init__()
        self.nhead = nhead
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            self.dropout,
            nn.Linear(dim_feedforward, d_model),
        )
        self.W_Q = nn.Parameter(torch.randn(d_model, d_model // nhead))
        self.W_K = nn.Parameter(torch.randn(d_model, d_model // nhead))
        self.W_V = nn.Parameter(torch.randn(d_model, d_model // nhead))
        self.W_O = nn.Parameter(torch.randn(d_model, d_model))

    def forward(self, x, padding_mask):
        # Save the pre-attention value of x
        x_p = x.clone()

        x = multi_head_attention(
            x, self.nhead, self.W_Q, self.W_K, self.W_V, self.W_O, padding_mask
        )

        # Add & Norm
        x = self.layer_norm1(x + x_p)
        x = self.dropout(x)

        # Save the pre-FFN value of x
        pre_ffn_x = x

        # Feed-forward network
        x = self.ffn(x)

        # Add & Norm after FFN
        x = self.layer_norm2(x + pre_ffn_x)
        x = self.dropout(x)

        return x


class TinyTransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        nhead,
        ndecoder_layers,
        device,
        max_seq_length,
        dropout=0.1,
        dim_feedforward=2048,
    ):
        super(TinyTransformerDecoder, self).__init__()
        # Params
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.device = device
        self.embeddings = nn.Embedding(self.vocab_size, self.d_model)
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(d_model, nhead, dim_feedforward, dropout)
                for _ in range(ndecoder_layers)
            ]
        )
        self.linear = nn.Linear(d_model, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # Create a padding mask for the input x
        padding_mask = (x == 0)

        x = self.embeddings(x)
        seq_len = x.size(1)
        pos_enc = positional_encoding(seq_len, self.d_model).to(self.device)
        x = x + pos_enc

        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, padding_mask)

        # Linear layer
        x = self.linear(x)

        # Softmax layer
        # x = self.softmax(x)
        return x


# Embedding layer
# Pos encoding
# Masked multi-head attention block
# Add and norm
# FFN
# Add and norm
# Linear
# Softmax
