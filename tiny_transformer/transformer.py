# %%
# Imports
import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import softmax
from config import NUM_DECODER_LAYERS
from attention import (
    create_look_ahead_mask,
    scaled_dot_product_attention,
    multi_head_attention,
)


# %%
# Transformer class
class TinyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        nhead,
        ndecoder_layers,
        dim_feedforward,
        # dropout,
        # activation,
        # max_seq_length,
    ):
        super(TinyTransformer, self).__init__()
        # Params
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.ndecoder_layers = ndecoder_layers
        self.dim_feedforward = dim_feedforward
        # self.dropout = dropout
        # self.activation = activation
        # self.max_seq_length = max_seq_length
        # Layers
        self.decoder_blocks = nn.Sequential(
            # Masked multi-head attention
            # Add
            # Norm
            # FFN
            # Add
            # Norm
        )
        self.linear = nn.Linear(d_model, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        for n in range(self.ndecoder_layers):
            x = self.decoder_blocks(x)
        x = self.linear(x)
        x = self.softmax(x)
        return x


# Embedding layer
# Pos encoding
# Masked multi-head attention block
# Add and norm
# FFN
# Add and norm
# Linear
# Softmax

# %%
# test
model = TinyTransformer(ndecoder_layers=NUM_DECODER_LAYERS)