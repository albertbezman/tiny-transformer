# %%
# Imports
import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import softmax
from config import NUM_DECODER_LAYERS, VOCAB_SIZE, NUM_HEADS, D_MODEL, MAX_SEQ_LENGTH
from attention import multi_head_attention
from positional_encoding import create_positional_encodings


# %%
# Transformer class
class TinyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        nhead,
        # ndecoder_layers,
        # dim_feedforward,
        # dropout,
        # activation,
        max_seq_length,
    ):
        super(TinyTransformer, self).__init__()
        # Params
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        # self.ndecoder_layers = ndecoder_layers
        # self.dim_feedforward = dim_feedforward
        # self.dropout = dropout
        # self.activation = activation
        # self.max_seq_length = max_seq_length
        self.embeddings = nn.Embedding(self.vocab_size, self.d_model)
        self.positional_encodings = create_positional_encodings
        self.multi_head_attention = multi_head_attention

        # Define the weight matrices as parameters
        self.W_Q = nn.Parameter(torch.randn(self.d_model, self.head_dim))
        self.W_K = nn.Parameter(torch.randn(self.d_model, self.head_dim))
        self.W_V = nn.Parameter(torch.randn(self.d_model, self.head_dim))
        self.W_O = nn.Parameter(torch.randn(self.d_model, self.d_model))

        # Layers
        # self.decoder_blocks = nn.Sequential(
        #     # Add
        #     # Norm
        #     # FFN
        #     # Add
        #     # Norm
        # )
        # self.linear = nn.Linear(d_model, vocab_size)
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.embeddings(x)
        # x = self.positional_encodings(x, self.max_seq_length, self.d_model)
        x = self.multi_head_attention(
            x, self.nhead, self.W_Q, self.W_K, self.W_V, self.W_O
        )
        # for n in range(self.ndecoder_layers):
        #     x = self.decoder_blocks(x)
        # x = self.linear(x)
        # x = self.softmax(x)
        print(x)
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
MAX_SEQ_LENGTH = 278
VOCAB_SIZE = 100279

model = TinyTransformer(
    d_model=D_MODEL,
    nhead=NUM_HEADS,
    vocab_size=VOCAB_SIZE,
    max_seq_length=MAX_SEQ_LENGTH,
)

from dataloader import *

train_dataloader = make_dataloader(f"{DATA_DIR}/data_tokenized.parquet", batch_size=10)

for i, s in enumerate(train_dataloader):
    output = model(s["input"])