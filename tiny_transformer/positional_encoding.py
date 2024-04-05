# %%
import torch
import math


def positional_encoding(seq_len, d_model):
    """
    Compute positional encodings for input sequences.

    Args:
    seq_len: The sequence length.
    d_model: The dimension of the model.

    Returns:
    A tensor of shape (seq_len, d_model) containing the positional encodings.
    """
    # Initialize a matrix (seq_len, d_model) where each row corresponds to a position in the sequence
    pos = torch.arange(seq_len).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
    )

    # Compute the positional encodings
    pos_enc = torch.zeros((seq_len, d_model))
    pos_enc[:, 0::2] = torch.sin(pos * div_term)
    pos_enc[:, 1::2] = torch.cos(pos * div_term)

    return pos_enc
