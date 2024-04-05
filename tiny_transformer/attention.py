# %%
import numpy as np
from torch.nn.functional import softmax
import torch

# def softmax(x):
#     """Compute softmax values for each sets of scores in x."""
#     e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
#     return e_x / e_x.sum(axis=-1, keepdims=True)


# %%
def create_look_ahead_mask(size):
    """
    Create a mask to hide future tokens.
    :param size: the size of the mask (the sequence length)
    :return: a 2D mask with shape (size, size) with the upper triangle filled with ones
    """
    mask = torch.triu(torch.ones((size, size)), diagonal=1)
    return mask


def scaled_dot_product_attention(queries, keys, values, mask):
    """
    Calculate the attention weights and output
    :param queries: a matrix of query vectors
    :param keys: a matrix of key vectors
    :param values: a matrix of value vectors
    :param mask: a masking array that prevents attention to certain positions
    :return: the weighted sum of value vectors and attention weights
    """
    # pdb.set_trace()  # Set breakpoint at the first line of the function

    # Dot product of queries with keys (transposed), scaled by size of the key vectors
    # This represents the score matrix - which is a way to measure the association of keys and queries
    attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(
        torch.tensor(keys.shape[-1], device=queries.device).float()
    )

    # Apply the mask to the scores (if any)
    if mask is not None:
        mask = mask.to(queries.device)
        attention_scores += mask * -1e9  # Use a large negative number to mask

    # Calculate the attention weights using softmax
    attention_weights = softmax(attention_scores, dim=-1)

    # Multiply the attention weights with the value vectors to get the output
    # Assuming values is a torch tensor
    output = torch.matmul(attention_weights, values)

    return output, attention_weights


# %%

# Example usage
# sequence_length = 5
# embedding_size = 64  # Example embedding size
# queries = np.random.rand(5, embedding_size)
# keys = np.random.rand(5, embedding_size)
# values = np.random.rand(5, embedding_size)
# mask = create_look_ahead_mask(sequence_length)


# # Call the function - this will trigger the debugger
# output, attention_weights = scaled_dot_product_attention(queries, keys, values, mask)


# %%
def multi_head_attention(x, num_heads, wq, wk, wv, wo, return_attention_weights=False):
    # Split the embedding dimension across the heads
    queries = keys = values = x
    seq_length = x.shape[1]
    all_outputs = []
    all_attention_weights = []

    for h in range(num_heads):
        W_Q_h = wq
        W_K_h = wk
        W_V_h = wv

        # Linearly project the queries, keys, and values for head 'h'
        queries_h = torch.matmul(queries, W_Q_h)
        keys_h = torch.matmul(keys, W_K_h)
        values_h = torch.matmul(values, W_V_h)

        # Calculate the attention output for head 'h'
        output_h, attention_weights_h = scaled_dot_product_attention(
            queries_h, keys_h, values_h, mask=create_look_ahead_mask(seq_length)
        )

        # Collect the results from each head
        all_outputs.append(output_h)
        all_attention_weights.append(attention_weights_h)

    # Concatenate the outputs from each head along the last dimension
    concatenated_outputs = torch.cat(all_outputs, dim=-1)

    # Apply a final linear projection if needed
    W_O = wo
    output = torch.matmul(concatenated_outputs, W_O)

    if return_attention_weights:
        return output, all_attention_weights
    else:
        return output



# Example usage
# num_heads = 8  # Let's assume we have 8 attention heads
# output, all_attention_weights = multi_head_attention(queries, keys, values, num_heads)
