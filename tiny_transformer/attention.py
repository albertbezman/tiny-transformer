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
    mask = torch.tensor(np.triu(np.ones((size, size)), k=1))
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
    attention_scores = torch.tensor(
        np.matmul(queries, keys.transpose()) / np.sqrt(keys.shape[-1])
    )

    # Apply the mask to the scores (if any)
    if mask is not None:
        attention_scores += mask * -1e9  # Use a large negative number to mask

    # Calculate the attention weights using softmax
    # This represents
    attention_weights = softmax(attention_scores, dim=-1)

    # Multiply the attention weights with the value vectors to get the output
    output = np.matmul(attention_weights, values)

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
def multi_head_attention(queries, keys, values, num_heads, embedding_size):
    # Split the embedding dimension across the heads
    head_dim = queries.shape[-1] // num_heads
    all_outputs = []
    all_attention_weights = []

    for h in range(num_heads):
        # Assuming W_Q, W_K, W_V are weight matrices for each head
        # They are initialized and learned during training
        # For this example, they are just placeholders to show the process
        W_Q_h = np.random.rand(embedding_size, head_dim)
        W_K_h = np.random.rand(embedding_size, head_dim)
        W_V_h = np.random.rand(embedding_size, head_dim)

        # Linearly project the queries, keys, and values for head 'h'
        queries_h = np.dot(queries, W_Q_h)
        keys_h = np.dot(keys, W_K_h)
        values_h = np.dot(values, W_V_h)

        # Calculate the attention output for head 'h'
        output_h, attention_weights_h = scaled_dot_product_attention(
            queries_h, keys_h, values_h, mask=create_look_ahead_mask(queries_h.shape[1])
        )

        # Collect the results from each head
        all_outputs.append(output_h)
        all_attention_weights.append(attention_weights_h)

    # Concatenate the outputs from each head along the last dimension
    concatenated_outputs = np.concatenate(all_outputs, axis=-1)

    # Apply a final linear projection if needed
    W_O = np.random.rand(num_heads * head_dim, embedding_size)
    output = np.dot(concatenated_outputs, W_O)

    return output, all_attention_weights


# Example usage
# num_heads = 8  # Let's assume we have 8 attention heads
# output, all_attention_weights = multi_head_attention(queries, keys, values, num_heads)