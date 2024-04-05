# %%
# Imports
from datasets import load_dataset
import tiktoken
from tqdm import tqdm
import pandas as pd
from config import DATA_DIR, VOCAB_SIZE

# %%
# Load the dataset
dataset = load_dataset("roneneldan/TinyStories", split="train")


# %%
def add_bos_eos_to_sequence(dataset, n=100):
    dataset_processed = [
        {"input": "<|bos|> " + sequence, "label": sequence + "<|eos|>"}
        for i, sequence in enumerate(dataset["text"][:n])
    ]
    return dataset_processed


# labelled_dataset = add_bos_eos_to_sequence(dataset, n=len(dataset))
labelled_dataset = add_bos_eos_to_sequence(dataset, n=100)

# %%
# Tokenize

# Initialize the tokenizer
cl100k_base = tiktoken.get_encoding("cl100k_base")

tokenizer = tiktoken.Encoding(
    # If you're changing the set of special tokens, make sure to use a different name
    # It should be clear from the name what behaviour to expect.
    name="cl100k_tt",
    pat_str=cl100k_base._pat_str,
    mergeable_ranks=cl100k_base._mergeable_ranks,
    special_tokens={
        **cl100k_base._special_tokens,
        "<|bos|>": 100277,
        "<|eos|>": 100278,
    },
)

# Tokenize the corpus sequences
tokenized_sequences = [
    {
        "input": tokenizer.encode(row["input"], allowed_special={"<|bos|>", "<|eos|>"}),
        "label": tokenizer.encode(row["label"], allowed_special={"<|bos|>", "<|eos|>"}),
    }
    for row in tqdm(labelled_dataset)
]

VOCAB_SIZE = tokenizer.n_vocab

# test
# Print the tokenized sequences
# for sequence in tokenized_sequences[:10]:
#     print(sequence)

# tokenizer.decode(tokenized_sequences[0]['label'])

# %%
# Save the tokenized sequences to a parquet file
# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(tokenized_sequences)

# Save the DataFrame to a parquet file
df.to_parquet(f"{DATA_DIR}/data_tokenized.parquet", engine="pyarrow", index=False)
