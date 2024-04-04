# %%
# Imports
from datasets import load_dataset
import tiktoken
from tqdm import tqdm
import pandas as pd
from config import DATA_DIR

#%%
# Load the dataset
dataset = load_dataset("roneneldan/TinyStories", split="train")

# %%
def add_bos_eos_to_sequence(dataset, n=100):
    dataset_processed = [
        {"input": "<bos> " + sequence, "label": sequence + " <eos>"}
        for i, sequence in enumerate(dataset["text"][:n])
    ]
    return dataset_processed

# labelled_dataset = add_bos_eos_to_sequence(dataset, n=len(dataset))
labelled_dataset = add_bos_eos_to_sequence(dataset, n=100)

# %%
# Tokenize

# Initialize the tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

# Tokenize the corpus sequences
tokenized_sequences = [
    {
        'input': tokenizer.encode(row['input']),
        'label': tokenizer.encode(row['label'])
    } 
    for row in tqdm(labelled_dataset)
]

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
df.to_parquet(f'{DATA_DIR}data_tokenized.parquet', engine='pyarrow', index=False)
