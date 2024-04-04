# %%
# Imports
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from config import DATA_DIR
from torch.nn.utils.rnn import pad_sequence


# %%
# Make a Pytorch dataset class
class TinyStoriesDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_parquet(file_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "input": torch.tensor(self.data["input"][idx], dtype=torch.long),
            "label": torch.tensor(self.data["label"][idx], dtype=torch.long),
        }


# %%
# Test load the dataset
train_dataset = TinyStoriesDataset(f"{DATA_DIR}/data_tokenized.parquet")


# %%
# Dataloader
def collate_fn(batch):
    inputs = [item["input"] for item in batch]
    labels = [item["label"] for item in batch]

    # Pad sequences
    inputs = pad_sequence(inputs, batch_first=True)
    labels = pad_sequence(labels, batch_first=True)

    return {"input": inputs, "label": labels}


# Dataloader
def make_dataloader(file_path, batch_size=32):
    dataset = TinyStoriesDataset(file_path)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    return dataloader


# %%
# Test the dataloader
# train_dataloader = make_dataloader(f'{DATA_DIR}/data_tokenized.parquet')

# for i, s in enumerate(train_dataloader):
#     print(i)
#     print(s)
#     break
