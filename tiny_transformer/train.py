# %%
# Config
from config import (
    D_MODEL,
    NUM_HEADS,
    LEARNING_RATE,
    NUM_EPOCHS,
    PROJ_ROOT,
    BATCH_SIZE,
    NUM_DECODER_LAYERS,
)

# Imports
from transformer import TinyTransformerDecoder
from torch import nn, optim
from dataloader import *
from tqdm import tqdm
import matplotlib.pyplot as plt


# %%
# Dataset
# Dataloader

# Transformer

# %%
# Train
VOCAB_SIZE = 50283

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TinyTransformerDecoder(
    d_model=D_MODEL,
    nhead=NUM_HEADS,
    vocab_size=VOCAB_SIZE,
    max_seq_length=MAX_SEQ_LENGTH,
    ndecoder_layers=NUM_DECODER_LAYERS,
    device=device,
)

model.to(device)

# %%
loss_fn = torch.nn.CrossEntropyLoss()  # Assuming a classification problem
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_dataloader = make_dataloader(
    f"{DATA_DIR}/data_tokenized.parquet", batch_size=BATCH_SIZE
)

losses = []

# %%
# Training loop

# Initialize a list to store the average loss for each epoch
for epoch in range(NUM_EPOCHS):
    model.train()  # Set the model to training mode
    total_loss = 0

    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
        inputs = batch["input"]
        targets = batch["label"]
        assert inputs.shape == targets.shape
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        # feed logits, feed one-hot encoded targets
        # [outputs: batch size, seq length, vocab size]
        # [targets: batch size, seq length, vocab size]

        # One-hot encode the targets
        targets_one_hot = torch.zeros(targets.size(0), targets.size(1), VOCAB_SIZE).to(
            device
        )
        targets_one_hot.scatter_(2, targets.unsqueeze(2), 1)

        # Now pass the one-hot encoded targets to the loss function
        loss = loss_fn(outputs, targets_one_hot)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Compute average loss for the epoch and store it
    avg_loss = total_loss / len(train_dataloader)
    losses.append(avg_loss)

    # Print average loss for the epoch
    print(f"Epoch {epoch+1}, Loss: {avg_loss}")

# %%
# After the training loop, plot the losses
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.show()


# %% Save the model
torch.save(model.state_dict(), f"{PROJ_ROOT}/model_checkpoints/model_{NUM_EPOCHS}.pth")
