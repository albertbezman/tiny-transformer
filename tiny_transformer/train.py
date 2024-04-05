# %%
# Imports
from transformer import TinyTransformerDecoder
from torch import nn, optim
from dataloader import *
from tqdm import tqdm
import matplotlib.pyplot as plt

# Config
from config import D_MODEL, NUM_HEADS, LEARNING_RATE, NUM_EPOCHS, PROJ_ROOT, BATCH_SIZE

# %%
# Dataset
# Dataloader
# Transformer

# %%
# Train
MAX_SEQ_LENGTH = 278
VOCAB_SIZE = 100279

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TinyTransformerDecoder(
    d_model=D_MODEL,
    nhead=NUM_HEADS,
    vocab_size=VOCAB_SIZE,
    max_seq_length=MAX_SEQ_LENGTH,
    device=device,
)

model.to(device)

# %%
loss_fn = torch.nn.CrossEntropyLoss()  # Assuming a classification problem
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_dataloader = make_dataloader(f"{DATA_DIR}/data_tokenized.parquet", batch_size=BATCH_SIZE)

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
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = loss_fn(outputs.view(-1, VOCAB_SIZE), targets.view(-1))

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


#%% Save the model
torch.save(model.state_dict(), f"{PROJ_ROOT}/model_checkpoints/model_{NUM_EPOCHS}.pth")