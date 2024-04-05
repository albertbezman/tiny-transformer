#%% Import necessary libraries
import torch
from transformer import TinyTransformerDecoder
from dataset import tokenizer
from config import D_MODEL, NUM_HEADS, VOCAB_SIZE, MAX_SEQ_LENGTH


#%% Define the function to generate a sequence
def generate_sequence(model, start_sequence, sequence_length):
    model.eval()  # Set the model to evaluation mode

    # Tokenize the starting sequence
    input_sequence = tokenizer.encode(start_sequence)

    # Convert the input sequence to a tensor and add a batch dimension
    input_sequence = torch.tensor(input_sequence).unsqueeze(0).to(device)

    # Initialize the generated sequence with the input sequence
    generated_sequence = input_sequence

    with torch.no_grad():  # No need to track gradients
        for _ in range(sequence_length):
            # Forward pass
            outputs = model(generated_sequence)

            # Get the predicted token by taking the argmax of the softmax output
            predicted_token = torch.argmax(outputs, dim=-1)[:, -1]

            # Add the predicted token to the generated sequence
            generated_sequence = torch.cat([generated_sequence, predicted_token.unsqueeze(0)], dim=-1)

    # Decode the generated sequence into text and return it
    return tokenizer.decode(generated_sequence[0].tolist())

#%% Load the pre-trained model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the pre-trained model
model = TinyTransformerDecoder(
    d_model=D_MODEL,
    nhead=NUM_HEADS,
    vocab_size=VOCAB_SIZE,
    max_seq_length=MAX_SEQ_LENGTH,
    device=device,
)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)

# Load the tokenizer
# tokenizer = tiktoken.Encoding.from_file("tokenizer.json")

#%% Generate a sequence
start_sequence = "<|bos|> Hello"
sequence_length = 20

output = generate_sequence(model, start_sequence, sequence_length)