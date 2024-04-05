# import os
from pathlib import Path

# Paths
PROJ_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJ_ROOT / 'data'

# Hyperparameters
NUM_EPOCHS = 3
BATCH_SIZE = 8
D_MODEL = 128
NUM_HEADS = 8
NUM_DECODER_LAYERS = 2
VOCAB_SIZE = None
MAX_SEQ_LENGTH = None
LEARNING_RATE = 0.001
DATASET_SIZE=100