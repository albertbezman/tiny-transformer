# import os
from pathlib import Path

# Paths
PROJ_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJ_ROOT / 'data'

# Hyperparameters
EPOCHS = 10
BATCH_SIZE = 32
D_MODEL = 512
NUM_HEADS = 8