"""
Global configuration and paths.
Edit values here before running train/eval/inference.
"""

from pathlib import Path
import os

# -------------------------
# Paths and Drive settings
# -------------------------
# Replace DRIVE_ROOT with your Google Drive folder where models and logs will be saved.
DRIVE_ROOT = Path("/content/drive/MyDrive/swinir_project")  # mounted drive location in Colab
DATA_ROOT = Path("/content/HD/dataset")  # local data folder in Colab runtime (symlink from drive if preferred)

# Where to save checkpoints and logs (inside DRIVE_ROOT)
CHECKPOINT_DIR = DRIVE_ROOT / "checkpoints"
EXPORTED_DIR = DRIVE_ROOT / "exported"
LOG_DIR = DRIVE_ROOT / "logs"
SAMPLES_DIR = DRIVE_ROOT / "inference_outputs"

# Make sure dirs exist
for p in (DRIVE_ROOT, DATA_ROOT, CHECKPOINT_DIR, EXPORTED_DIR, LOG_DIR, SAMPLES_DIR):
    os.makedirs(p, exist_ok=True)

# -------------------------
# Data settings
# -------------------------
HR_DIR = DATA_ROOT / "HR"   # Put your high-resolution images here
LR_DIR = DATA_ROOT / "LR"   # Optional: place pre-generated LR images here if you want to use them
MANIFEST_CSV = DATA_ROOT / "manifest.csv"  # optional metadata (original size, padding offsets)

# Target training resolution and padding behavior
TARGET_RESOLUTION = 1024  # the square size we will pad/rescale to (1024x1024)
PAD_MODE = "reflect"      # 'reflect' helps avoid border artifacts

# Patch training (train on patches)
PATCH_SIZE = 256  # patch size to crop from padded images during training
PATCHES_PER_IMAGE = 8  # how many random patches to sample per HR image per epoch (augments dataset)

# -------------------------
# Model & training hyperparams
# -------------------------
# Device decision - training will use GPU if available on Colab.
USE_CUDA = True
DEVICE = "cuda" if (USE_CUDA and __import__("torch").cuda.is_available()) else "cpu"

# Training
NUM_EPOCHS = 60
BATCH_SIZE = 4
LR = 1e-4
WEIGHT_DECAY = 0.0
NUM_WORKERS = 4

# Loss weights
LOSS_L1_WEIGHT = 1.0
LOSS_PERCEPTUAL_WEIGHT = 0.01
LOSS_EDGE_WEIGHT = 0.5

# Checkpointing / resume
SAVE_EVERY_EPOCHS = 1
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL_NAME = "best_model.pth"
LAST_CHECKPOINT_NAME = "last_checkpoint.pth"

# Colab specifics: where to mount drive in notebook before training
# In Colab run:
# from google.colab import drive
# drive.mount('/content/drive')
# Then ensure DRIVE_ROOT exists.
COLAB_MOUNT_NOTE = "In Google Colab: run `from google.colab import drive; drive.mount('/content/drive')` and ensure DRIVE_ROOT path exists."

# -------------------------
# Inference
# -------------------------
TILE_SIZE = 512
TILE_OVERLAP = 32

# -------------------------
# Model architecture params (SwinIR-lite)
# -------------------------
IN_CHANS = 3
OUT_CHANS = 3
EMBED_DIM = 60      # smaller than paper for speed
DEPTHS = [2, 2, 2]  # number of Swin blocks per stage (lightweight)
NUM_HEADS = [3, 6, 12]
WINDOW_SIZE = 8
MLP_RATIO = 2.0
UPSCALE = 1  # We are doing restoration (same size). For SR set e.g. 2 or 4 and adjust heads.

# -------------------------
# Misc
# -------------------------
SEED = 42
