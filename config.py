from pathlib import Path
import os, torch

DRIVE_ROOT = Path("/content/drive/MyDrive/swinir_project")
DATA_ROOT = Path("/content/HD/dataset")

CHECKPOINT_DIR = DRIVE_ROOT / "checkpoints"
EXPORTED_DIR = DRIVE_ROOT / "exported"
LOG_DIR = DRIVE_ROOT / "logs"
SAMPLES_DIR = DRIVE_ROOT / "inference_outputs"
for p in (DRIVE_ROOT, DATA_ROOT, CHECKPOINT_DIR, EXPORTED_DIR, LOG_DIR, SAMPLES_DIR):
    os.makedirs(p, exist_ok=True)

HR_DIR = DATA_ROOT / "HR"
LR_DIR = DATA_ROOT / "LR"
MANIFEST_CSV = DATA_ROOT / "manifest.csv"

TARGET_RESOLUTION = 1024
PAD_MODE = "reflect"
PATCH_SIZE = 192        # ↓ from 256 to save memory
PATCHES_PER_IMAGE = 6   # fewer patches per HR image

USE_CUDA = True
DEVICE = "cuda" if (USE_CUDA and torch.cuda.is_available()) else "cpu"

NUM_EPOCHS = 5
BATCH_SIZE = 2          # ↓ from 4 to save memory
LR = 1e-4
WEIGHT_DECAY = 0.0
NUM_WORKERS = 2

LOSS_L1_WEIGHT = 1.0
LOSS_PERCEPTUAL_WEIGHT = 0.01
LOSS_EDGE_WEIGHT = 0.5

SAVE_EVERY_EPOCHS = 1
BEST_MODEL_NAME = "best_model.pth"
LAST_CHECKPOINT_NAME = "last_checkpoint.pth"

TILE_SIZE = 512
TILE_OVERLAP = 32

IN_CHANS = 3
OUT_CHANS = 3
EMBED_DIM = 192        # ↓ much lighter
DEPTHS = [4, 4, 4]     # ↓ reduce depth per stage
NUM_HEADS = [4, 4, 4]  # divides 192 cleanly (192/4 = 48 per head)
WINDOW_SIZE = 8
MLP_RATIO = 2.0
UPSCALE = 1

SEED = 42
