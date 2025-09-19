import random
from pathlib import Path
from typing import Dict, Tuple

from PIL import Image, ImageOps
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as torchvision_transforms

import config


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(config.SEED)


def resize_preserve_aspect_and_pad(
    pil_img: Image.Image, target_size: int, pad_mode: str = "reflect"
) -> Tuple[Image.Image, Dict]:
    """Resize shorter side to target_size, pad to square."""

    w, h = pil_img.size
    scale = float(target_size) / max(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = pil_img.resize((new_w, new_h), resample=Image.BICUBIC)

    pad_w, pad_h = target_size - new_w, target_size - new_h
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)

    if pad_mode == "reflect":
        arr = np.array(resized)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        arr = np.pad(arr, ((top, bottom), (left, right), (0, 0)), mode="reflect")
        padded = Image.fromarray(arr)
    else:
        padded = ImageOps.expand(resized, border=(left, top, right, bottom), fill=0)

    meta = {
        "orig_w": w,
        "orig_h": h,
        "resized_w": new_w,
        "resized_h": new_h,
        "pad_left": left,
        "pad_right": right,
        "pad_top": top,
        "pad_bottom": bottom,
        "scale": scale,
    }
    return padded, meta


class SwinIRDataset(Dataset):
    IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

    def __init__(
        self,
        hr_dir: Path,
        lr_dir: Path,
        target_resolution: int = config.TARGET_RESOLUTION,
        mode: str = "train",
        samples_per_image: int = 1,
    ):
        self.target_resolution = target_resolution
        self.mode = mode
        self.samples_per_image = samples_per_image

        self.hr_root = Path(hr_dir)
        self.lr_root = Path(lr_dir)

        if not self.hr_root.exists() or not self.lr_root.exists():
            raise AssertionError("HR or LR directory does not exist")

        self.hr_files = sorted(
            [p for p in self.hr_root.iterdir() if p.suffix.lower() in self.IMG_EXTS]
        )
        self.lr_files = sorted(
            [p for p in self.lr_root.iterdir() if p.suffix.lower() in self.IMG_EXTS]
        )

        if len(self.hr_files) != len(self.lr_files):
            raise AssertionError("Mismatch between HR and LR file counts")

        self.to_tensor = torchvision_transforms.ToTensor()

    def __len__(self):
        return len(self.hr_files) * self.samples_per_image

    def __getitem__(self, idx):
        idx0 = idx // self.samples_per_image
        hr_path, lr_path = self.hr_files[idx0], self.lr_files[idx0]

        hr = Image.open(hr_path).convert("RGB")
        lr = Image.open(lr_path).convert("RGB")

        hr_pil, meta = resize_preserve_aspect_and_pad(
            hr, self.target_resolution, pad_mode=config.PAD_MODE
        )
        lr_pil, _ = resize_preserve_aspect_and_pad(
            lr, self.target_resolution, pad_mode=config.PAD_MODE
        )

        hr_t = self.to_tensor(hr_pil)
        lr_t = self.to_tensor(lr_pil)

        if self.mode == "train":
            _, H, W = hr_t.shape
            if H >= config.PATCH_SIZE and W >= config.PATCH_SIZE:
                top = random.randint(0, H - config.PATCH_SIZE)
                left = random.randint(0, W - config.PATCH_SIZE)
                hr_t = hr_t[:, top : top + config.PATCH_SIZE, left : left + config.PATCH_SIZE]
                lr_t = lr_t[:, top : top + config.PATCH_SIZE, left : left + config.PATCH_SIZE]
            if random.random() < 0.5:
                hr_t = torch.flip(hr_t, dims=[2])
                lr_t = torch.flip(lr_t, dims=[2])
            if random.random() < 0.5:
                hr_t = torch.flip(hr_t, dims=[1])
                lr_t = torch.flip(lr_t, dims=[1])

        return lr_t, hr_t, meta
