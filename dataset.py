"""
Dataset utilities:
- generation of LR variants (sketch edges + degraded photo)
- padding/resizing preserving aspect ratio
- PyTorch Dataset class which can generate LR on-the-fly (and optionally save)
"""

import random
import os
from pathlib import Path
from typing import Tuple, Optional, Dict, List

import numpy as np
from PIL import Image, ImageFilter, ImageOps
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as torchvision_transforms

import config

# ---------- Helpers ----------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(config.SEED)

def resize_preserve_aspect_and_pad(pil_img: Image.Image, target_size: int, pad_mode: str = "reflect") -> Tuple[Image.Image, Dict]:
    """
    Resize the PIL image so the longer side == target_size while preserving aspect ratio.
    Then pad symmetrically to target_size x target_size.
    Returns (padded_image, meta_dict) where meta_dict contains original size and padding offsets.
    """
    w, h = pil_img.size
    scale = float(target_size) / max(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = pil_img.resize((new_w, new_h), resample=Image.BICUBIC)

    pad_w = target_size - new_w
    pad_h = target_size - new_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    if pad_mode == "reflect":
        # Convert to numpy array and reflect-pad. Ensure 3 channels.
        arr = np.array(resized)
        if arr.ndim == 2:  # grayscale -> make 3 channels
            arr = np.stack([arr] * 3, axis=-1)
        # pad: ((top, bottom), (left, right), (0,0))
        arr = np.pad(arr, ((top, bottom), (left, right), (0, 0)), mode='reflect')
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
        "scale": scale
    }
    return padded, meta

# ---------- Degradation & sketch generation ----------
def degrade_photo(pil_img: Image.Image, downscale: int = 4, sigma_range=(0.2, 2.0), jpeg_quality_range=(30, 95)) -> Image.Image:
    """
    Create a degraded (low-res, noisy, compressed) version of the input image.
    Steps:
     - Downscale by factor and upscale back (bicubic)
     - Add Gaussian blur
     - Add Gaussian noise
     - JPEG compress
    """
    # Downscale/upscale
    w, h = pil_img.size
    down_w, down_h = max(1, w // downscale), max(1, h // downscale)
    lr = pil_img.resize((down_w, down_h), resample=Image.BICUBIC)
    lr_up = lr.resize((w, h), resample=Image.BICUBIC)

    # Gaussian blur
    sigma = random.uniform(*sigma_range)
    pil_blur = lr_up.filter(ImageFilter.GaussianBlur(radius=sigma))

    # JPEG compression (via OpenCV)
    arr = np.array(pil_blur)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(*jpeg_quality_range)]
    result, encimg = cv2.imencode('.jpg', cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), encode_param)
    if result:
        arr2 = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
        arr2 = cv2.cvtColor(arr2, cv2.COLOR_BGR2RGB)
    else:
        arr2 = arr

    # Add Gaussian noise
    noise_std = random.uniform(0, 10)
    noisy = arr2.astype(np.float32) + np.random.normal(0, noise_std, arr2.shape).astype(np.float32)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)

def edges_to_sketch(pil_img: Image.Image, method: str = "canny", width_augment=(1,3)) -> Image.Image:
    """
    Generate an edge/sketch style image from a PIL image.
    Methods supported: 'canny', 'dog'
    Returns a 3-channel RGB image that looks like a sketch (edges over white background)
    """
    gray = np.array(pil_img.convert("L"))
    if method == "canny":
        # Canny edge detector; tweak thresholds
        v = np.median(gray)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edges = cv2.Canny(gray, lower, upper)
        # optionally dilate to simulate stroke width
        k = random.randint(*width_augment)
        if k > 1:
            kernel = np.ones((k, k), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
    elif method == "dog":
        # Difference of Gaussians then threshold
        g1 = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
        g2 = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0)
        dog = g1 - g2
        _, edges = cv2.threshold((dog + 128).astype(np.uint8), 120, 255, cv2.THRESH_BINARY_INV)
    else:
        raise ValueError("Unsupported method")
    # Convert to RGB sketch on white background (invert edges if needed)
    sketch = 255 - edges
    rgb = np.stack([sketch]*3, axis=-1)
    return Image.fromarray(rgb.astype(np.uint8))


class SwinIRDataset(Dataset):
    """
    Dataset that yields (LR_tensor, HR_tensor, meta) pairs.
    Can either read pre-generated LR images from LR_DIR or create LR variants on-the-fly from HR images.

    Behaviors added/clarified:
    - If `hr_dir` points to a dataset root that contains `HR/` and `LR/` subfolders, those will be used.
    - If `lr_dir` is provided it will be used (overriding automatic detection).
    - If `generate_on_fly` is False, the dataset expects paired LR files and will only keep HR files that have a matching LR by filename.
    - Optional `manifest_csv` support (simple CSV with columns pointing to HR/LR paths or a single filename column).
    """

    IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

    def __init__(self,
                 hr_dir: Path,
                 lr_dir: Optional[Path] = None,
                 manifest_csv: Optional[Path] = None,
                 target_resolution: int = config.TARGET_RESOLUTION,
                 mode: str = "train",
                 transforms=None,
                 generate_on_fly: bool = True,
                 sketch_prob: float = 0.7,
                 downscale_choices=(2, 4),
                 samples_per_image: int = 1):

        self.target_resolution = target_resolution
        self.generate_on_fly = generate_on_fly
        self.sketch_prob = sketch_prob
        self.downscale_choices = downscale_choices
        self.mode = mode
        self.samples_per_image = samples_per_image
        self.transforms = transforms

        # Resolve hr_dir and possible HR/LR subfolders
        self.hr_root = Path(hr_dir)
        if not self.hr_root.exists():
            raise AssertionError(f"HR dir {hr_dir} does not exist")

        # If user passed the root that contains HR/ LR, prefer those
        if (self.hr_root / "HR").exists() and (self.hr_root / "LR").exists():
            # If provided a root that contains HR and LR subfolders, use HR as hr_root
            root_parent = self.hr_root
            self.hr_root = root_parent / "HR"
            auto_lr_root = root_parent / "LR"
        else:
            auto_lr_root = None

        # If lr_dir explicitly provided, use it; else use auto detection
        if lr_dir is not None:
            self.lr_root = Path(lr_dir)
            if not self.lr_root.exists():
                raise AssertionError(f"LR dir {lr_dir} does not exist")
        else:
            self.lr_root = auto_lr_root

        # If manifest provided, try to read pairs from it
        self.paired_map: Dict[str, Path] = {}
        if manifest_csv is not None:
            try:
                import pandas as pd
                df = pd.read_csv(manifest_csv)
                cols_lower = [c.lower() for c in df.columns]
                if 'hr' in cols_lower and 'lr' in cols_lower:
                    hr_col = [c for c in df.columns if c.lower() == 'hr'][0]
                    lr_col = [c for c in df.columns if c.lower() == 'lr'][0]
                    for _, r in df.iterrows():
                        hrp = Path(r[hr_col])
                        lrp = Path(r[lr_col])
                        self.paired_map[hrp.name] = lrp
                elif 'filename' in cols_lower:
                    fname_col = [c for c in df.columns if c.lower() == 'filename'][0]
                    if self.lr_root is None:
                        raise AssertionError("manifest provides filenames but no lr_root was found/provided")
                    for _, r in df.iterrows():
                        nm = Path(r[fname_col]).name
                        self.paired_map[nm] = (self.lr_root / nm)
                else:
                    # fallback: try first two columns as hr, lr
                    if df.shape[1] >= 2:
                        for _, r in df.iterrows():
                            hrp = Path(r.iloc[0])
                            lrp = Path(r.iloc[1])
                            self.paired_map[hrp.name] = lrp
            except Exception:
                # if pandas unavailable or CSV malformed, ignore manifest (user can still rely on auto pairing)
                self.paired_map = {}

        # Collect HR files (only files, ignore directories)
        hr_candidates = sorted([p for p in self.hr_root.iterdir() if p.is_file() and p.suffix.lower() in self.IMG_EXTS])

        # If we have a LR root and not generating on the fly, pair by filename intersection
        if (self.lr_root is not None or len(self.paired_map) > 0) and not self.generate_on_fly:
            # Build lr lookup
            lr_lookup: Dict[str, Path] = {}
            if len(self.paired_map) > 0:
                # Use manifest pairs where available
                for nm, p in self.paired_map.items():
                    lr_lookup[nm] = Path(p)
            elif self.lr_root is not None:
                lr_files = sorted([p for p in self.lr_root.iterdir() if p.is_file() and p.suffix.lower() in self.IMG_EXTS])
                for p in lr_files:
                    lr_lookup[p.name] = p

            # Keep only HR files that have a matching LR file
            paired_hr_files: List[Path] = []
            paired_lr_map: Dict[str, Path] = {}
            for h in hr_candidates:
                if h.name in lr_lookup:
                    paired_hr_files.append(h)
                    paired_lr_map[h.name] = lr_lookup[h.name]

            if len(paired_hr_files) == 0:
                raise AssertionError("No paired HR-LR files found. Provide lr_dir or set generate_on_fly=True.")

            self.files = paired_hr_files
            self.paired = True
            self.paired_lr_map = paired_lr_map
        else:
            # Either generate on the fly, or LR optional -> use full HR list
            self.files = hr_candidates
            self.paired = False
            self.paired_lr_map = {}

        # quick checks
        if len(self.files) == 0:
            raise AssertionError("No HR images found in provided HR directory.")

        # to-tensor converter
        self.to_tensor = torchvision_transforms.ToTensor()

    def __len__(self):
        return len(self.files) * self.samples_per_image

    def _make_pair(self, hr_path: Path) -> Tuple[Image.Image, Image.Image, dict]:
        """
        Given an HR image path, produce (LR_pil, HR_pil, meta). HR is resized/padded to target resolution.
        This is the "generate_on_fly" behavior that degrades or sketches the HR image.
        """
        hr = Image.open(hr_path).convert("RGB")
        padded_hr, meta = resize_preserve_aspect_and_pad(hr, self.target_resolution, pad_mode=config.PAD_MODE)

        # Decide whether this LR should be a sketch or a degraded photo
        if random.random() < self.sketch_prob:
            # Sketch variant
            method = random.choice(["canny", "dog"])
            lr = edges_to_sketch(padded_hr, method=method)
            # optionally add slight blur / degrade to make lines imperfect
            if random.random() < 0.3:
                lr = lr.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.0, 1.0)))
        else:
            # Photographic degradation
            down = random.choice(self.downscale_choices)
            lr = degrade_photo(padded_hr, downscale=down)

        return lr, padded_hr, meta

    def __getitem__(self, idx):
        idx0 = idx // self.samples_per_image
        hr_path = self.files[idx0]

        if self.paired and not self.generate_on_fly:
            # Load HR and paired LR from disk, ensure both are padded/resized to target resolution
            hr = Image.open(hr_path).convert("RGB")
            hr_pil, meta = resize_preserve_aspect_and_pad(hr, self.target_resolution, pad_mode=config.PAD_MODE)

            lr_path = self.paired_lr_map[hr_path.name]
            lr = Image.open(lr_path).convert("RGB")
            lr_pil, _meta_lr = resize_preserve_aspect_and_pad(lr, self.target_resolution, pad_mode=config.PAD_MODE)
        else:
            # generate on the fly (degrade or sketch) from HR
            lr_pil, hr_pil, meta = self._make_pair(hr_path)

        # Convert to tensors (range 0..1)
        lr_t = self.to_tensor(lr_pil)
        hr_t = self.to_tensor(hr_pil)

        # Optionally apply patch cropping augmentation for training
        if self.mode == "train":
            _, H, W = hr_t.shape
            if H >= config.PATCH_SIZE and W >= config.PATCH_SIZE:
                top = random.randint(0, H - config.PATCH_SIZE)
                left = random.randint(0, W - config.PATCH_SIZE)
                hr_t = hr_t[:, top:top + config.PATCH_SIZE, left:left + config.PATCH_SIZE]
                lr_t = lr_t[:, top:top + config.PATCH_SIZE, left:left + config.PATCH_SIZE]
            # random flips/rot
            if random.random() < 0.5:
                hr_t = torch.flip(hr_t, dims=[2])
                lr_t = torch.flip(lr_t, dims=[2])
            if random.random() < 0.5:
                hr_t = torch.flip(hr_t, dims=[1])
                lr_t = torch.flip(lr_t, dims=[1])
        else:
            # For val/test, use full padded image (no crop)
            pass

        return lr_t, hr_t, meta
