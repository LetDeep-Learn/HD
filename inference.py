"""
Inference utilities:
- load a checkpoint
- single-image inference with padding preserved and removal
- tiled inference for large images
- save outputs to SAMPLES_DIR
- optional: export to ONNX for CPU acceleration (basic)
"""

import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

import config
from model import create_model
from dataset import resize_preserve_aspect_and_pad, set_seed

set_seed(config.SEED)

def load_model(path=None, device=torch.device("cpu")):
    model = create_model().to(device)
    if path is None:
        ckpt_path = config.CHECKPOINT_DIR / config.BEST_MODEL_NAME
        if not ckpt_path.exists():
            ckpt_path = config.CHECKPOINT_DIR / config.LAST_CHECKPOINT_NAME
        path = ckpt_path
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

def run_inference_on_image(model, pil_img: Image.Image, device, tile_size=config.TILE_SIZE, overlap=config.TILE_OVERLAP):
    """
    Perform inference while respecting padding and optionally tiling for large images.
    Steps:
        - resize/pad to target resolution (using dataset helper)
        - if tiled: split into overlapping tiles, infer, and blend
        - remove padding and rescale back to original size
    """
    padded, meta = resize_preserve_aspect_and_pad(pil_img, config.TARGET_RESOLUTION, pad_mode=config.PAD_MODE)
    # convert to tensor
    arr = np.array(padded).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).float().to(device)
    B,C,H,W = tensor.shape
    # If tile_size >= H, run directly
    if tile_size >= H:
        with torch.no_grad():
            out = model(tensor)
    else:
        # tile and blend
        stride = tile_size - overlap
        out_full = torch.zeros_like(tensor)
        weight = torch.zeros(1,1,H,W, device=device)
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                y0 = y
                x0 = x
                y1 = min(y0 + tile_size, H)
                x1 = min(x0 + tile_size, W)
                y0 = y1 - tile_size if y1 - tile_size >= 0 else 0
                x0 = x1 - tile_size if x1 - tile_size >= 0 else 0
                patch = tensor[..., y0:y1, x0:x1]
                with torch.no_grad():
                    out_patch = model(patch)
                out_full[..., y0:y1, x0:x1] += out_patch
                weight[..., 0, y0:y1, x0:x1] += 1.0
        out = out_full / weight

    # remove padding
    left = meta["pad_left"]
    right = meta["pad_right"]
    top = meta["pad_top"]
    bottom = meta["pad_bottom"]
    out_np = out.squeeze(0).permute(1,2,0).cpu().numpy()
    h_resized = meta["resized_h"]
    w_resized = meta["resized_w"]
    # crop according to padding offsets (we padded resized -> target)
    y0 = top
    y1 = top + h_resized
    x0 = left
    x1 = left + w_resized
    cropped = out_np[y0:y1, x0:x1, :]
    # rescale back to original size
    orig_h = meta["orig_h"]
    orig_w = meta["orig_w"]
    pil_out = Image.fromarray(np.clip((cropped * 255.0), 0, 255).astype(np.uint8))
    pil_out = pil_out.resize((orig_w, orig_h), resample=Image.BICUBIC)
    return pil_out

def infer_image_file(image_path: str, checkpoint_path: str = None, output_name: str = None, device_str: str = "cpu"):
    device = torch.device(device_str)
    model = load_model(checkpoint_path, device=device)
    pil = Image.open(image_path).convert("RGB")
    out = run_inference_on_image(model, pil, device=device)
    out_name = output_name or (Path(image_path).stem + "_restored.png")
    out_path = config.SAMPLES_DIR / out_name
    out.save(out_path)
    print(f"Saved restored image to {out_path}")
    return out_path

def export_onnx(checkpoint_path: str = None, onnx_path: str = None):
    """
    Export model to ONNX (simple: no dynamic shapes). Use ONNX Runtime for CPU inference.
    """
    device = torch.device("cpu")
    model = load_model(checkpoint_path, device=device)
    model.eval()
    dummy = torch.randn(1, config.IN_CHANS, config.TARGET_RESOLUTION, config.TARGET_RESOLUTION, device=device)
    onnx_path = Path(onnx_path or (config.EXPORTED_DIR / "swinir_export.onnx"))
    torch.onnx.export(model, dummy, str(onnx_path),
                      export_params=True, opset_version=12, do_constant_folding=True,
                      input_names=['input'], output_names=['output'])
    print(f"Exported ONNX model to {onnx_path}")
    return onnx_path

if __name__ == "__main__":
    # quick demo usage (edit paths)
    # Example (after mounting drive): python inference.py /content/data/test.jpg
    import sys
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path> [checkpoint_path] [device]")
        sys.exit(1)
    image_path = sys.argv[1]
    ckpt = sys.argv[2] if len(sys.argv) > 2 else None
    dev = sys.argv[3] if len(sys.argv) > 3 else "cpu"
    infer_image_file(image_path, checkpoint_path=ckpt, device_str=dev)
