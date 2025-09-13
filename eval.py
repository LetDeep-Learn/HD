"""
Evaluation script that computes PSNR and SSIM on a dataset.
Loads model from CHECKPOINT_DIR/last or best.
"""

import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

import config
from dataset import SwinIRDataset
from model import create_model

def load_model_checkpoint(model, path=None):
    if path is None:
        path = config.CHECKPOINT_DIR / config.BEST_MODEL_NAME
        if not path.exists():
            path = config.CHECKPOINT_DIR / config.LAST_CHECKPOINT_NAME
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt['model_state'])
    return model

def to_numpy(img_tensor):
    # tensor BxCxHxW, values [0,1]
    img = img_tensor.squeeze(0).permute(1,2,0).cpu().numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img

def compute_metrics(pred, gt):
    # pred and gt are HxWxC uint8
    pred = pred.astype(np.float32) / 255.0
    gt = gt.astype(np.float32) / 255.0
    mse = np.mean((pred - gt) ** 2)
    psnr = 20 * np.log10(1.0 / np.sqrt(mse + 1e-10))
    s = 0.0
    # compute mean SSIM over channels by converting to gray
    if pred.shape[2] == 3:
        pred_gray = np.dot(pred[..., :3], [0.2989, 0.5870, 0.1140])
        gt_gray = np.dot(gt[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        pred_gray = pred[..., 0]
        gt_gray = gt[..., 0]
    s = ssim(pred_gray, gt_gray, data_range=1.0)
    return psnr, s

def evaluate():
    device = torch.device(config.DEVICE)
    model = create_model().to(device)
    model = load_model_checkpoint(model)
    model.eval()

    val_ds = SwinIRDataset(hr_dir=config.HR_DIR, mode="val", generate_on_fly=True, samples_per_image=1)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=2)
    psnrs = []
    ssims = []

    with torch.no_grad():
        for lr, hr, meta in tqdm(val_loader):
            lr = lr.to(device)
            hr = hr.to(device)
            pred = model(lr)
            p = to_numpy(pred)
            g = to_numpy(hr)
            p_psnr, p_ssim = compute_metrics(p, g)
            psnrs.append(p_psnr)
            ssims.append(p_ssim)

    print(f"Avg PSNR: {np.mean(psnrs):.3f} dB")
    print(f"Avg SSIM: {np.mean(ssims):.4f}")

if __name__ == "__main__":
    evaluate()
