"""
Training script for SwinIR-lite.
- Uses dataset.SwinIRDataset
- Checkpointing and resume support (saves to Google Drive CHECKPOINT_DIR)
- Prints human-readable training progress in Colab terminal after each epoch
"""

import os
import time
import json
import shutil
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import torchvision

import config
from dataset import SwinIRDataset, set_seed
from model import create_model

# ---------------- Losses ---------------- #

# Perceptual loss using pretrained VGG
class PerceptualLoss(nn.Module):
    def __init__(self, layer="relu3_3"):
        super().__init__()
        vgg = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1).features.eval()
        self.vgg_layers = vgg
        for p in self.vgg_layers.parameters():
            p.requires_grad = False
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

    def forward(self, x, y):
        if x.shape[1] == 1:
            x = x.repeat(1,3,1,1)
            y = y.repeat(1,3,1,1)
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        y = (y - self.mean.to(y.device)) / self.std.to(y.device)
        f_x = self.vgg_layers[:16](x)
        f_y = self.vgg_layers[:16](y)
        return nn.functional.l1_loss(f_x, f_y)

# Edge loss: L1 on Sobel maps
def edge_loss(pred, target):
    def sobel(img):
        img_gray = 0.299*img[:, 0:1] + 0.587*img[:, 1:2] + 0.114*img[:, 2:3]
        sobel_x = torch.tensor(
            [[[[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]]]],
            dtype=torch.float32,
            device=img.device
        )
        sobel_y = torch.tensor(
            [[[[-1, -2, -1],
               [ 0,  0,  0],
               [ 1,  2,  1]]]],
            dtype=torch.float32,
            device=img.device
        )
        gx = nn.functional.conv2d(img_gray, weight=sobel_x, padding=1)
        gy = nn.functional.conv2d(img_gray, weight=sobel_y, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
        return mag
    return nn.functional.l1_loss(sobel(pred), sobel(target))

def psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return 100.0
    return 20 * torch.log10(max_val / torch.sqrt(mse))

# ---------------- Optimised Checkpoint Saving ---------------- #

def _to_cpu_state_dict(sd):
    return {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in sd.items()}

def _optimizer_state_to_cpu(opt_state):
    state = deepcopy(opt_state)
    for k, v in state['state'].items():
        for tname, tval in v.items():
            if isinstance(tval, torch.Tensor):
                state['state'][k][tname] = tval.detach().cpu()
    return state

def save_checkpoint(state, is_best: bool, epoch: int):
    # Ensure dirs
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    local_dir = Path("/content/checkpoints")
    local_dir.mkdir(parents=True, exist_ok=True)

    # Move heavy tensors to CPU
    state_cpu = {
        "epoch": state["epoch"],
        "best_metric": state["best_metric"],
        "model_state": _to_cpu_state_dict(state["model_state"]),
        "optimizer_state": _optimizer_state_to_cpu(state["optimizer_state"]),
        "scheduler_state": state["scheduler_state"],
    }

    # Paths
    local_last = local_dir / config.LAST_CHECKPOINT_NAME
    drive_last = config.CHECKPOINT_DIR / config.LAST_CHECKPOINT_NAME
    local_best = local_dir / config.BEST_MODEL_NAME
    drive_best = config.CHECKPOINT_DIR / config.BEST_MODEL_NAME
    local_epoch = local_dir / f"checkpoint_epoch_{epoch}.pth"
    drive_epoch = config.CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pth"

    # Save locally first
    torch.save(state_cpu, local_last, _use_new_zipfile_serialization=False)
    shutil.copy2(local_last, drive_last)

    if is_best:
        shutil.copy2(local_last, local_best)
        shutil.copy2(local_best, drive_best)

    # Optional: save epoch-stamped every N epochs
    N = getattr(config, "EPOCH_SAVE_INTERVAL", 0)  # 0 disables
    if N and (epoch % N == 0):
        shutil.copy2(local_last, local_epoch)
        shutil.copy2(local_epoch, drive_epoch)

def load_checkpoint(model, optimizer=None, scheduler=None):
    last_path = config.CHECKPOINT_DIR / config.LAST_CHECKPOINT_NAME
    if last_path.exists():
        ckpt = torch.load(last_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        if optimizer is not None and "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler is not None and "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_metric = ckpt.get("best_metric", None)
        print(f"Resumed from checkpoint at epoch {start_epoch-1}, best_metric={best_metric}")
        return start_epoch, best_metric
    else:
        return 1, None

# ---------------- Training ---------------- #

def train():
    device = torch.device(config.DEVICE)
    print("Device:", device)
    set_seed(config.SEED)

    # dataset and loader
    train_ds = SwinIRDataset(hr_dir=config.HR_DIR, lr_dir=config.LR_DIR, mode="train",
                             generate_on_fly=True, sketch_prob=0.7,
                             samples_per_image=config.PATCHES_PER_IMAGE)
    val_ds = SwinIRDataset(hr_dir=config.HR_DIR, lr_dir=config.LR_DIR, mode="val",
                           generate_on_fly=True, sketch_prob=0.7, samples_per_image=1)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=min(config.NUM_WORKERS, 2), pin_memory=True,
                              persistent_workers=True if config.NUM_WORKERS > 0 else False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = create_model().to(device)
    optimizer = Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    l1_loss = nn.L1Loss().to(device)
    perceptual = PerceptualLoss().to(device)

    # resume if checkpoint present
    start_epoch, best_metric = load_checkpoint(model, optimizer, scheduler)

    # If no best_metric found in checkpoint, set to +inf (we minimize val loss)
    if best_metric is None:
        best_metric = float("inf")

    global_step = 0
    try:
        for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
            model.train()
            epoch_loss = 0.0
            epoch_psnr = 0.0
            start_time = time.time()
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{config.NUM_EPOCHS}")
            for i, (lr, hr, meta) in pbar:
                lr = lr.to(device)
                hr = hr.to(device)

                optimizer.zero_grad()
                pred = model(lr)

                loss_l1 = l1_loss(pred, hr) * config.LOSS_L1_WEIGHT
                loss_perc = perceptual(pred, hr) * config.LOSS_PERCEPTUAL_WEIGHT
                loss_edge = edge_loss(pred, hr) * config.LOSS_EDGE_WEIGHT
                loss = loss_l1 + loss_perc + loss_edge

                loss.backward()
                if getattr(config, "GRAD_CLIP", 0) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                optimizer.step()

                with torch.no_grad():
                    batch_psnr = psnr(pred, hr).item()

                epoch_loss += loss.item()
                epoch_psnr += batch_psnr
                global_step += 1

                # Update tqdm with live stats
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "l1": f"{loss_l1.item():.4f}",
                    "perc": f"{loss_perc.item():.4f}",
                    "edge": f"{loss_edge.item():.4f}",
                    "psnr": f"{batch_psnr:.3f}"
                })

            # End of epoch training stats
            avg_train_loss = epoch_loss / len(train_loader)
            avg_train_psnr = epoch_psnr / len(train_loader)

            # Validation
            model.eval()
            val_loss = 0.0
            val_psnr = 0.0
            with torch.no_grad():
                for j, (lr, hr, meta) in enumerate(tqdm(val_loader, desc="Validation", leave=False)):
                    lr = lr.to(device)
                    hr = hr.to(device)
                    pred = model(lr)
                    loss_l1_v = l1_loss(pred, hr) * config.LOSS_L1_WEIGHT
                    loss_perc_v = perceptual(pred, hr) * config.LOSS_PERCEPTUAL_WEIGHT
                    loss_edge_v = edge_loss(pred, hr) * config.LOSS_EDGE_WEIGHT
                    loss_v = loss_l1_v + loss_perc_v + loss_edge_v

                    val_loss += loss_v.item()
                    val_psnr += psnr(pred, hr).item()

            avg_val_loss = val_loss / len(val_loader)
            avg_val_psnr = val_psnr / len(val_loader)

            # Scheduler step (ReduceLROnPlateau expects validation metric)
            scheduler.step(avg_val_loss)

            elapsed = time.time() - start_time
            current_lr = optimizer.param_groups[0]['lr']

            # Checkpointing: lower validation loss is better
            is_best = avg_val_loss < best_metric
            if is_best:
                best_metric = avg_val_loss

            state = {
                "epoch": epoch,
                "best_metric": best_metric,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
            }
            save_checkpoint(state, is_best=is_best, epoch=epoch)

            # Human readable epoch summary
            print(
                f"Epoch {epoch:03d} | Time: {elapsed:.1f}s | LR: {current_lr:.3e} | "
                f"Train Loss: {avg_train_loss:.4f} | Train PSNR: {avg_train_psnr:.3f} | "
                f"Val Loss: {avg_val_loss:.4f} | Val PSNR: {avg_val_psnr:.3f} | Best Val Loss: {best_metric:.4f}"
            )

    except KeyboardInterrupt:
        print("Training interrupted by user. Saving last checkpoint.")
        # Save interrupted checkpoint
        state = {
            "epoch": epoch,
            "best_metric": best_metric,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        }
        save_checkpoint(state, is_best=False, epoch=epoch)
        raise

if __name__ == "__main__":
    train()
