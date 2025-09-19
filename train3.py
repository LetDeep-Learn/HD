"""
Training script for SwinIR-lite.
- Uses dataset.SwinIRDataset
- Checkpointing and resume support (saves to Google Drive CHECKPOINT_DIR)
- Prints human-readable training progress in Colab terminal after each epoch
- Validation intentionally skipped: trains only and saves checkpoints per epoch
"""

import time
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
        vgg = torchvision.models.vgg19(
            weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1
        ).features.eval()
        self.vgg_layers = vgg
        for p in self.vgg_layers.parameters():
            p.requires_grad = False
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def forward(self, x, y):
        # x and y in [0,1] -> normalize to VGG
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        y = (y - self.mean.to(y.device)) / self.std.to(y.device)
        # up to relu2_2 (index 16)
        f_x = self.vgg_layers[:16](x)
        f_y = self.vgg_layers[:16](y)
        return nn.functional.l1_loss(f_x, f_y)


# Edge loss: L1 on Sobel maps
def edge_loss(pred, target):
    def sobel(img):
        # expects BxCxHxW, compute gradient magnitude
        img_gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]

        # define kernels once in float32 on the right device
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


# ---------------- Optimized checkpointing (local-first, CPU tensors) ---------------- #

def _to_cpu_state_dict(sd):
    return {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in sd.items()}


def _optimizer_state_to_cpu(opt_state):
    state = deepcopy(opt_state)  # {'state': {param_id: {...}}, 'param_groups': [...]}
    for param_id, slots in state.get('state', {}).items():
        for tname, tval in slots.items():
            if isinstance(tval, torch.Tensor):
                slots[tname] = tval.detach().cpu()
    return state


def save_checkpoint(state, epoch: int):
    # Ensure target directories exist
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    local_dir = Path("/content/checkpoints")
    local_dir.mkdir(parents=True, exist_ok=True)

    # Move heavy tensors to CPU before serialization
    state_cpu = {
        "epoch": state["epoch"],
        "model_state": _to_cpu_state_dict(state["model_state"]),
        "optimizer_state": _optimizer_state_to_cpu(state["optimizer_state"]),
        "scheduler_state": state["scheduler_state"],
    }

    # Paths
    local_last = local_dir / config.LAST_CHECKPOINT_NAME
    drive_last = config.CHECKPOINT_DIR / config.LAST_CHECKPOINT_NAME
    local_epoch = local_dir / f"checkpoint_epoch_{epoch}.pth"
    drive_epoch = config.CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pth"

    # Save locally first (fast), then copy to Drive
    torch.save(state_cpu, local_last, _use_new_zipfile_serialization=False)
    shutil.copy2(local_last, drive_last)

    # Save epoch-stamped every epoch
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
        print(f"Resumed from checkpoint at epoch {start_epoch-1}")
        return start_epoch
    else:
        return 1


# ---------------- Training (no validation) ---------------- #

def train():
    device = torch.device(config.DEVICE)
    print("Device:", device, flush=True)
    set_seed(config.SEED)

    # Ensure checkpoint directory exists early
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # dataset and loader
    train_ds = SwinIRDataset(
        hr_dir=config.HR_DIR,
        lr_dir=config.LR_DIR,
        mode="train",
        generate_on_fly=True,
        sketch_prob=0.7,
        samples_per_image=config.PATCHES_PER_IMAGE
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=min(getattr(config, "NUM_WORKERS", 2), 2),  # keep it conservative on Colab
        pin_memory=True,
        persistent_workers=True if getattr(config, "NUM_WORKERS", 2) > 0 else False,
    )

    model = create_model().to(device)
    optimizer = Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    # Step LR on train loss since validation is skipped
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    l1_loss = nn.L1Loss().to(device)
    perceptual = PerceptualLoss().to(device)

    # resume if checkpoint present
    start_epoch = load_checkpoint(model, optimizer, scheduler)

    global_step = 0
    for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        epoch_psnr = 0.0
        start_time = time.time()

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch}/{config.NUM_EPOCHS}"
        )
        for i, (lr, hr, meta) in pbar:
            lr = lr.to(device)
            hr = hr.to(device)
            optimizer.zero_grad()

            pred = model(lr)

            loss_l1 = l1_loss(pred, hr) * config.LOSS_L1_WEIGHT
            loss_perc = perceptual(pred, hr) * config.LOSS_PERCEPTUAL_WEIGHT
            loss_edge_val = edge_loss(pred, hr) * config.LOSS_EDGE_WEIGHT
            loss = loss_l1 + loss_perc + loss_edge_val

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_psnr = psnr(pred.detach(), hr.detach()).item()
            epoch_psnr += batch_psnr
            global_step += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "psnr": f"{batch_psnr:.2f}"})

        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_psnr = epoch_psnr / len(train_loader)

        # scheduler step on training loss
        scheduler.step(avg_train_loss)

        # checkpointing (save last and epoch-stamped)
        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        }
        save_checkpoint(state, epoch)

        epoch_time = time.time() - start_time

        # Human-friendly printout for Colab terminal
        print("=" * 60)
        print(f"Epoch {epoch:03d} finished. Time: {epoch_time:.1f}s")
        print(f"Train Loss: {avg_train_loss:.6f}  | Train PSNR: {avg_train_psnr:.3f}")
        print(f"Checkpoint saved to {config.CHECKPOINT_DIR}")
        print("=" * 60)


if __name__ == "__main__":
    train()
