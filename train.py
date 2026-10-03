import argparse
import os
import subprocess
import torch
import torch.nn as nn
from torch.optim import AdamW
from config import CFG
from models.factory import get_model
from utils.dataloader import get_loaders
from utils.metrics import dice_coef, iou_score
import numpy as np
from tqdm import tqdm
from utils.helpers import get_logits
from utils.cli import parse_args
import sys
import warnings
warnings.filterwarnings("ignore", message=".*NCCL.*")

from models.model_zoo import MODEL_ZOO

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ------------------ CLI ARGUMENTS ------------------

args = parse_args()

# --- Resolve architecture defaults from MODEL_ZOO ---
arch = args.architecture
defaults = MODEL_ZOO.get(arch, {})

# --- Core Config ---
CFG.architecture   = arch
CFG.model_name     = args.model_name or defaults.get("default_model", None)
CFG.dataset_root   = args.data_root
CFG.label_csv      = args.label_csv

# --- Model-related ---
CFG.in_channels    = args.in_channels or defaults.get("in_channels", 3)
CFG.num_classes    = args.num_classes or defaults.get("num_classes", 2)
CFG.freeze_encoder = args.freeze_encoder
CFG.use_dice_loss  = args.use_dice_loss
CFG.dice_weight    = args.dice_weight

# --- Training ---
CFG.epochs         = args.epochs
CFG.batch_size     = args.batch_size
CFG.learning_rate  = args.learning_rate
CFG.weight_decay   = args.weight_decay
CFG.val_every      = args.val_every
CFG.patience       = args.patience

# --- Eval / Logging ---
CFG.save_best_only        = args.save_best_only
CFG.num_eval_samples      = args.num_eval_samples
CFG.show_sample_predictions = args.show_sample_predictions
CFG.weights               = args.weights

# --- Image size from MODEL_ZOO ---
CFG.image_size = defaults.get("image_size", CFG.image_size)

# ------------------ SETUP ------------------
torch.manual_seed(CFG.seed)
os.makedirs(CFG.output_dir, exist_ok=True)

device = CFG.device
model = get_model()

# Enable multi-GPU if available
#if torch.cuda.device_count() > 1:
    #print(f"[INFO] Using {torch.cuda.device_count()} GPUs")
    #model = nn.DataParallel(model)

model = model.to(device)

train_loader, val_loader = get_loaders(CFG.dataset_root, CFG.label_csv)

# Optimizer
# optimizer = AdamW(model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay)


# IF you want to use RMSprop():
optimizer = torch.optim.RMSprop(model.parameters(),lr=CFG.learning_rate,alpha=0.995,weight_decay=0.0)

ce_loss = nn.CrossEntropyLoss(ignore_index=CFG.ignore_index)

def loss_fn(pred, target):
    ce = ce_loss(pred, target)
    if CFG.use_dice_loss:
        dice = dice_coef(pred, target, num_classes=CFG.num_classes)
        return ce + CFG.dice_weight * (1 - dice)
    return ce


#check
#print(type(model.module))
print("Number of classes:", CFG.num_classes)


# ------------------ TRAINING ------------------

## For tracking best val loss and patience counter for early stopping 
best_val_loss = float("inf")
no_improve_counter = 0

## 
dataset_name = os.path.basename(os.path.normpath(CFG.dataset_root))
ckpt_dir = os.path.join(CFG.output_dir, dataset_name, CFG.architecture, "checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)

for epoch in range(CFG.epochs):
    model.train()
    train_loss = []

    for images, masks in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{CFG.epochs}] Training", disable=False):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()

        outputs = get_logits(model(images))
        outputs = nn.functional.interpolate(outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

    avg_train_loss = np.mean(train_loss)

    if (epoch + 1) % CFG.val_every == 0:
        model.eval()
        val_loss = []
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validating", disable=False):
                images, masks = images.to(device), masks.to(device)
                outputs = get_logits(model(images))
                outputs = nn.functional.interpolate(outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                loss = loss_fn(outputs, masks)
                val_loss.append(loss.item())

        avg_val_loss = np.mean(val_loss)
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_counter = 0
            checkpoint_path = os.path.join(ckpt_dir, f"{dataset_name}_{CFG.architecture}_best.pt")
            torch.save(
                model.state_dict(),
                checkpoint_path)
            print(f"[Checkpoint] Saved best model to {checkpoint_path}")
        else:
            no_improve_counter += 1
            print(f"[Early Stop] No improvement for {no_improve_counter}/{CFG.patience} rounds")

        if no_improve_counter >= CFG.patience:
            print("[Early Stop] Validation loss did not improve, stopping training.")
            break
