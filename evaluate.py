import os
import torch
import torch.nn.functional as F
from config import CFG
from models.factory import get_model
from utils.dataloader import get_loaders
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.metrics import iou_score
from utils.helpers import get_logits
from utils.cli import parse_args
from tqdm import tqdm
import numpy as np
from datetime import datetime
from utils.visualization import save_mask, save_overlay, load_palette_from_csv
from models.model_zoo import MODEL_ZOO

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ------------------ CLI ARGUMENTS ------------------
args = parse_args()

# --- Resolve architecture defaults ---
arch = args.architecture
defaults = MODEL_ZOO.get(arch, {})

# Core
CFG.architecture   = arch
CFG.model_name     = args.model_name or defaults.get("default_model", None)
CFG.dataset_root   = args.data_root
CFG.label_csv      = args.label_csv

# Model-related
CFG.in_channels    = args.in_channels or defaults.get("in_channels", 3)
CFG.num_classes    = args.num_classes or defaults.get("num_classes", 2)
CFG.freeze_encoder = args.freeze_encoder
CFG.use_dice_loss  = args.use_dice_loss
CFG.dice_weight    = args.dice_weight

# Eval/Logging
CFG.save_best_only          = args.save_best_only
CFG.num_eval_samples        = args.num_eval_samples
CFG.show_sample_predictions = args.show_sample_predictions

# Image size
CFG.image_size = defaults.get("image_size", CFG.image_size)

dataset_name = os.path.basename(os.path.normpath(CFG.dataset_root))

if args.weights is not None:
    CFG.weights = args.weights
else:
    CFG.weights = os.path.join(
        "results",
        dataset_name,
        arch,
        "checkpoints",
        f"{dataset_name}_{arch}_best.pt"
    )

print(f"[INFO] Using weights: {CFG.weights}")
if not os.path.exists(CFG.weights):
    raise FileNotFoundError(f"[ERROR] Checkpoint not found: {CFG.weights}")

# ------------------ LOAD CHECKPOINT ------------------
ckpt = torch.load(CFG.weights, map_location=CFG.device, weights_only=True)

# Restore config from checkpoint if available
if isinstance(ckpt, dict) and "cfg" in ckpt:
    print("[INFO] Loading CFG from checkpoint...")
    for k, v in ckpt["cfg"].items():
        setattr(CFG, k, v)

# Extract state dict
if isinstance(ckpt, dict) and "state_dict" in ckpt:
    state = ckpt["state_dict"]
else:
    state = ckpt

# Handle DataParallel-trained models
if any(k.startswith("module.") for k in state.keys()):
    print("[INFO] Stripping 'module.' prefix from state_dict keys...")
    state = {k.replace("module.", ""): v for k, v in state.items()}

# --- Re-apply CLI flags (after loading ckpt) ---
CFG.save_best_only          = args.save_best_only
CFG.num_eval_samples        = args.num_eval_samples
CFG.show_sample_predictions = args.show_sample_predictions

print(f"[INFO] Final eval flags -> "
      f"save_best_only={CFG.save_best_only}, "
      f"show_sample_predictions={CFG.show_sample_predictions}, "
      f"num_eval_samples={CFG.num_eval_samples}")

# ------------------ BUILD MODEL ------------------
model = get_model().to(CFG.device)
try:
    model.load_state_dict(state, strict=True)
except RuntimeError as e:
    raise RuntimeError(
        f"[LOAD ERROR] State dict doesn't match the current model.\n"
        f"- architecture: {CFG.architecture}\n"
        f"- model_name:   {CFG.model_name}\n"
        f"- num_classes:  {CFG.num_classes}\n"
        f"- in_channels:  {CFG.in_channels}\n\n"
        f"Common causes:\n"
        f"  • Mismatch in num_classes (e.g., 150 default vs dataset’s 2)\n"
        f"  • Mismatch in in_channels (e.g., RGB 3 vs RGB+Thermal 4)\n"
        f"  • Different model variant (e.g., segformer-b0 vs segformer-b3)\n\n"
        f"Original error:\n{e}"
    )

print(f"[INFO] Model loaded: {CFG.architecture}")
model.eval()

# ------------------ DATA ------------------
test_loader = get_loaders(CFG.dataset_root, CFG.label_csv, include_test_only=True)
csv_path = os.path.join(CFG.dataset_root, CFG.label_csv)
palette = load_palette_from_csv(csv_path)

# Output dirs
today = datetime.now().strftime("%Y-%m-%d")
mask_dir = os.path.join(CFG.output_dir, dataset_name, CFG.architecture, today, "masks")
overlay_dir = os.path.join(CFG.output_dir, dataset_name, CFG.architecture, today, "overlays")
os.makedirs(mask_dir, exist_ok=True)
os.makedirs(overlay_dir, exist_ok=True)

# ------------------ EVALUATION ------------------
flat_preds, flat_targets = [], []
all_preds_tensor, all_targets_tensor = [], []
sample_count = 0

with torch.no_grad():
    for images, masks in tqdm(test_loader, desc="Evaluating"):
        images, masks = images.to(CFG.device), masks.to(CFG.device)
        outputs = get_logits(model(images))
        outputs = F.interpolate(outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        preds = outputs.argmax(dim=1)

        flat_preds.extend(preds.detach().cpu().numpy().reshape(-1))
        flat_targets.extend(masks.detach().cpu().numpy().reshape(-1))

        all_preds_tensor.append(preds.detach().cpu())
        all_targets_tensor.append(masks.detach().cpu())

        if CFG.show_sample_predictions and sample_count < CFG.num_eval_samples:
            for b in range(images.size(0)):
                save_mask(preds[b], os.path.join(mask_dir, f"sample_{sample_count}_mask.png"), palette)
                save_overlay(images[b], preds[b], os.path.join(overlay_dir, f"sample_{sample_count}_overlay.png"), palette)
                sample_count += 1
                if sample_count >= CFG.num_eval_samples:
                    break

# ------------------ SANITY & COMPATIBILITY ------------------
all_preds_tensor = torch.cat(all_preds_tensor, dim=0).long()
all_targets_tensor = torch.cat(all_targets_tensor, dim=0).long()

if getattr(CFG, "ignore_index", None) is not None:
    tgt_valid_mask = all_targets_tensor != CFG.ignore_index
else:
    tgt_valid_mask = torch.ones_like(all_targets_tensor, dtype=torch.bool)

pred_max = int(all_preds_tensor.max().item()) if all_preds_tensor.numel() else 0
tgt_max = int(all_targets_tensor[tgt_valid_mask].max().item()) if tgt_valid_mask.any() else 0
observed_max = max(pred_max, tgt_max)

effective_num_classes = CFG.num_classes

if observed_max >= CFG.num_classes:
    if CFG.num_classes == 2:
        print(f"[WARN] Observed labels up to {observed_max} with CFG.num_classes=2 — collapsing to binary (0 vs >0).")
        all_preds_tensor = (all_preds_tensor != 0).long()
        if getattr(CFG, "ignore_index", None) is not None:
            remap_targets = torch.where(all_targets_tensor == CFG.ignore_index,
                                        all_targets_tensor,
                                        (all_targets_tensor != 0).long())
            all_targets_tensor = remap_targets
        else:
            all_targets_tensor = (all_targets_tensor != 0).long()
        observed_max = 1
        effective_num_classes = 2
    else:
        effective_num_classes = observed_max + 1
        print(f"[WARN] Observed labels up to {observed_max}; using effective_num_classes={effective_num_classes} for metrics.")

# ------------------ METRICS ------------------
flat_preds_np = all_preds_tensor.view(-1)
flat_targets_np = all_targets_tensor.view(-1)

if getattr(CFG, "ignore_index", None) is not None:
    valid = flat_targets_np != CFG.ignore_index
else:
    valid = torch.ones_like(flat_targets_np, dtype=torch.bool)

flat_preds_np = flat_preds_np[valid].cpu().numpy()
flat_targets_np = flat_targets_np[valid].cpu().numpy()

labels_for_sklearn = list(range(effective_num_classes))

acc  = accuracy_score(flat_targets_np, flat_preds_np)
prec = precision_score(flat_targets_np, flat_preds_np, labels=labels_for_sklearn, average="macro", zero_division=0)
rec  = recall_score(flat_targets_np, flat_preds_np, labels=labels_for_sklearn, average="macro", zero_division=0)
f1   = f1_score(flat_targets_np, flat_preds_np, labels=labels_for_sklearn, average="macro", zero_division=0)

iou = iou_score(all_preds_tensor, all_targets_tensor, effective_num_classes, ignore_index=CFG.ignore_index)

print("\n[Evaluation Results]")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"IoU:       {iou:.4f}")
