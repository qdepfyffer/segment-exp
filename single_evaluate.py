#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import numpy as np
from PIL import Image
import cv2
from datetime import datetime

from config import CFG
from models.factory import get_model
from models.model_zoo import MODEL_ZOO
from utils.helpers import get_logits
from utils.visualization import save_mask, save_overlay, load_palette_from_csv
from utils.metrics import iou_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.flir_extractor import (
    FlirImageExtractor,
    crop_mask_and_overlay_temps,
    calculateCWSI
)

from torchvision import transforms

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# ------------------ CLI ------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Single image evaluation with temperature extraction")
    parser.add_argument("--image_path", type=str, required=True, help="Path to FLIR image")
    parser.add_argument("--architecture", type=str, required=True, help="Model architecture (e.g. setr, frrn_a)")
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root")
    parser.add_argument("--in_channels", type=int, default=None)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--weights", type=str, default=None, help="Optional checkpoint path")
    parser.add_argument("--at", type=float, default=30.0, help="Atmospheric temperature for filtering")
    parser.add_argument("--val_sub", type=float, default=0.0, help="Subtract threshold from Ta")
    parser.add_argument("--val_add", type=float, default=0.0, help="Add threshold to Ta")
    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------ CONFIG ------------------
    arch = args.architecture
    defaults = MODEL_ZOO.get(arch, {})

    # Core
    CFG.architecture   = arch
    CFG.model_name     = defaults.get("default_model", None)
    CFG.dataset_root   = args.data_root

    # Model-related
    CFG.in_channels    = args.in_channels or defaults.get("in_channels", 3)
    CFG.num_classes    = args.num_classes or defaults.get("num_classes", 2)
    CFG.trust_remote_code = defaults.get("trust_remote_code", False)

    # Input
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

    if isinstance(ckpt, dict) and "cfg" in ckpt:
        print("[INFO] Loading CFG from checkpoint...")
        for k, v in ckpt["cfg"].items():
            setattr(CFG, k, v)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    if any(k.startswith("module.") for k in state.keys()):
        print("[INFO] Stripping 'module.' prefix from state_dict keys...")
        state = {k.replace("module.", ""): v for k, v in state.items()}

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
            f"Original error:\n{e}"
        )

    print(f"[INFO] Model loaded: {CFG.architecture}")
    model.eval()

    # ------------------ LOAD IMAGE ------------------
    orig_img = Image.open(args.image_path).convert("RGB")
    orig_size = orig_img.size

    # Normalize image_size
    if isinstance(CFG.image_size, int):
        resize_size = (CFG.image_size, CFG.image_size)
    elif isinstance(CFG.image_size, (tuple, list)) and len(CFG.image_size) == 2:
        resize_size = CFG.image_size
    else:
        raise ValueError(f"Unexpected CFG.image_size format: {CFG.image_size}")

    preprocess = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.ToTensor(),
    ])

    image_tensor = preprocess(orig_img).unsqueeze(0).to(CFG.device)

    # ------------------ PREDICT MASK ------------------
    with torch.no_grad():
        logits = get_logits(model(image_tensor))
        preds = logits.argmax(dim=1).squeeze(0).cpu().numpy()

    pred_mask = cv2.resize(preds.astype(np.uint8), orig_size, interpolation=cv2.INTER_NEAREST)

    # ------------------ LOAD GROUND TRUTH ------------------
    basename = os.path.splitext(os.path.basename(args.image_path))[0]
    gt_path = os.path.join(CFG.dataset_root, "test_labels", basename + "_L.png")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth mask not found: {gt_path}")

    gt_mask_raw = np.array(Image.open(gt_path))

    # Map RGB mask to class indices using class_dict.csv
    csv_path = os.path.join(CFG.dataset_root, getattr(CFG, "label_csv", "labels.csv"))
    palette = load_palette_from_csv(csv_path) if os.path.exists(csv_path) else None

    if gt_mask_raw.ndim == 3 and gt_mask_raw.shape[-1] == 3 and palette is not None:
        # reshape flat list into Nx3 triplets
        palette_arr = np.array(palette).reshape(-1, 3)
        rgb2id = {tuple(v): i for i, v in enumerate(palette_arr)}

        h, w, _ = gt_mask_raw.shape
        gt_mask = np.zeros((h, w), dtype=np.int64)
        for rgb, idx in rgb2id.items():
            matches = np.all(gt_mask_raw == rgb, axis=-1)
            gt_mask[matches] = idx
    else:
        # already single channel
        gt_mask = gt_mask_raw

    # ------------------ METRICS ------------------
    acc = accuracy_score(gt_mask.flatten(), pred_mask.flatten())
    prec = precision_score(gt_mask.flatten(), pred_mask.flatten(), average="macro", zero_division=0)
    rec = recall_score(gt_mask.flatten(), pred_mask.flatten(), average="macro", zero_division=0)
    f1 = f1_score(gt_mask.flatten(), pred_mask.flatten(), average="macro", zero_division=0)
    iou = iou_score(torch.tensor(pred_mask)[None, ...], torch.tensor(gt_mask)[None, ...], CFG.num_classes)

    print("\n[Evaluation Results]")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"IoU:       {iou:.4f}")

    # ------------------ SAVE VISUALS ------------------
    csv_path = os.path.join(CFG.dataset_root, getattr(CFG, "label_csv", "labels.csv"))
    palette = load_palette_from_csv(csv_path) if os.path.exists(csv_path) else None
    today = datetime.now().strftime("%Y-%m-%d")
    out_dir = os.path.join("outputs", "single_eval", today)
    os.makedirs(out_dir, exist_ok=True)

    # Save pred_mask at original resolution (for downstream use / temperature extraction)
    save_mask(torch.tensor(pred_mask), os.path.join(out_dir, f"{basename}_mask.png"), palette)

    # For overlay, resize pred_mask to match network input image_tensor
    mask_for_overlay = cv2.resize(
        pred_mask,
        (image_tensor.shape[2], image_tensor.shape[1]),  # width, height from resized tensor
        interpolation=cv2.INTER_NEAREST
    )
    #save_overlay(image_tensor[0], torch.tensor(mask_for_overlay), os.path.join(out_dir, f"{basename}_overlay.png"), palette)


    # ------------------ TEMPERATURE EXTRACTION ------------------
    fie = FlirImageExtractor(
    exiftool_path=r"C:\Users\AIIS2-admin\exiftool\exiftool-13.38_64\exiftool.exe"
)
    fie.extracted_metadata = fie.extract_metadata(args.image_path)
    fie.updated_metadata = fie.extracted_metadata
    fie.process_image(args.image_path)
    thermal_np = fie.get_thermal_np()

    # Align mask to thermal resolution
    if pred_mask.shape != thermal_np.shape:
        print(f"[WARN] Resizing pred_mask from {pred_mask.shape} to match thermal {thermal_np.shape}")
        pred_mask = cv2.resize(pred_mask, (thermal_np.shape[1], thermal_np.shape[0]), interpolation=cv2.INTER_NEAREST)

    sunlit_temps = thermal_np[pred_mask == 1]
    if sunlit_temps.size > 0:
        avg_temp = np.mean(sunlit_temps)
        min_temp = np.min(sunlit_temps)
        max_temp = np.max(sunlit_temps)
        print("\n[Temperature Extraction]")
        print(f"Average sunlit temp: {avg_temp:.2f} °C")
        print(f"Min sunlit temp:     {min_temp:.2f} °C")
        print(f"Max sunlit temp:     {max_temp:.2f} °C")

        mean_sunlit_temp, temps_masked = crop_mask_and_overlay_temps(
            thermal_np, os.path.join(out_dir, f"{basename}_mask.png"),
            crop_w=0, crop_h=0,
            at=args.at, val_sub=args.val_sub, val_add=args.val_add
        )
        print(f"Filtered mean sunlit temp: {mean_sunlit_temp:.2f} °C")

        rh = float(fie.extracted_metadata['RelativeHumidity'])
        cwsi = calculateCWSI(args.at, avg_temp, rh)
        print(f"CWSI: {cwsi:.3f}")
    else:
        print("[WARN] No sunlit pixels found.")


if __name__ == "__main__":
    main()
