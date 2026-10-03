from types import SimpleNamespace
import torch

CFG = SimpleNamespace(
    # General
    project_name="segmentation-pipeline",
    seed=42,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),

    # Placeholders – must be set by train/evaluate/single_evaluate
    architecture=None,
    model_name=None,
    num_classes=None,
    in_channels=None,
    image_size=(512, 512),
    ignore_index=255,
    pretrained=True,
    freeze_encoder=False,

    # Training
    epochs=100,
    batch_size=4,
    num_workers=8,
    learning_rate=1e-4,
    weight_decay=1e-4,
    val_every=1,
    patience=7,

    # Loss
    use_dice_loss=False,
    dice_weight=0.5,

    # Data paths
    dataset_root=None,
    label_csv="class_dict.csv",

    # Logging / Outputs
    output_dir="./results/",
    save_best_only=True,
    log_dir="./logs/",

    # Evaluation
    show_sample_predictions=True,
    num_eval_samples=20,

    # Thermal defaults
    USE_THERMAL=False,
    THERM_UPSAMPLE="bilinear",
    THERM_MEAN=0.0,
    THERM_STD=1.0,
    MODALITY_DROPOUT_P=0.0,
)
