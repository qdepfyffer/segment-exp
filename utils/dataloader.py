import os
from torch.utils.data import DataLoader
from torchvision import transforms as T
from dataset import SegmentationDataset
from config import CFG

def get_transforms():
    """Basic image transform. Expand here if needed."""
    return T.Compose([
        T.ToTensor()
    ])

def get_loaders(data_root="data", label_csv="class_dict.csv", include_test_only=False):
    """
    Returns train, val, and optionally test DataLoaders.

    Args:
        data_root (str): root directory of dataset (with train/val/test subfolders)
        label_csv (str): name of label CSV (inside data_root)
        include_test (bool): whether to return test_loader

    Returns:
        loaders: tuple of train_loader, val_loader[, test_loader]
    """
    transform = get_transforms()
    pin_memory = CFG.device.type == "cuda"
    num_workers = min(CFG.num_workers, os.cpu_count() or 1)

    if include_test_only:
        test_set = SegmentationDataset(data_root, split="test", label_csv=label_csv, transform=transform)
        test_loader = DataLoader(
            test_set,
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        return test_loader

    train_set = SegmentationDataset(data_root, split="train", label_csv=label_csv, transform=transform)
    val_set   = SegmentationDataset(data_root, split="val",   label_csv=label_csv, transform=transform)

    train_loader = DataLoader(
        train_set,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_set,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader
