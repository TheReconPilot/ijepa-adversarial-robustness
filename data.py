from __future__ import annotations
import random
from typing import Tuple, Dict, Optional
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from sklearn.model_selection import StratifiedShuffleSplit

# For ImageNet-100 (HF datasets)
try:
    from datasets import load_dataset  # type: ignore
except Exception:
    load_dataset = None  # will error at runtime if imagenet100 selected

_BICUBIC = InterpolationMode.BICUBIC


def _resolve_target_size_from_processor(processor, default: int = 224) -> int:
    size = getattr(processor, "size", None) or getattr(processor, "crop_size", None)
    if isinstance(size, dict):
        return size.get("shortest_edge", size.get("height", default))
    elif isinstance(size, int):
        return size
    return default


def _build_transforms(processor) -> Tuple[transforms.Compose, transforms.Compose]:
    mean = getattr(processor, "image_mean", None)
    std = getattr(processor, "image_std", None)
    target = _resolve_target_size_from_processor(processor, 224)

    normalize = (
        transforms.Normalize(mean=mean, std=std)
        if (mean is not None and std is not None)
        else transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(target, scale=(0.8, 1.0), interpolation=_BICUBIC, antialias=True),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        normalize,
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(256, interpolation=_BICUBIC, antialias=True),
        transforms.CenterCrop(target),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, eval_tf


def _seed_all(seed: int):
    import numpy as np
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


# ------------------------------
# CIFAR-100
# ------------------------------

def get_dataloaders_cifar100(
    processor,
    batch_size: int = 256,
    workers: int = 8,
    val_ratio: float = 0.1,
    seed: int = 0,
):
    """
    Stratified train/val split from CIFAR-100 train (50k → 45k/5k by default) + official test (10k).
    Transforms derived from the HF processor.
    """
    _seed_all(seed)
    train_tf, eval_tf = _build_transforms(processor)

    full_train = datasets.CIFAR100(root="./data", train=True, download=True, transform=train_tf)
    full_train_for_val = datasets.CIFAR100(root="./data", train=True, download=False, transform=eval_tf)
    test_set = datasets.CIFAR100(root="./data", train=False, download=True, transform=eval_tf)

    targets = full_train.targets
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(sss.split(list(range(len(targets))), targets))

    train_set = Subset(full_train, train_idx)
    val_set = Subset(full_train_for_val, val_idx)

    common = dict(batch_size=batch_size, num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **common)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader, test_loader


# ------------------------------
# ImageNet-100 (HF Hub: ilee0022/ImageNet100)
# ------------------------------

def get_dataloaders_imagenet100(
    processor,
    batch_size: int = 256,
    workers: int = 8,
    seed: int = 0,
):
    """Use the official train/validation/test splits provided by the HF dataset."""
    if load_dataset is None:
        raise ImportError("datasets[vision] is required for imagenet100 dataloader. Install `pip install datasets`.")

    _seed_all(seed)
    train_tf, eval_tf = _build_transforms(processor)

    dsets = load_dataset("ilee0022/ImageNet100")
    if not all(k in dsets for k in ["train", "validation", "test"]):
        raise RuntimeError("ImageNet100 dataset missing required splits {train, validation, test}.")

    def apply_transforms(batch: Dict, tf: transforms.Compose) -> Dict:
        batch["pixel_values"] = [tf(img.convert("RGB")) for img in batch["image"]]
        return batch

    dsets["train"].set_transform(lambda batch: apply_transforms(batch, train_tf))
    dsets["validation"].set_transform(lambda batch: apply_transforms(batch, eval_tf))
    dsets["test"].set_transform(lambda batch: apply_transforms(batch, eval_tf))

    def collate_fn(batch):
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        labels = torch.tensor([item["label"] for item in batch])
        return pixel_values, labels

    common = dict(batch_size=batch_size, num_workers=workers, pin_memory=True, persistent_workers=workers > 0, collate_fn=collate_fn)
    train_loader = DataLoader(dsets["train"], shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(dsets["validation"], shuffle=False, drop_last=False, **common)
    test_loader = DataLoader(dsets["test"], shuffle=False, drop_last=False, **common)
    return train_loader, val_loader, test_loader


# ------------------------------
# Dispatcher / public API
# ------------------------------

_DATASETS = {
    "cifar100": {
        "loader": get_dataloaders_cifar100,
        "num_classes": 100,
    },
    "imagenet100": {
        "loader": get_dataloaders_imagenet100,
        "num_classes": 100,
    },
}


def get_num_classes(dataset: str) -> int:
    key = dataset.lower()
    if key not in _DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Available: {list(_DATASETS.keys())}")
    return _DATASETS[key]["num_classes"]


def get_dataloaders(
    dataset: str,
    processor,
    batch_size: int = 256,
    workers: int = 8,
    val_ratio: float = 0.1,
    seed: int = 0,
):
    key = dataset.lower()
    if key not in _DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Available: {list(_DATASETS.keys())}")
    if key == "cifar100":
        return _DATASETS[key]["loader"](processor, batch_size=batch_size, workers=workers, val_ratio=val_ratio, seed=seed)
    else:
        # imagenet100 ignores val_ratio
        return _DATASETS[key]["loader"](processor, batch_size=batch_size, workers=workers, seed=seed)

