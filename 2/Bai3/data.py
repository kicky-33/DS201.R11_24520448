import os
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def _is_data_root(path):
    return (path / "train").is_dir() and (path / "test").is_dir()


def resolve_data_root(data_root=None):
    """Tìm path."""
    project_dir = Path(__file__).resolve().parent
    candidates = []
    if data_root:
        candidates.append(Path(data_root))
    if os.environ.get("VINAFOOD21_ROOT"):
        candidates.append(Path(os.environ["VINAFOOD21_ROOT"]))
    candidates.extend(
        [
            project_dir.parent / "data" / "VinaFood21",
            project_dir / "data" / "VinaFood21",
            Path.cwd() / "data" / "VinaFood21",
            Path.cwd() / "VinaFood21",
        ]
    )
    for parent in Path.cwd().resolve().parents:
        candidates.append(parent / "data" / "VinaFood21")

    checked = []
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate in checked:
            continue
        checked.append(candidate)
        if _is_data_root(candidate):
            return candidate
    raise FileNotFoundError(
        "Không tìm thấy VinaFood21 có train/ và test/. Truyền --data_root hoặc "
        "đặt biến môi trường VINAFOOD21_ROOT. Đã kiểm tra:\n- "
        + "\n- ".join(map(str, checked))
    )


def build_transforms(img_size=96, augment=True):
    evaluation = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    if not augment:
        return evaluation, evaluation
    training = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    return training, evaluation


def _loader(dataset, batch_size, shuffle, num_workers, generator=None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def build_train_val_loaders(
    data_root=None,
    img_size=96,
    batch_size=32,
    val_ratio=0.15,
    num_workers=0,
    seed=42,
    augment=True,
):
    root = resolve_data_root(data_root)
    train_transform, eval_transform = build_transforms(img_size, augment)
    train_all = datasets.ImageFolder(root / "train", transform=train_transform)

    if (root / "val").is_dir():
        train_dataset = train_all
        val_dataset = datasets.ImageFolder(root / "val", transform=eval_transform)
        if train_all.classes != val_dataset.classes:
            raise ValueError("Thứ tự lớp train/val không khớp")
    else:
        labels = np.asarray(train_all.targets)
        indices = np.arange(len(labels))
        train_indices, val_indices = train_test_split(
            indices, test_size=val_ratio, stratify=labels, random_state=seed
        )
        train_dataset = Subset(train_all, train_indices.tolist())
        val_all = datasets.ImageFolder(root / "train", transform=eval_transform)
        val_dataset = Subset(val_all, val_indices.tolist())

    generator = torch.Generator().manual_seed(seed)
    train_loader = _loader(
        train_dataset, batch_size, True, num_workers, generator=generator
    )
    val_loader = _loader(val_dataset, batch_size, False, num_workers)
    return train_loader, val_loader, train_all.classes, root


def build_test_loader(data_root=None, img_size=96, batch_size=32, num_workers=0):
    root = resolve_data_root(data_root)
    _, evaluation = build_transforms(img_size, augment=False)
    dataset = datasets.ImageFolder(root / "test", transform=evaluation)
    return _loader(dataset, batch_size, False, num_workers), dataset.classes, root
