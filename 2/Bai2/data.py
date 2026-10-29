from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def resolve_data_root(data_root=None):
    """Tìm dataset."""
    project_2 = Path(__file__).resolve().parent.parent
    candidates = []
    if data_root:
        candidates.append(Path(data_root))
    candidates.extend(
        [
            project_2 / "data" / "VinaFood21",
            Path.cwd() / "data" / "VinaFood21",
            Path.cwd() / "2" / "data" / "VinaFood21",
            Path.cwd().parent / "data" / "VinaFood21",
            Path.cwd() / "VinaFood21",
        ]
    )

    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if (candidate / "train").is_dir() and (candidate / "test").is_dir():
            return candidate

    checked = "\n- ".join(str(path.expanduser().resolve()) for path in candidates)
    raise FileNotFoundError(
        "Không tìm thấy VinaFood21 có đủ train/ và test/. Đã kiểm tra:\n- " + checked
    )


def build_transforms(img_size, augment=True):
    eval_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    if not augment:
        return eval_tf, eval_tf

    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    return train_tf, eval_tf


def build_train_val_loaders(
    data_root,
    img_size=64,
    batch_size=32,
    val_ratio=0.15,
    num_workers=0,
    seed=42,
    augment=True,
):
    root = resolve_data_root(data_root)
    train_tf, eval_tf = build_transforms(img_size, augment=augment)
    train_all = datasets.ImageFolder(root / "train", transform=train_tf)

    if (root / "val").is_dir():
        train_ds = train_all
        val_ds = datasets.ImageFolder(root / "val", transform=eval_tf)
        if train_all.classes != val_ds.classes:
            raise ValueError("Thứ tự lớp train/val không khớp")
    else:
        labels = np.asarray(train_all.targets)
        indices = np.arange(len(labels))
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_ratio,
            stratify=labels,
            random_state=seed,
        )
        train_ds = Subset(train_all, train_idx.tolist())
        val_all = datasets.ImageFolder(root / "train", transform=eval_tf)
        val_ds = Subset(val_all, val_idx.tolist())

    generator = torch.Generator().manual_seed(seed)
    common = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available())
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        **common,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        **common,
    )
    return train_dl, val_dl, train_all.classes, root


def build_test_loader(data_root, img_size=64, batch_size=32, num_workers=0):
    root = resolve_data_root(data_root)
    _, eval_tf = build_transforms(img_size, augment=False)
    test_ds = datasets.ImageFolder(root / "test", transform=eval_tf)
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return test_dl, test_ds.classes, root
