import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from data import build_test_loader
from engine import evaluate
from model import LeNet5

PROJECT_DIR = Path(__file__).resolve().parent


def get_args():
    parser = argparse.ArgumentParser(
        description="Đánh giá best checkpoint của LeNet-5 trên tập test"
    )
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--ckpt", nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def save_confusion_matrix(cm, classes, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(classes)), classes, rotation=90)
    ax.set_yticks(np.arange(len(classes)), classes)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Nhãn thật")
    ax.set_title(title)
    for row in range(cm.shape[0]):
        for column in range(cm.shape[1]):
            ax.text(column, row, int(cm[row, column]), ha="center", va="center", fontsize=6)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_per_class(rows, path):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def evaluate_checkpoint(checkpoint_path, args, device):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = load_checkpoint(checkpoint_path)
    classes = checkpoint["classes"]
    pool = checkpoint["pool"]
    img_size = int(checkpoint["img_size"])
    run_name = checkpoint.get("model_name", checkpoint_path.stem.removesuffix("_best"))

    model = LeNet5(
        in_ch=3,
        img_size=img_size,
        num_classes=len(classes),
        pool=pool,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_dl, test_classes, data_root = build_test_loader(
        args.data_root,
        img_size=img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    if test_classes != classes:
        raise ValueError("Thứ tự lớp test khác checkpoint")

    test_loss, summary, per_class, cm, _, _ = evaluate(
        model, test_dl, nn.CrossEntropyLoss(), device, classes
    )
    summary.update({
        "model": run_name,
        "pool": pool,
        "checkpoint": str(checkpoint_path),
        "best_val_epoch": int(checkpoint["epoch"]),
        "best_val_f1": float(checkpoint["best_val_f1"]),
        "test_loss": float(test_loss),
    })

    output_dir = checkpoint_path.parent
    save_per_class(per_class, output_dir / f"{run_name}_per_class.csv")
    save_confusion_matrix(
        cm,
        classes,
        output_dir / f"{run_name}_confusion_matrix.png",
        f"{run_name}: confusion matrix 21x21",
    )
    with (output_dir / f"{run_name}_test.json").open("w", encoding="utf-8") as stream:
        json.dump({
            "summary": summary,
            "per_class": per_class,
            "confusion_matrix": cm.tolist(),
        }, stream, ensure_ascii=False, indent=2)

    print(
        f"{run_name} | data={data_root} | best epoch={checkpoint['epoch']} | "
        f"test accuracy={summary['accuracy']:.4f} | "
        f"precision={summary['precision_macro']:.4f} | "
        f"recall={summary['recall_macro']:.4f} | "
        f"macro-F1={summary['f1_macro']:.4f}"
    )
    return summary


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_paths = args.ckpt
    if not checkpoint_paths:
        checkpoint_paths = sorted((PROJECT_DIR / "outputs").glob("Bai1_LeNet_*_best.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError("Chưa có checkpoint trong thư mục outputs/")

    summaries = [
        evaluate_checkpoint(path, args, device)
        for path in checkpoint_paths
    ]
    # Ghi cạnh các checkpoint. Trên Kaggle, PROJECT_DIR thường nằm trong
    # /kaggle/input (read-only), còn checkpoint nằm trong /kaggle/working.
    comparison_path = Path(checkpoint_paths[0]).expanduser().resolve().parent / "Bai1_comparison.csv"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    with comparison_path.open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "model", "pool", "best_val_epoch", "best_val_f1", "test_loss",
            "accuracy", "precision_macro", "recall_macro", "f1_macro", "checkpoint",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    print("Đã lưu bảng so sánh:", comparison_path)


if __name__ == "__main__":
    main()
