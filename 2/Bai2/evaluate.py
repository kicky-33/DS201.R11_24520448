import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from data import build_test_loader
from engine import evaluate
from model import GoogLeNetSmall

PROJECT_DIR = Path(__file__).resolve().parent


def get_args():
    parser = argparse.ArgumentParser(
        description="Đánh giá best checkpoint GoogLeNet trên tập test"
    )
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument(
        "--ckpt",
        type=str,
        default=str(PROJECT_DIR / "outputs" / "Bai2_GoogLeNet_best.pt"),
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def save_confusion_matrix(cm, classes, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(classes)), classes, rotation=90)
    ax.set_yticks(np.arange(len(classes)), classes)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Nhãn thật")
    ax.set_title("Bai2_GoogLeNet: confusion matrix 21x21")
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


def save_bai1_comparison(summary, output_dir):
    bai1_path = PROJECT_DIR.parent / "1" / "outputs" / "Bai1_comparison.csv"
    rows = []
    if bai1_path.is_file():
        with bai1_path.open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                rows.append({
                    "model": row["model"],
                    "params": 339_421,
                    "seconds_per_epoch": "",
                    "accuracy": row["accuracy"],
                    "precision_macro": row["precision_macro"],
                    "recall_macro": row["recall_macro"],
                    "f1_macro": row["f1_macro"],
                })
    rows.append({
        "model": summary["model"],
        "params": summary["params"],
        "seconds_per_epoch": summary["seconds_per_epoch"],
        "accuracy": summary["accuracy"],
        "precision_macro": summary["precision_macro"],
        "recall_macro": summary["recall_macro"],
        "f1_macro": summary["f1_macro"],
    })
    path = output_dir / "Bai2_vs_Bai1.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return path


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.ckpt).expanduser().resolve()
    checkpoint = load_checkpoint(checkpoint_path)
    classes = checkpoint["classes"]
    img_size = int(checkpoint["img_size"])

    model = GoogLeNetSmall(num_classes=len(classes)).to(device)
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
        "model": checkpoint.get("model_name", "Bai2_GoogLeNet"),
        "params": int(checkpoint["params"]),
        "seconds_per_epoch": float(checkpoint["seconds_per_epoch"]),
        "checkpoint": str(checkpoint_path),
        "best_val_epoch": int(checkpoint["epoch"]),
        "best_val_f1": float(checkpoint["best_val_f1"]),
        "test_loss": float(test_loss),
    })

    output_dir = checkpoint_path.parent
    save_per_class(per_class, output_dir / "Bai2_GoogLeNet_per_class.csv")
    save_confusion_matrix(cm, classes, output_dir / "Bai2_GoogLeNet_confusion_matrix.png")
    with (output_dir / "Bai2_GoogLeNet_test.json").open("w", encoding="utf-8") as stream:
        json.dump({
            "summary": summary,
            "per_class": per_class,
            "confusion_matrix": cm.tolist(),
        }, stream, ensure_ascii=False, indent=2)
    comparison_path = save_bai1_comparison(summary, output_dir)

    print(
        f"Bai2_GoogLeNet | data={data_root} | best epoch={checkpoint['epoch']} | "
        f"test accuracy={summary['accuracy']:.4f} | "
        f"precision={summary['precision_macro']:.4f} | "
        f"recall={summary['recall_macro']:.4f} | "
        f"macro-F1={summary['f1_macro']:.4f}"
    )
    print("Đã lưu bảng so sánh với Bài 1:", comparison_path)


if __name__ == "__main__":
    main()
