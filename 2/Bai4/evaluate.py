import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data import build_test_loader
from engine import evaluate
from model import PretrainedResNet50

PROJECT_DIR = Path(__file__).resolve().parent


def get_args():
    parser = argparse.ArgumentParser(description="Đánh giá best checkpoint Bài 4")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--mode", choices=("feature", "finetune"), default="finetune")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def save_confusion_matrix(matrix, classes, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(12, 10))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(np.arange(len(classes)), classes, rotation=90)
    axis.set_yticks(np.arange(len(classes)), classes)
    axis.set(xlabel="Dự đoán", ylabel="Nhãn thật", title=title)
    for row in range(len(classes)):
        for column in range(len(classes)):
            axis.text(
                column,
                row,
                int(matrix[row, column]),
                ha="center",
                va="center",
                fontsize=6,
            )
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_lab_comparison(output_dir):
    """Tổng hợp so sánh kết quả"""
    columns = [
        "model",
        "params",
        "trainable_params",
        "seconds_per_epoch",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
    ]
    rows = []
    result_dirs = [output_dir]
    result_dirs.extend(
        PROJECT_DIR.parent / folder / "outputs" for folder in ("1", "2", "3", "4")
    )
    for result_dir in dict.fromkeys(result_dirs):
        for path in sorted(result_dir.glob("*_test.json")):
            with path.open(encoding="utf-8") as stream:
                summary = json.load(stream).get("summary", {})
            if summary.get("model"):
                rows.append({column: summary.get(column, "") for column in columns})
    comparison_path = output_dir / "Lab2_model_comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return comparison_path


def main():
    args = get_args()
    run_name = f"Bai4_ResNet50_{args.mode}"
    checkpoint_path = (
        Path(args.ckpt or PROJECT_DIR / "outputs" / f"{run_name}_best.pt")
        .expanduser()
        .resolve()
    )
    checkpoint = load_checkpoint(checkpoint_path)
    classes = checkpoint["classes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PretrainedResNet50(
        len(classes),
        freeze_backbone=checkpoint["freeze_backbone"],
        model_id=checkpoint["model_id"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loader, test_classes, data_root = build_test_loader(
        args.data_root, checkpoint["img_size"], args.batch_size, args.num_workers
    )
    if classes != test_classes:
        raise ValueError("Thứ tự lớp test khác checkpoint")
    test_loss, summary, per_class, matrix = evaluate(
        model, test_loader, nn.CrossEntropyLoss(), device, classes
    )
    summary.update(
        {
            "model": checkpoint["model_name"],
            "params": checkpoint["params"],
            "trainable_params": checkpoint["trainable_params"],
            "best_val_epoch": checkpoint["epoch"],
            "best_val_f1": checkpoint["best_val_f1"],
            "test_loss": test_loss,
            "checkpoint": str(checkpoint_path),
        }
    )
    output_dir = checkpoint_path.parent
    with (output_dir / f"{run_name}_per_class.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=per_class[0].keys())
        writer.writeheader()
        writer.writerows(per_class)
    save_confusion_matrix(
        matrix, classes, output_dir / f"{run_name}_confusion_matrix.png", run_name
    )
    with (output_dir / f"{run_name}_test.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "summary": summary,
                "per_class": per_class,
                "confusion_matrix": matrix.tolist(),
            },
            stream,
            ensure_ascii=False,
            indent=2,
        )
    comparison_path = save_lab_comparison(output_dir)
    print(
        f"{run_name} | data={data_root} | best epoch={checkpoint['epoch']} | "
        f"accuracy={summary['accuracy']:.4f} | macro-F1={summary['f1_macro']:.4f}"
    )
    print("Bảng tổng hợp các kết quả hiện có:", comparison_path)


if __name__ == "__main__":
    main()
