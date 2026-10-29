import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data import build_test_loader
from engine import evaluate
from model import ResNet18

PROJECT_DIR = Path(__file__).resolve().parent


def get_args():
    parser = argparse.ArgumentParser(description="Đánh giá best checkpoint Bài 3")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--variant", choices=("noaug", "aug"), default="aug")
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
            axis.text(column, row, int(matrix[row, column]),
                      ha="center", va="center", fontsize=6)
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    args = get_args()
    run_name = f"Bai3_ResNet18_{args.variant}"
    checkpoint_path = Path(args.ckpt or PROJECT_DIR / "outputs" /
                           f"{run_name}_best.pt").expanduser().resolve()
    checkpoint = load_checkpoint(checkpoint_path)
    classes = checkpoint["classes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18(len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loader, test_classes, data_root = build_test_loader(
        args.data_root, checkpoint["img_size"], args.batch_size, args.num_workers
    )
    if classes != test_classes:
        raise ValueError("Thứ tự lớp test khác checkpoint")
    test_loss, summary, per_class, matrix = evaluate(
        model, test_loader, nn.CrossEntropyLoss(), device, classes
    )
    summary.update({
        "model": checkpoint["model_name"], "params": checkpoint["params"],
        "best_val_epoch": checkpoint["epoch"],
        "best_val_f1": checkpoint["best_val_f1"], "test_loss": test_loss,
        "checkpoint": str(checkpoint_path),
    })
    output_dir = checkpoint_path.parent
    with (output_dir / f"{run_name}_per_class.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=per_class[0].keys())
        writer.writeheader(); writer.writerows(per_class)
    save_confusion_matrix(
        matrix, classes, output_dir / f"{run_name}_confusion_matrix.png", run_name
    )
    with (output_dir / f"{run_name}_test.json").open("w", encoding="utf-8") as stream:
        json.dump({"summary": summary, "per_class": per_class,
                   "confusion_matrix": matrix.tolist()},
                  stream, ensure_ascii=False, indent=2)
    print(f"{run_name} | data={data_root} | best epoch={checkpoint['epoch']} | "
          f"accuracy={summary['accuracy']:.4f} | macro-F1={summary['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
