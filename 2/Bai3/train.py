import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data import build_train_val_loaders
from engine import evaluate, seed_everything, train_one_epoch
from model import ResNet18, n_params

PROJECT_DIR = Path(__file__).resolve().parent


def get_args():
    parser = argparse.ArgumentParser(description="Train ResNet-18 từ đầu trên VinaFood21")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--variant", choices=("both", "noaug", "aug"), default="both")
    parser.add_argument("--img_size", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default=str(PROJECT_DIR / "outputs"))
    return parser.parse_args()


def save_history(history, path):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def save_loss_curve(history, path, run_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train_loss"] for row in history], label="train loss")
    plt.plot(epochs, [row["val_loss"] for row in history], label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross entropy loss")
    plt.title(run_name)
    plt.grid(alpha=0.2)
    plt.legend()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()


def train_variant(args, variant, device, out_dir):
    seed_everything(args.seed)
    augment = variant == "aug"
    run_name = f"Bai3_ResNet18_{variant}"
    train_loader, val_loader, classes, data_root = build_train_val_loaders(
        args.data_root, args.img_size, args.batch_size, args.val_ratio,
        args.num_workers, args.seed, augment,
    )
    if len(classes) != 21:
        raise ValueError(f"Cần đúng 21 lớp; tìm thấy {len(classes)}")
    model = ResNet18(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    checkpoint_path = out_dir / f"{run_name}_best.pt"
    history, best_f1, best_epoch = [], -1.0, 0
    print(f"\n{run_name} | device={device} | data={data_root} | "
          f"train={len(train_loader.dataset)} | val={len(val_loader.dataset)} | "
          f"params={n_params(model):,}")

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_summary, _, _ = evaluate(
            model, val_loader, criterion, device, classes
        )
        elapsed = time.perf_counter() - started
        val_f1 = val_summary["f1_macro"]
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "train_accuracy": train_accuracy, "val_loss": val_loss,
            "val_accuracy": val_summary["accuracy"],
            "val_f1_macro": val_f1, "seconds": elapsed,
        })
        marker = ""
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            torch.save({
                "format_version": 1, "model_state_dict": best_state,
                "model_name": run_name, "variant": variant,
                "classes": classes, "img_size": args.img_size,
                "epoch": epoch, "best_val_f1": best_f1,
                "params": n_params(model), "seed": args.seed,
                "seconds_per_epoch": float(np.mean([row["seconds"] for row in history])),
            }, checkpoint_path)
            marker = f" -> saved {checkpoint_path.name}"
        print(f"epoch {epoch:02d} | train {train_loss:.4f}/{train_accuracy:.4f} | "
              f"val {val_loss:.4f}/{val_summary['accuracy']:.4f} | "
              f"macro-F1 {val_f1:.4f} | {elapsed:.1f}s{marker}")

    save_history(history, out_dir / f"{run_name}_history.csv")
    save_loss_curve(history, out_dir / f"{run_name}_loss.png", run_name)
    print(f"Best epoch {best_epoch}, validation macro-F1={best_f1:.4f}")


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = ("noaug", "aug") if args.variant == "both" else (args.variant,)
    for variant in variants:
        train_variant(args, variant, device, out_dir)


if __name__ == "__main__":
    main()
