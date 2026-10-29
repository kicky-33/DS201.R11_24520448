import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn

from data import build_train_val_loaders
from engine import evaluate, seed_everything, train_one_epoch
from model import LeNet5, n_params

PROJECT_DIR = Path(__file__).resolve().parent


def get_args():
    parser = argparse.ArgumentParser(
        description="Huấn luyện LeNet-5 AvgPool và MaxPool trên VinaFood21"
    )
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--pool", choices=("avg", "max", "both"), default="both")
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default=str(PROJECT_DIR / "outputs"))
    return parser.parse_args()


def save_history(history, path):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def save_loss_curve(history, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train_loss"] for row in history], label="train loss")
    plt.plot(epochs, [row["val_loss"] for row in history], label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross entropy loss")
    plt.title(title)
    plt.grid(alpha=0.2)
    plt.legend()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()


def train_variant(pool, args, device, out_dir):
    seed_everything(args.seed)
    train_dl, val_dl, classes, data_root = build_train_val_loaders(
        args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        seed=args.seed,
        augment=True,
    )
    if len(classes) != 21:
        raise ValueError(f"Cần đúng 21 lớp; tìm thấy {len(classes)}")

    model = LeNet5(
        in_ch=3,
        img_size=args.img_size,
        num_classes=len(classes),
        pool=pool,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    run_name = f"Bai1_LeNet_{pool}"
    checkpoint_path = out_dir / f"{run_name}_best.pt"
    history, best_f1, best_epoch = [], -1.0, 0

    print(
        f"\n{run_name} | device={device} | data={data_root} | "
        f"train={len(train_dl.dataset)} | val={len(val_dl.dataset)} | "
        f"params={n_params(model):,}"
    )
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_dl, criterion, optimizer, device
        )
        val_loss, val_summary, _, _, _, _ = evaluate(
            model, val_dl, criterion, device, classes
        )
        val_f1 = val_summary["f1_macro"]
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_summary["accuracy"],
            "val_f1_macro": val_f1,
        })

        marker = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            torch.save({
                "format_version": 1,
                "model_state_dict": best_state,
                "model_name": run_name,
                "pool": pool,
                "classes": classes,
                "img_size": args.img_size,
                "epoch": epoch,
                "best_val_f1": best_f1,
                "seed": args.seed,
            }, checkpoint_path)
            marker = f"  -> saved {checkpoint_path.name}"

        print(
            f"epoch {epoch:02d} | train {train_loss:.4f}/{train_acc:.4f} | "
            f"val {val_loss:.4f}/{val_summary['accuracy']:.4f} | "
            f"macro-F1 {val_f1:.4f}{marker}"
        )

    save_history(history, out_dir / f"{run_name}_history.csv")
    save_loss_curve(
        history,
        out_dir / f"{run_name}_loss.png",
        f"{run_name}: train và validation loss",
    )
    print(
        f"Best epoch {best_epoch}, validation macro-F1={best_f1:.4f}: "
        f"{checkpoint_path}"
    )


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pools = ("avg", "max") if args.pool == "both" else (args.pool,)
    for pool in pools:
        train_variant(pool, args, device, out_dir)


if __name__ == "__main__":
    main()
