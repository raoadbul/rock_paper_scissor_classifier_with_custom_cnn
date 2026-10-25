import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import RPSNet


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_args():
    p = argparse.ArgumentParser(description="Train a custom Rock-Paper-Scissors CNN from scratch.")
    p.add_argument("--data", type=str, default="rps", help="ImageFolder root containing rock/paper/scissor folders")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--project", type=str, default="rps-custom-cnn")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--no-wandb", action="store_true", help="Optional local-only mode; assignment submission should use W&B.")
    return p.parse_args()


def main():
    args = get_args()
    seed_everything(args.seed)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data)

    if not data_root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_root.resolve()}")

    train_tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=12),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    # ImageFolder recursively discovers images inside each top-level class folder,
    # including nested source/person directories.
    # scan_ds = datasets.ImageFolder(data_root)
    class_dirs = [
        d for d in data_root.iterdir()
        if d.is_dir() and d.name != "__pycache__"
    ]

    # Scan only actual class folders

    print("Dataset root:", data_root)
    print("Directories:", [p.name for p in Path(data_root).iterdir()
                        if p.is_dir()])
    scan_ds = datasets.ImageFolder(
        root=str(data_root),
        allow_empty=False
    )
    class_names = scan_ds.classes
    if len(class_names) != 3:
        raise ValueError(f"Expected exactly 3 class folders, found: {class_names}")
    expected = {"rock", "paper", "scissor"}
    if set(class_names) != expected:
        raise ValueError(f"Expected class folders {sorted(expected)}, found {class_names}")

    targets = np.array(scan_ds.targets)
    counts = {name: int((targets == i).sum()) for i, name in enumerate(class_names)}
    print("Device:", device)
    print("Class mapping:", scan_ds.class_to_idx)
    print("Image counts:", counts)
    print("Total images:", len(scan_ds))

    if any(v < 500 for v in counts.values()) or len(scan_ds) < 2000:
        print("\nDATASET REQUIREMENT WARNING: assignment asks for >=500 images/class and >=2000 total.")
        print("Add genuine images you collect; do not duplicate images to inflate the count.")
    if len(set(counts.values())) == 1:
        print("CLASS IMBALANCE WARNING: current class counts are equal; assignment asks for class imbalance.")

    # 70/15/15 stratified split. The test split is held out until final evaluation.
    all_indices = np.arange(len(scan_ds))
    train_idx, temp_idx = train_test_split(
        all_indices, test_size=0.30, random_state=args.seed, stratify=targets
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=args.seed, stratify=targets[temp_idx]
    )

    # Separate ImageFolder instances permit different transforms on each split.
    train_ds_full = datasets.ImageFolder(data_root, transform=train_tf)
    eval_ds_full = datasets.ImageFolder(data_root, transform=eval_tf)
    train_ds = Subset(train_ds_full, train_idx.tolist())
    val_ds = Subset(eval_ds_full, val_idx.tolist())
    test_ds = Subset(eval_ds_full, test_idx.tolist())

    workers = max(0, args.workers)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=workers, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=workers, pin_memory=(device.type == "cuda"))

    # Inverse-frequency class weights help training with an imbalanced dataset.
    train_targets = targets[train_idx]
    class_counts_train = np.bincount(train_targets, minlength=len(class_names))
    class_weights = len(train_targets) / (len(class_names) * np.maximum(class_counts_train, 1))
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )

    model = RPSNet(num_classes=len(class_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.project,
                name=args.run_name,
                config={
                    "architecture": "custom RPSNet CNN from scratch",
                    "pretrained": False,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "image_size": args.image_size,
                    "learning_rate": args.lr,
                    "seed": args.seed,
                    "split": "70/15/15 stratified train/validation/test",
                    "class_counts": counts,
                    "class_weights": class_weights.tolist(),
                    "classes": class_names,
                },
            )
            wandb.watch(model, log="all", log_freq=100)
        except Exception as exc:
            raise RuntimeError(
                "W&B initialization failed. Run `wandb login` and try again, "
                "or use --no-wandb only for local debugging."
            ) from exc

    out_dir = Path("checkpoints")
    out_dir.mkdir(exist_ok=True)
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    best_path = out_dir / "rps_custom_cnn.pth"

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, correct, seen = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            seen += labels.size(0)

        train_loss = running_loss / max(seen, 1)
        train_acc = correct / max(seen, 1)

        model.eval()
        val_loss_sum, val_correct, val_seen = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss_sum += loss.item() * labels.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_seen += labels.size(0)

        val_loss = val_loss_sum / max(val_seen, 1)
        val_acc = val_correct / max(val_seen, 1)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.2%} | "
            f"val loss {val_loss:.4f} acc {val_acc:.2%}"
        )

        if wandb_run is not None:
            import wandb
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "validation/loss": val_loss,
                "validation/accuracy": val_acc,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }, step=epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "image_size": args.image_size,
                "normalization_mean": [0.5, 0.5, 0.5],
                "normalization_std": [0.5, 0.5, 0.5],
                "best_epoch": best_epoch,
                "val_loss": best_val_loss,
            }, best_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping.")
                break

    # Plot train/validation curves.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"], label="Validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(results_dir / "training_curves.png", dpi=150)
    plt.close(fig)

    # Reload best validation checkpoint, then evaluate test set once.
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images.to(device))
            y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
            y_true.extend(labels.numpy().tolist())

    test_acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))),
        target_names=class_names, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))),
        target_names=class_names, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    print(f"\nBest validation epoch: {best_epoch}")
    print(f"Held-out test accuracy: {test_acc:.2%}")
    print(report_text)
    print("Confusion matrix (rows=true, columns=predicted):\n", cm)

    disp = __import__("sklearn.metrics", fromlist=["ConfusionMatrixDisplay"]).ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=class_names
    )
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Held-out Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    with open(results_dir / "test_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "test_accuracy": test_acc,
            "best_epoch": best_epoch,
            "class_counts": counts,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "class_names": class_names,
        }, f, indent=2)

    if wandb_run is not None:
        import wandb
        wandb.log({
            "test/accuracy": test_acc,
            "test/confusion_matrix": wandb.plot.confusion_matrix(
                y_true=y_true, preds=y_pred, class_names=class_names
            ),
        })
        wandb.save(str(best_path))
        wandb.save(str(results_dir / "training_curves.png"))
        wandb.save(str(results_dir / "confusion_matrix.png"))
        wandb.finish()

    print(f"\nBest model saved: {best_path}")
    print("Results saved in:", results_dir.resolve())


if __name__ == "__main__":
    main()
