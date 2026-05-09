"""
Train face emotion CNN from scratch on FER-2013 (HuggingFace).

Architecture is a 4-block VGG-style network. Trained with Adam + cosine LR,
strong augmentation, MixUp regulariser, and class-balanced cross-entropy.

Usage:
    python -m app.models.face.train

Saves best checkpoint to saved_models/face_best.pt and metrics JSON next to it.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from datasets import load_dataset
from tqdm import tqdm

from app.models.face.architecture import FaceEmotionCNN, EMOTION_CLASSES


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# FER-2013 emotion order from the most common HF mirror.
FER_TO_OURS = {
    0: "angry",
    1: "disgusted",
    2: "fearful",
    3: "happy",
    4: "sad",
    5: "surprised",
    6: "neutral",
}
LABEL_TO_IDX = {name: i for i, name in enumerate(EMOTION_CLASSES)}


def _row_image(row):
    return row.get("image") or row.get("jpg") or row.get("img")


def _row_label(row):
    for key in ("label", "cls", "emotion"):
        if key in row and row[key] is not None:
            return int(row[key])
    return None


class FERDataset(Dataset):
    def __init__(self, hf_split, transform):
        self.records = []
        for row in hf_split:
            label_id = _row_label(row)
            if label_id is None:
                continue
            our_name = FER_TO_OURS.get(label_id)
            if our_name is None:
                continue
            self.records.append({"image": _row_image(row), "label": LABEL_TO_IDX[our_name]})
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = rec["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        img = img.convert("L")
        img = self.transform(img)
        return img, rec["label"]


def load_fer_splits():
    """Load FER-2013 from HuggingFace. Falls back across candidate dataset names."""
    candidates = [
        "clip-benchmark/wds_fer2013",
        "Jeneral/fer2013",
        "CaptainHaaz/FER2013",
    ]
    last_err = None
    for name in candidates:
        try:
            print(f"Trying dataset: {name}")
            ds = load_dataset(name)
            print(f"Loaded {name}")
            return ds, name
        except Exception as e:
            last_err = e
            print(f"  failed: {type(e).__name__}: {str(e)[:200]}")
    raise RuntimeError(f"Could not load FER-2013 from any candidate: {last_err}")


def _ensure_label_column(split):
    """Cast cls/emotion column to label as a ClassLabel so train_test_split can stratify."""
    cols = split.column_names
    if "label" not in cols:
        for source in ("cls", "emotion"):
            if source in cols:
                split = split.rename_column(source, "label")
                break
    feat = split.features.get("label")
    if feat is not None and feat.__class__.__name__ != "ClassLabel":
        split = split.class_encode_column("label")
    return split


def split_dataset(ds_dict):
    """Build train/val/test (70/15/15 stratified) from whatever splits the dataset provides."""
    if "train" in ds_dict and ("test" in ds_dict or "validation" in ds_dict):
        train = _ensure_label_column(ds_dict["train"])
        if "validation" in ds_dict and "test" in ds_dict:
            return train, _ensure_label_column(ds_dict["validation"]), _ensure_label_column(ds_dict["test"])
        other = ds_dict.get("test") or ds_dict.get("validation")
        other = _ensure_label_column(other)
        split = other.train_test_split(test_size=0.5, seed=42, stratify_by_column="label")
        return train, split["train"], split["test"]

    full = ds_dict["train"] if "train" in ds_dict else list(ds_dict.values())[0]
    full = _ensure_label_column(full)
    a = full.train_test_split(test_size=0.30, seed=42, stratify_by_column="label")
    b = a["test"].train_test_split(test_size=0.50, seed=42, stratify_by_column="label")
    return a["train"], b["train"], b["test"]


def _mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    """Standard MixUp. Returns mixed x, two label tensors, and the lambda weight."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
    perm = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[perm]
    return x_mix, y, y[perm], float(lam)


def train_one_epoch(model, loader, criterion, optimizer, device, mixup_alpha: float = 0.2):
    model.train()
    total_loss = 0.0
    n = 0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        x, ya, yb, lam = _mixup(images, labels, alpha=mixup_alpha)
        optimizer.zero_grad()
        logits = model(x)
        loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n = 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels_dev = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels_dev)
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())
    return {
        "loss": total_loss / max(n, 1),
        "f1_macro": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "accuracy": accuracy_score(all_labels, all_preds),
        "preds": all_preds,
        "labels": all_labels,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--out", type=str, default="saved_models/face_best.pt")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    ds_dict, ds_name = load_fer_splits()
    train_raw, val_raw, test_raw = split_dataset(ds_dict)

    # Stronger augmentation: random crops, flips, affine, contrast jitter, random erasing.
    train_tf = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.25, contrast=0.25),
        transforms.RandomCrop(48, padding=4, padding_mode="reflect"),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    train_set = FERDataset(train_raw, train_tf)
    val_set = FERDataset(val_raw, eval_tf)
    test_set = FERDataset(test_raw, eval_tf)
    print(f"Splits: train={len(train_set)} val={len(val_set)} test={len(test_set)}")

    counts = np.bincount([r["label"] for r in train_set.records], minlength=len(EMOTION_CLASSES))
    print("Class distribution (train):")
    for i, name in enumerate(EMOTION_CLASSES):
        print(f"  {name:>10s}: {counts[i]}")

    pin_mem = device.type == "cuda"
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin_mem)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin_mem)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=pin_mem)

    model = FaceEmotionCNN(num_classes=len(EMOTION_CLASSES)).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    train_labels = [r["label"] for r in train_set.records]
    cw = compute_class_weight("balanced", classes=np.arange(len(EMOTION_CLASSES)), y=train_labels)
    weight_tensor = torch.tensor(cw, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = out_path.with_suffix(".metrics.json")

    best_f1 = -1.0
    best_epoch = -1
    epochs_since_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device,
                                     mixup_alpha=args.mixup_alpha)
        val = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        epoch_time = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val['loss']:.4f} val_f1={val['f1_macro']:.4f} "
            f"val_acc={val['accuracy']:.4f} time={epoch_time:.1f}s"
        )
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val["loss"], "val_f1": val["f1_macro"], "val_acc": val["accuracy"],
        })

        if val["f1_macro"] > best_f1:
            best_f1 = val["f1_macro"]
            best_epoch = epoch
            epochs_since_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "classes": EMOTION_CLASSES,
                "val_f1": best_f1,
                "epoch": epoch,
            }, out_path)
            print(f"  saved (val_f1={best_f1:.4f})")
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= args.patience:
                print(f"Early stopping at epoch {epoch} (best_epoch={best_epoch}, best_f1={best_f1:.4f})")
                break

    state = torch.load(out_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    test = evaluate(model, test_loader, criterion, device)
    report = classification_report(test["labels"], test["preds"], target_names=EMOTION_CLASSES,
                                   zero_division=0, output_dict=True)
    print("\nTest results:")
    print(f"  f1_macro: {test['f1_macro']:.4f}")
    print(f"  accuracy: {test['accuracy']:.4f}")
    print(classification_report(test["labels"], test["preds"], target_names=EMOTION_CLASSES, zero_division=0))

    metrics = {
        "dataset": ds_name,
        "best_epoch": best_epoch,
        "val_f1": best_f1,
        "test_f1_macro": test["f1_macro"],
        "test_accuracy": test["accuracy"],
        "test_per_class": report,
        "history": history,
        "classes": EMOTION_CLASSES,
        "params": int(sum(p.numel() for p in model.parameters())),
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
