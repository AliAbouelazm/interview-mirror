"""
Train face emotion CNN from scratch on FER+ relabels of FER-2013.

FER+ is the same 35k images as FER-2013 but with majority-vote labels from 10
human annotators per image, which is meaningfully cleaner than the original
single-annotator FER-2013 labels and is what most modern papers benchmark on.

Architecture is the 4-block VGG-style CNN (~4.8M params). Training adds:
  * MixUp (alpha 0.2) and CutMix (alpha 1.0), 50/50
  * EMA weights (decay 0.999) used for the saved checkpoint
  * Test-time augmentation (centre + horizontal flip, average) at eval
  * Linear warmup for 5 epochs, then cosine annealing
  * Strong augmentation: flip, affine, colour jitter, padded crop, random erase
  * Class-balanced cross-entropy with label smoothing 0.05

Usage:
    python -m app.models.face.train

Saves best checkpoint to saved_models/face_best.pt.
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
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


# FER+ label order from `deanngkl/ferplus-7cls`:
#   anger, disgust, fear, happiness, neutral, sadness, surprise
FERPLUS_NAMES = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
FERPLUS_TO_OURS = {
    "anger":     "angry",
    "disgust":   "disgusted",
    "fear":      "fearful",
    "happiness": "happy",
    "neutral":   "neutral",
    "sadness":   "sad",
    "surprise":  "surprised",
}
LABEL_TO_IDX = {name: i for i, name in enumerate(EMOTION_CLASSES)}
FERPLUS_IDX_TO_OUR_IDX = {
    i: LABEL_TO_IDX[FERPLUS_TO_OURS[name]] for i, name in enumerate(FERPLUS_NAMES)
}


class FERPlusDataset(Dataset):
    def __init__(self, hf_split, transform):
        self.records = []
        for row in hf_split:
            label_id = row.get("label")
            if label_id is None:
                continue
            label_id = int(label_id)
            mapped = FERPLUS_IDX_TO_OUR_IDX.get(label_id)
            if mapped is None:
                continue
            self.records.append({"image": row.get("image"), "label": mapped})
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = rec["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        img = img.convert("L")
        return self.transform(img), rec["label"]


def load_ferplus():
    name = "deanngkl/ferplus-7cls"
    print(f"Loading {name}")
    return load_dataset(name), name


def stratified_split(ds, train_frac: float, val_frac: float, seed: int = 42):
    """Stratified train/val/test split by label."""
    full = ds["train"] if "train" in ds else list(ds.values())[0]
    labels = full["label"]
    rng = np.random.default_rng(seed)
    by_label: dict[int, list[int]] = {}
    for i, l in enumerate(labels):
        by_label.setdefault(int(l), []).append(i)
    train_idx, val_idx, test_idx = [], [], []
    for cls, idxs in by_label.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = int(round(train_frac * n))
        n_val = int(round(val_frac * n))
        train_idx.extend(idxs[:n_train])
        val_idx.extend(idxs[n_train:n_train + n_val])
        test_idx.extend(idxs[n_train + n_val:])
    return full.select(train_idx), full.select(val_idx), full.select(test_idx)


def _mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float):
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1 - lam)
    perm = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[perm], y, y[perm], lam


def _cutmix_batch(x: torch.Tensor, y: torch.Tensor, alpha: float):
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    h, w = x.size(2), x.size(3)
    cut_ratio = math.sqrt(1 - lam)
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)
    cy = np.random.randint(h)
    cx = np.random.randint(w)
    y1 = max(cy - cut_h // 2, 0)
    y2 = min(cy + cut_h // 2, h)
    x1 = max(cx - cut_w // 2, 0)
    x2 = min(cx + cut_w // 2, w)
    x_mix = x.clone()
    x_mix[:, :, y1:y2, x1:x2] = x[perm][:, :, y1:y2, x1:x2]
    lam = 1 - ((y2 - y1) * (x2 - x1) / (h * w))
    return x_mix, y, y[perm], float(lam)


def warmup_cosine_lr(epoch: int, base_lr: float, total_epochs: int, warmup_epochs: int) -> float:
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / max(warmup_epochs, 1)
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    return 0.5 * base_lr * (1 + math.cos(math.pi * progress))


class EMA:
    """Exponential moving average of model parameters."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def apply_to(self, model: nn.Module):
        out = copy.deepcopy(model)
        out.load_state_dict(self.shadow, strict=True)
        return out


def train_one_epoch(model, loader, criterion, optimizer, device, ema, mixup_alpha, cutmix_alpha):
    model.train()
    total_loss = 0.0
    n = 0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if np.random.rand() < 0.5:
            x, ya, yb, lam = _mixup_batch(images, labels, mixup_alpha)
        else:
            x, ya, yb, lam = _cutmix_batch(images, labels, cutmix_alpha)
        optimizer.zero_grad()
        logits = model(x)
        loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
        loss.backward()
        optimizer.step()
        ema.update(model)
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, tta: bool = False):
    model.eval()
    total_loss = 0.0
    n = 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels_dev = labels.to(device, non_blocking=True)
        logits = model(images)
        if tta:
            logits = logits + model(torch.flip(images, dims=(3,)))
            logits = logits / 2
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
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--cutmix-alpha", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--out", type=str, default="saved_models/face_best.pt")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    ds_dict, ds_name = load_ferplus()
    train_raw, val_raw, test_raw = stratified_split(ds_dict, 0.70, 0.15)

    train_tf = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=12, translate=(0.06, 0.06), scale=(0.92, 1.08)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomCrop(48, padding=4, padding_mode="reflect"),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.18)),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    train_set = FERPlusDataset(train_raw, train_tf)
    val_set = FERPlusDataset(val_raw, eval_tf)
    test_set = FERPlusDataset(test_raw, eval_tf)
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
    ema = EMA(model, decay=args.ema_decay)

    train_labels = [r["label"] for r in train_set.records]
    cw = compute_class_weight("balanced", classes=np.arange(len(EMOTION_CLASSES)), y=train_labels)
    weight_tensor = torch.tensor(cw, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = out_path.with_suffix(".metrics.json")

    best_f1 = -1.0
    best_epoch = -1
    epochs_since_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        lr = warmup_cosine_lr(epoch - 1, args.lr, args.epochs, args.warmup_epochs)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device,
                                     ema, args.mixup_alpha, args.cutmix_alpha)
        ema_model = ema.apply_to(model)
        val = evaluate(ema_model, val_loader, criterion, device, tta=False)
        epoch_time = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{args.epochs} lr={lr:.2e} "
            f"train_loss={train_loss:.4f} val_loss={val['loss']:.4f} "
            f"val_f1={val['f1_macro']:.4f} val_acc={val['accuracy']:.4f} time={epoch_time:.1f}s"
        )
        history.append({
            "epoch": epoch, "lr": lr, "train_loss": train_loss,
            "val_loss": val["loss"], "val_f1": val["f1_macro"], "val_acc": val["accuracy"],
        })

        if val["f1_macro"] > best_f1:
            best_f1 = val["f1_macro"]
            best_epoch = epoch
            epochs_since_improve = 0
            torch.save({
                "model_state": ema_model.state_dict(),
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
    eval_model = FaceEmotionCNN(num_classes=len(EMOTION_CLASSES)).to(device)
    eval_model.load_state_dict(state["model_state"])
    test = evaluate(eval_model, test_loader, criterion, device, tta=True)
    report = classification_report(test["labels"], test["preds"], target_names=EMOTION_CLASSES,
                                   zero_division=0, output_dict=True)
    print("\nTest results (TTA enabled):")
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
        "params": int(sum(p.numel() for p in eval_model.parameters())),
        "tta": True,
        "ema_decay": args.ema_decay,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
