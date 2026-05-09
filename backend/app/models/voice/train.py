"""
Train voice emotion CNN on RAVDESS (HuggingFace).

Usage:
    python -m app.models.voice.train

Saves best checkpoint to saved_models/voice_best.pt.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from datasets import load_dataset, Audio
from tqdm import tqdm

from app.models.voice.architecture import (
    SAMPLE_RATE,
    STAT_FEATURES,
    VOICE_CLASSES,
    VoiceEmotionCNN,
)
from app.models.voice.voice_loader import extract_features


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# RAVDESS file naming and HF mirrors use this canonical 8-emotion label order.
RAVDESS_TO_OURS = {
    1: "neutral", 2: "calm", 3: "happy", 4: "sad",
    5: "angry", 6: "fearful", 7: "disgusted", 8: "surprised",
}
LABEL_TO_IDX = {n: i for i, n in enumerate(VOICE_CLASSES)}


def load_ravdess():
    """Try several RAVDESS mirrors. All are emotion-labelled with the canonical 1-8 mapping."""
    candidates = [
        "narad/ravdess",
        "xbgoose/ravdess",
        "Codec-SUPERB/RAVDESS",
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
            print(f"  failed: {type(e).__name__}: {e}")
    raise RuntimeError(f"Could not load RAVDESS from any candidate: {last_err}")


def split_dataset(ds_dict):
    train = ds_dict["train"] if "train" in ds_dict else list(ds_dict.values())[0]
    a = train.train_test_split(test_size=0.30, seed=42, stratify_by_column="label" if "label" in train.column_names else None)
    b = a["test"].train_test_split(test_size=0.50, seed=42, stratify_by_column="label" if "label" in a["test"].column_names else None)
    return a["train"], b["train"], b["test"]


STRING_LABEL_ALIASES = {
    "neutral": "neutral",
    "calm": "calm",
    "happy": "happy", "happiness": "happy",
    "sad": "sad", "sadness": "sad",
    "angry": "angry", "anger": "angry",
    "fearful": "fearful", "fear": "fearful",
    "disgusted": "disgusted", "disgust": "disgusted",
    "surprised": "surprised", "surprise": "surprised",
}


def _resolve_string_label(s: str) -> int | None:
    s = s.strip().lower()
    if s in STRING_LABEL_ALIASES:
        return LABEL_TO_IDX.get(STRING_LABEL_ALIASES[s])
    if s in LABEL_TO_IDX:
        return LABEL_TO_IDX[s]
    return None


def parse_label(row) -> int | None:
    """RAVDESS HF rows usually have either an explicit label, an emotion id, or a parseable filename."""
    if "label" in row and row["label"] is not None:
        v = row["label"]
        if isinstance(v, str):
            return _resolve_string_label(v)
        n = int(v)
        ravdess_id = n + 1 if 0 <= n <= 7 else n
        name = RAVDESS_TO_OURS.get(ravdess_id)
        return LABEL_TO_IDX.get(name) if name else None
    if "emotion" in row and row["emotion"] is not None:
        e = row["emotion"]
        if isinstance(e, str):
            return _resolve_string_label(e)
        return LABEL_TO_IDX.get(RAVDESS_TO_OURS.get(int(e)))
    path = row.get("audio", {}).get("path") if isinstance(row.get("audio"), dict) else None
    if path:
        try:
            stem = Path(path).stem
            parts = stem.split("-")
            ravdess_id = int(parts[2])
            return LABEL_TO_IDX.get(RAVDESS_TO_OURS.get(ravdess_id))
        except (IndexError, ValueError):
            return None
    return None


class RAVDESSDataset(Dataset):
    def __init__(self, hf_split, augment: bool = False):
        ds = hf_split.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
        self.records = []
        for row in ds:
            label_idx = parse_label(row)
            if label_idx is None:
                continue
            audio = row["audio"]["array"]
            self.records.append({"audio": np.asarray(audio, dtype=np.float32), "label": label_idx})
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        audio = rec["audio"]
        if self.augment:
            audio = audio + np.random.normal(0, 0.005, size=audio.shape).astype(np.float32)
            shift = np.random.randint(-int(0.1 * SAMPLE_RATE), int(0.1 * SAMPLE_RATE))
            audio = np.roll(audio, shift)
        mel, stats, _ = extract_features(audio, SAMPLE_RATE)
        return (
            torch.from_numpy(mel).unsqueeze(0),
            torch.from_numpy(stats),
            rec["label"],
        )


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    for mel, stats, labels in tqdm(loader, desc="train", leave=False):
        mel = mel.to(device, non_blocking=True)
        stats = stats.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(mel, stats)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * mel.size(0)
        n += mel.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n = 0
    all_preds, all_labels = [], []
    for mel, stats, labels in loader:
        mel = mel.to(device, non_blocking=True)
        stats = stats.to(device, non_blocking=True)
        labels_dev = labels.to(device, non_blocking=True)
        logits = model(mel, stats)
        loss = criterion(logits, labels_dev)
        total_loss += loss.item() * mel.size(0)
        n += mel.size(0)
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
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--out", type=str, default="saved_models/voice_best.pt")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    ds_dict, ds_name = load_ravdess()
    train_raw, val_raw, test_raw = split_dataset(ds_dict)

    print("Building feature cache (this can take a few minutes)...")
    train_set = RAVDESSDataset(train_raw, augment=False)
    val_set = RAVDESSDataset(val_raw, augment=False)
    test_set = RAVDESSDataset(test_raw, augment=False)
    print(f"Splits: train={len(train_set)} val={len(val_set)} test={len(test_set)}")

    train_labels_arr = np.array([r["label"] for r in train_set.records])
    counts = np.bincount(train_labels_arr, minlength=len(VOICE_CLASSES))
    print("Class distribution (train):")
    for i, name in enumerate(VOICE_CLASSES):
        print(f"  {name:>10s}: {counts[i]}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    model = VoiceEmotionCNN(num_classes=len(VOICE_CLASSES), stat_features=STAT_FEATURES).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    cw = compute_class_weight("balanced", classes=np.arange(len(VOICE_CLASSES)), y=train_labels_arr)
    weight_tensor = torch.tensor(cw, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
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
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        et = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val['loss']:.4f} "
            f"val_f1={val['f1_macro']:.4f} val_acc={val['accuracy']:.4f} time={et:.1f}s"
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
                "classes": VOICE_CLASSES,
                "stat_features": STAT_FEATURES,
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
    print("\nTest results:")
    print(f"  f1_macro: {test['f1_macro']:.4f}")
    print(f"  accuracy: {test['accuracy']:.4f}")
    print(classification_report(test["labels"], test["preds"], target_names=VOICE_CLASSES, zero_division=0))

    metrics = {
        "dataset": ds_name,
        "best_epoch": best_epoch,
        "val_f1": best_f1,
        "test_f1_macro": test["f1_macro"],
        "test_accuracy": test["accuracy"],
        "test_per_class": classification_report(test["labels"], test["preds"],
                                                target_names=VOICE_CLASSES,
                                                zero_division=0, output_dict=True),
        "history": history,
        "classes": VOICE_CLASSES,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
