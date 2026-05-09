"""
Train voice emotion CNN from scratch on a combined RAVDESS + CREMA-D corpus.

RAVDESS provides 1,440 clips across 8 emotions (8 actors, 4 statements each).
CREMA-D provides 7,442 clips across 6 emotions (no calm or surprised). Combining
the two gives roughly 9k samples while still covering all 8 of our target classes.

Augmentation includes gaussian noise, time-shift, and SpecAugment-style time +
frequency masking.

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
    return LABEL_TO_IDX.get(s)


def parse_ravdess_label(row) -> int | None:
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
    return None


def parse_cremad_label(row) -> int | None:
    """CREMA-D's `major_emotion` strings: anger, sadness, disgust, fear, neutral, happy."""
    name = row.get("major_emotion")
    if not name:
        return None
    return _resolve_string_label(name)


def load_ravdess() -> list[dict]:
    candidates = ["xbgoose/ravdess", "narad/ravdess", "Codec-SUPERB/RAVDESS"]
    last_err = None
    for name in candidates:
        try:
            print(f"  trying RAVDESS: {name}")
            ds = load_dataset(name)
            split = ds["train"] if "train" in ds else list(ds.values())[0]
            split = split.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
            out = []
            for row in split:
                label = parse_ravdess_label(row)
                if label is None:
                    continue
                out.append({
                    "audio": np.asarray(row["audio"]["array"], dtype=np.float32),
                    "label": label,
                    "source": "ravdess",
                })
            print(f"  RAVDESS loaded ({len(out)} samples)")
            return out
        except Exception as e:
            last_err = e
            print(f"    failed: {type(e).__name__}: {str(e)[:200]}")
    raise RuntimeError(f"Could not load RAVDESS: {last_err}")


def load_cremad() -> list[dict]:
    candidates = ["AbstractTTS/CREMA-D"]
    last_err = None
    for name in candidates:
        try:
            print(f"  trying CREMA-D: {name}")
            ds = load_dataset(name)
            split = ds["train"] if "train" in ds else list(ds.values())[0]
            split = split.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
            out = []
            for row in split:
                label = parse_cremad_label(row)
                if label is None:
                    continue
                out.append({
                    "audio": np.asarray(row["audio"]["array"], dtype=np.float32),
                    "label": label,
                    "source": "cremad",
                })
            print(f"  CREMA-D loaded ({len(out)} samples)")
            return out
        except Exception as e:
            last_err = e
            print(f"    failed: {type(e).__name__}: {str(e)[:200]}")
    print(f"  CREMA-D unavailable, continuing without it: {last_err}")
    return []


def stratified_split(records: list[dict], train_frac: float, val_frac: float, seed: int = 42):
    """Stratified train/val/test split by label with reproducible permutation."""
    rng = np.random.default_rng(seed)
    by_label: dict[int, list[int]] = {}
    for i, r in enumerate(records):
        by_label.setdefault(r["label"], []).append(i)
    train_idx, val_idx, test_idx = [], [], []
    for label, idxs in by_label.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = int(round(train_frac * n))
        n_val = int(round(val_frac * n))
        train_idx.extend(idxs[:n_train])
        val_idx.extend(idxs[n_train:n_train + n_val])
        test_idx.extend(idxs[n_train + n_val:])
    return [records[i] for i in train_idx], [records[i] for i in val_idx], [records[i] for i in test_idx]


def spec_augment(mel: np.ndarray, freq_mask: int = 18, time_mask: int = 25, n_masks: int = 2) -> np.ndarray:
    """Apply SpecAugment-style time and frequency masks in place."""
    out = mel.copy()
    n_freq, n_time = out.shape
    for _ in range(n_masks):
        if freq_mask > 0:
            f = np.random.randint(0, freq_mask + 1)
            f0 = np.random.randint(0, max(1, n_freq - f))
            out[f0:f0 + f, :] = out.min()
        if time_mask > 0:
            t = np.random.randint(0, time_mask + 1)
            t0 = np.random.randint(0, max(1, n_time - t))
            out[:, t0:t0 + t] = out.min()
    return out


class CombinedVoiceDataset(Dataset):
    """
    Pre-computes mel + stats features once (CPU bound), then serves cached tensors.
    Augmentation happens at the mel-tensor level (SpecAugment) so we keep variation
    across epochs without recomputing librosa features each time.
    """

    def __init__(self, records: list[dict], augment: bool = False):
        self.augment = augment
        self.labels: list[int] = []
        self.mels: list[np.ndarray] = []
        self.stats: list[np.ndarray] = []
        for rec in tqdm(records, desc="extract", leave=False):
            audio = rec["audio"]
            mel, stats, _ = extract_features(audio, SAMPLE_RATE)
            self.mels.append(mel.astype(np.float32))
            self.stats.append(stats.astype(np.float32))
            self.labels.append(rec["label"])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mel = self.mels[idx]
        stats = self.stats[idx]
        if self.augment:
            mel = spec_augment(mel)
        return (
            torch.from_numpy(mel).unsqueeze(0),
            torch.from_numpy(stats),
            self.labels[idx],
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
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--out", type=str, default="saved_models/voice_best.pt")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    print("Loading datasets...")
    records = []
    records += load_ravdess()
    records += load_cremad()
    if not records:
        raise RuntimeError("No voice data loaded.")
    print(f"Total samples after combination: {len(records)}")

    train_recs, val_recs, test_recs = stratified_split(records, 0.70, 0.15)
    print(f"Splits: train={len(train_recs)} val={len(val_recs)} test={len(test_recs)}")

    counts_by_class = np.bincount([r["label"] for r in train_recs], minlength=len(VOICE_CLASSES))
    counts_by_source: dict[str, int] = {}
    for r in train_recs:
        counts_by_source[r["source"]] = counts_by_source.get(r["source"], 0) + 1
    print("Class distribution (train):")
    for i, name in enumerate(VOICE_CLASSES):
        print(f"  {name:>10s}: {counts_by_class[i]}")
    print(f"Sources (train): {counts_by_source}")

    train_set = CombinedVoiceDataset(train_recs, augment=True)
    val_set = CombinedVoiceDataset(val_recs, augment=False)
    test_set = CombinedVoiceDataset(test_recs, augment=False)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    model = VoiceEmotionCNN(num_classes=len(VOICE_CLASSES), stat_features=STAT_FEATURES).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    train_labels_arr = np.array([r["label"] for r in train_recs])
    cw = compute_class_weight("balanced", classes=np.arange(len(VOICE_CLASSES)), y=train_labels_arr)
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
                "datasets": list(counts_by_source.keys()),
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
        "datasets": ["RAVDESS", "CREMA-D"],
        "best_epoch": best_epoch,
        "val_f1": best_f1,
        "test_f1_macro": test["f1_macro"],
        "test_accuracy": test["accuracy"],
        "test_per_class": classification_report(test["labels"], test["preds"],
                                                target_names=VOICE_CLASSES,
                                                zero_division=0, output_dict=True),
        "history": history,
        "classes": VOICE_CLASSES,
        "params": int(sum(p.numel() for p in model.parameters())),
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
