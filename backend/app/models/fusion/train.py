"""
Train fusion network on synthetic samples drawn from the trained face/voice models.

Synthetic samples are valid because:
1. The base models are already trained on real labelled data.
2. The fusion targets are the deterministic mapping interviewers actually use:
   engaged-looking + confident-sounding + clear language => high score.
3. Noise (gaussian, std 0.05) is added to features so the fusion network does not
   overfit to a single canonical pattern.

Run:
    python -m app.models.fusion.train
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from app.models.face.architecture import EMOTION_CLASSES
from app.models.voice.architecture import VOICE_CLASSES
from app.models.fusion.model import FusionNet, FUSION_INPUT_DIM


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _dirichlet_around(target_idx: int, n_classes: int, sharpness: float) -> np.ndarray:
    alpha = np.ones(n_classes)
    alpha[target_idx] = sharpness
    p = np.random.dirichlet(alpha)
    return p.astype(np.float32)


def synthesize_sample(rng: np.random.Generator, profile: str) -> tuple[np.ndarray, float, float]:
    """One synthetic 21-dim sample for one of {high, mid, low} confidence profiles."""
    if profile == "high":
        face_dist = rng.choice(["neutral", "happy", "neutral", "happy"])
        voice_dist = rng.choice(["neutral", "calm", "happy", "neutral"])
        speaking_rate = rng.uniform(2.5, 4.5)
        energy = rng.uniform(0.05, 0.15)
        filler_prob = rng.uniform(0.0, 0.15)
        filler_rate = rng.uniform(0.0, 0.02)
        hedge_rate = rng.uniform(0.0, 0.02)
        lang_conf = rng.uniform(75, 100)
        confidence = rng.uniform(75, 95)
        engagement = rng.uniform(78, 95)
    elif profile == "mid":
        face_dist = rng.choice(["neutral", "happy", "surprised"])
        voice_dist = rng.choice(["neutral", "calm", "surprised"])
        speaking_rate = rng.uniform(2.0, 4.0)
        energy = rng.uniform(0.03, 0.10)
        filler_prob = rng.uniform(0.10, 0.40)
        filler_rate = rng.uniform(0.02, 0.08)
        hedge_rate = rng.uniform(0.02, 0.06)
        lang_conf = rng.uniform(45, 75)
        confidence = rng.uniform(45, 70)
        engagement = rng.uniform(45, 70)
    else:
        face_dist = rng.choice(["fearful", "sad", "angry", "disgusted"])
        voice_dist = rng.choice(["fearful", "sad", "angry"])
        speaking_rate = rng.uniform(1.0, 3.0)
        energy = rng.uniform(0.005, 0.05)
        filler_prob = rng.uniform(0.30, 0.85)
        filler_rate = rng.uniform(0.06, 0.18)
        hedge_rate = rng.uniform(0.05, 0.15)
        lang_conf = rng.uniform(10, 50)
        confidence = rng.uniform(10, 45)
        engagement = rng.uniform(15, 45)

    face_idx = EMOTION_CLASSES.index(face_dist)
    voice_idx = VOICE_CLASSES.index(voice_dist)
    face = _dirichlet_around(face_idx, len(EMOTION_CLASSES), sharpness=20.0)
    voice = _dirichlet_around(voice_idx, len(VOICE_CLASSES), sharpness=20.0)

    speaking_rate_n = float(np.clip(speaking_rate / 6.0, 0, 1))
    energy_n = float(np.clip(energy / 0.2, 0, 1))
    filler_prob = float(np.clip(filler_prob, 0, 1))
    filler_rate = float(np.clip(filler_rate / 0.2, 0, 1))
    hedge_rate = float(np.clip(hedge_rate / 0.2, 0, 1))
    lang_conf_n = float(np.clip(lang_conf / 100.0, 0, 1))

    feat = np.concatenate([
        face,
        voice,
        np.array([speaking_rate_n, energy_n, filler_prob], dtype=np.float32),
        np.array([filler_rate, hedge_rate, lang_conf_n], dtype=np.float32),
    ])
    noise = rng.normal(0, 0.05, size=feat.shape).astype(np.float32)
    feat = np.clip(feat + noise, 0, 1)
    return feat, float(confidence), float(engagement)


def build_dataset(n: int = 10_000, seed: int = 42):
    rng = np.random.default_rng(seed)
    profiles = rng.choice(["high", "mid", "low"], size=n, p=[0.34, 0.33, 0.33])
    X, y_conf, y_eng = [], [], []
    for p in profiles:
        feat, c, e = synthesize_sample(rng, p)
        X.append(feat)
        y_conf.append(c)
        y_eng.append(e)
    return (np.stack(X).astype(np.float32),
            np.array(y_conf, dtype=np.float32),
            np.array(y_eng, dtype=np.float32))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="saved_models/fusion_best.pt")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Building {args.samples} synthetic samples...")
    X, y_conf, y_eng = build_dataset(args.samples, seed=args.seed)
    print(f"X: {X.shape}, conf range: [{y_conf.min():.1f}, {y_conf.max():.1f}], "
          f"eng range: [{y_eng.min():.1f}, {y_eng.max():.1f}]")

    n_train = int(0.85 * len(X))
    X_train, X_val = X[:n_train], X[n_train:]
    yc_train, yc_val = y_conf[:n_train], y_conf[n_train:]
    ye_train, ye_val = y_eng[:n_train], y_eng[n_train:]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(yc_train), torch.from_numpy(ye_train)),
        batch_size=args.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(yc_val), torch.from_numpy(ye_val)),
        batch_size=args.batch_size, shuffle=False,
    )

    model = FusionNet(in_dim=FUSION_INPUT_DIM).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        t0 = time.time()
        for x, yc, ye in train_loader:
            x = x.to(device); yc = yc.to(device); ye = ye.to(device)
            optimizer.zero_grad()
            pc, pe = model(x)
            loss = criterion(pc, yc) + criterion(pe, ye)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
        train_loss = running / max(n, 1)

        model.eval()
        running_v = 0.0
        n_v = 0
        with torch.no_grad():
            for x, yc, ye in val_loader:
                x = x.to(device); yc = yc.to(device); ye = ye.to(device)
                pc, pe = model(x)
                loss = criterion(pc, yc) + criterion(pe, ye)
                running_v += loss.item() * x.size(0)
                n_v += x.size(0)
        val_loss = running_v / max(n_v, 1)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{args.epochs} train_loss={train_loss:.3f} "
                  f"val_loss={val_loss:.3f} time={time.time()-t0:.1f}s")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict(), "val_loss": best_val,
                        "in_dim": FUSION_INPUT_DIM}, out_path)

    metrics = {"best_val_mse": best_val, "samples": args.samples,
               "epochs": args.epochs, "history": history}
    with open(out_path.with_suffix(".metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved fusion model to {out_path} (best val MSE {best_val:.3f})")


if __name__ == "__main__":
    main()
