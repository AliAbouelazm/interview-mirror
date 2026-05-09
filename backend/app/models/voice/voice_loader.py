"""
Audio feature extraction for the voice CNN.

Produces a 128 x 130 mel spectrogram and an 80-dim statistical feature vector
(40 MFCC means, 40 MFCC stds, plus 6 acoustic descriptors padded to 80).
"""
from __future__ import annotations

import numpy as np
import librosa

from app.models.voice.architecture import (
    HOP_LENGTH,
    MEL_FRAMES,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
    STAT_FEATURES,
    WINDOW_SECONDS,
)

TARGET_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)


def pad_or_truncate(audio: np.ndarray, length: int = TARGET_SAMPLES) -> np.ndarray:
    if audio.shape[0] >= length:
        return audio[:length]
    out = np.zeros(length, dtype=np.float32)
    out[: audio.shape[0]] = audio
    return out


def mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=N_FFT,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    if log_mel.shape[1] < MEL_FRAMES:
        pad = MEL_FRAMES - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad)), mode="constant", constant_values=log_mel.min())
    elif log_mel.shape[1] > MEL_FRAMES:
        log_mel = log_mel[:, :MEL_FRAMES]
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
    return log_mel.astype(np.float32)


def speaking_rate_from_energy(audio: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """Crude speaking rate proxy: peaks per second in RMS energy envelope."""
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
    if rms.size < 3:
        return 0.0
    threshold = rms.mean() + 0.3 * rms.std()
    peaks = (rms[1:-1] > rms[:-2]) & (rms[1:-1] > rms[2:]) & (rms[1:-1] > threshold)
    duration = max(audio.shape[0] / sr, 1e-6)
    return float(peaks.sum() / duration)


def statistical_features(audio: np.ndarray) -> np.ndarray:
    """80-dim feature vector. 40 MFCC mean + 40 MFCC std, then 6 descriptors zero-padded if needed."""
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=40)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    centroid = librosa.feature.spectral_centroid(y=audio, sr=SAMPLE_RATE)[0]
    rms = librosa.feature.rms(y=audio)[0]

    descriptors = np.array([
        zcr.mean(), zcr.std(),
        centroid.mean() / SAMPLE_RATE, centroid.std() / SAMPLE_RATE,
        rms.mean(), rms.std(),
    ], dtype=np.float32)

    feats = np.concatenate([mfcc_mean, mfcc_std, descriptors]).astype(np.float32)
    if feats.shape[0] < STAT_FEATURES:
        feats = np.pad(feats, (0, STAT_FEATURES - feats.shape[0]))
    elif feats.shape[0] > STAT_FEATURES:
        feats = feats[:STAT_FEATURES]
    # Per-sample z-score keeps the FC head's input distribution stable across clips.
    feats = (feats - feats.mean()) / (feats.std() + 1e-6)
    return feats


def extract_features(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Resample, pad, and return (mel, stats, descriptors_dict) for prediction."""
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
    audio = audio.astype(np.float32)
    audio = pad_or_truncate(audio)

    mel = mel_spectrogram(audio)
    stats = statistical_features(audio)

    rms = librosa.feature.rms(y=audio)[0]
    descriptors = {
        "speaking_rate": speaking_rate_from_energy(audio),
        "energy_level": float(rms.mean()),
        "rms_std": float(rms.std()),
    }
    return mel, stats, descriptors


def load_and_extract(path: str) -> tuple[np.ndarray, np.ndarray, dict]:
    audio, sr = librosa.load(path, sr=None, mono=True)
    return extract_features(audio, sr)
