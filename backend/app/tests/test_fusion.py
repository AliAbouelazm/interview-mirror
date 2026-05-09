"""Unit tests for the fusion network and feature builder."""
import numpy as np
import pytest
import torch

from app.models.face.architecture import EMOTION_CLASSES
from app.models.voice.architecture import VOICE_CLASSES
from app.models.fusion.model import FUSION_INPUT_DIM, FusionNet
from app.models.fusion.predict import build_feature_vector
from app.models.fusion.train import build_dataset, synthesize_sample


def test_fusion_input_dim_matches_components():
    assert FUSION_INPUT_DIM == len(EMOTION_CLASSES) + len(VOICE_CLASSES) + 6


def test_fusion_forward_shapes_and_range():
    model = FusionNet(in_dim=FUSION_INPUT_DIM)
    x = torch.zeros(4, FUSION_INPUT_DIM)
    c, e = model(x)
    assert c.shape == (4,)
    assert e.shape == (4,)
    assert torch.all(c >= 0) and torch.all(c <= 100)
    assert torch.all(e >= 0) and torch.all(e <= 100)


def test_fusion_forward_extreme_inputs_clamp():
    model = FusionNet(in_dim=FUSION_INPUT_DIM)
    x_high = torch.full((1, FUSION_INPUT_DIM), 1e3)
    x_low = torch.full((1, FUSION_INPUT_DIM), -1e3)
    c_h, e_h = model(x_high)
    c_l, e_l = model(x_low)
    for v in (c_h, e_h, c_l, e_l):
        assert torch.all(v >= 0) and torch.all(v <= 100)


def test_build_feature_vector_shape_and_clip():
    face_scores = {c: 1.0 / len(EMOTION_CLASSES) for c in EMOTION_CLASSES}
    voice_scores = {c: 1.0 / len(VOICE_CLASSES) for c in VOICE_CLASSES}
    v = build_feature_vector(
        face_scores=face_scores,
        voice_scores=voice_scores,
        speaking_rate=10.0,
        energy_level=2.0,
        filler_probability=2.0,
        filler_rate=2.0,
        hedge_rate=2.0,
        language_confidence=200.0,
    )
    assert v.shape == (FUSION_INPUT_DIM,)
    assert v.min() >= 0
    assert v.max() <= 1


def test_build_feature_vector_handles_missing_keys():
    v = build_feature_vector(
        face_scores={},
        voice_scores={},
        speaking_rate=0.0,
        energy_level=0.0,
        filler_probability=0.0,
        filler_rate=0.0,
        hedge_rate=0.0,
        language_confidence=0.0,
    )
    assert v.shape == (FUSION_INPUT_DIM,)
    assert np.allclose(v, 0)


def test_synthesize_sample_within_bounds():
    rng = np.random.default_rng(0)
    for profile in ("high", "mid", "low"):
        feat, conf, eng = synthesize_sample(rng, profile)
        assert feat.shape == (FUSION_INPUT_DIM,)
        assert 0 <= feat.min() and feat.max() <= 1.001
        assert 0 <= conf <= 100
        assert 0 <= eng <= 100


def test_build_dataset_label_separability():
    X, yc, ye = build_dataset(n=300, seed=0)
    assert X.shape == (300, FUSION_INPUT_DIM)
    high_mean = yc[(yc >= 70)].mean() if (yc >= 70).any() else 0
    low_mean = yc[(yc < 50)].mean() if (yc < 50).any() else 0
    assert high_mean > low_mean
