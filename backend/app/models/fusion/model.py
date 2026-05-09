"""
Fusion network. Combines face softmax (7), voice softmax (8), acoustic descriptors (3),
and language descriptors (3) into confidence and engagement scores in [0, 100].

Synthetic training is valid here because the fusion model is learning to combine
signals that already carry semantic meaning from the trained face/voice base models.
It is not learning low-level features from raw audio or pixels. The targets are a
deterministic mapping from labelled signal patterns ("looks engaged + sounds confident
+ low filler" -> high confidence) plus gaussian noise.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


FUSION_INPUT_DIM = 21
FACE_DIM = 7
VOICE_DIM = 8
ACOUSTIC_DIM = 3
LANGUAGE_DIM = 3


class FusionNet(nn.Module):
    def __init__(self, in_dim: int = FUSION_INPUT_DIM):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, 64)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 32)
        self.head_confidence = nn.Linear(32, 1)
        self.head_engagement = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.norm(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        confidence = torch.sigmoid(self.head_confidence(x)).squeeze(-1) * 100.0
        engagement = torch.sigmoid(self.head_engagement(x)).squeeze(-1) * 100.0
        return confidence, engagement
