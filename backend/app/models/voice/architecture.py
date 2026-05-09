import torch
import torch.nn as nn
import torch.nn.functional as F


VOICE_CLASSES = [
    "neutral", "calm", "happy", "sad", "angry", "fearful", "disgusted", "surprised",
]

VOICE_SIGNAL_MAP = {
    "neutral": "confident",
    "calm": "confident",
    "happy": "confident",
    "fearful": "nervous",
    "sad": "nervous",
    "angry": "tense",
    "disgusted": "tense",
    "surprised": "reactive",
}

# Sample rate, mel params, window length used in voice_loader.py.
SAMPLE_RATE = 22050
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
WINDOW_SECONDS = 3.0
MEL_FRAMES = 130
STAT_FEATURES = 80


class VoiceEmotionCNN(nn.Module):
    """Mel spectrogram CNN with auxiliary statistical features fused at the FC stage."""

    def __init__(self, num_classes: int = 8, stat_features: int = STAT_FEATURES):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_conv = nn.Dropout(0.3)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(128 + stat_features, 256)
        self.dropout_fc = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, mel: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(mel))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout_conv(x)
        x = self.gap(x).flatten(1)
        x = torch.cat([x, stats], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        return self.fc2(x)
