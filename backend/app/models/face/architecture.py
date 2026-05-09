import torch
import torch.nn as nn
import torch.nn.functional as F


EMOTION_CLASSES = [
    "neutral", "happy", "sad", "angry", "fearful", "surprised", "disgusted",
]

INTERVIEW_SIGNAL_MAP = {
    "neutral": "engaged",
    "happy": "engaged",
    "fearful": "nervous",
    "sad": "nervous",
    "angry": "tense",
    "disgusted": "tense",
    "surprised": "reactive",
}


class FaceEmotionCNN(nn.Module):
    """Lightweight CNN for 48x48 grayscale facial emotion classification."""

    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_conv = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.dropout_fc = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout_conv(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        return self.fc2(x)
