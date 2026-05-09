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


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class FaceEmotionCNN(nn.Module):
    """
    From-scratch VGG-style CNN for 48x48 grayscale facial emotion classification.

    Four blocks of double-3x3 conv with batch norm. Final feature map is
    3x3 x 512 = 4608 features, pooled and projected through a 512-unit FC
    head to 7 emotion classes.
    """

    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.block1 = _conv_block(1, 64)
        self.block2 = _conv_block(64, 128)
        self.block3 = _conv_block(128, 256)
        self.block4 = _conv_block(256, 512)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_conv = nn.Dropout2d(0.25)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.block1(x))      # 48 -> 24
        x = self.pool(self.block2(x))      # 24 -> 12
        x = self.pool(self.block3(x))      # 12 -> 6
        x = self.pool(self.block4(x))      # 6  -> 3
        x = self.dropout_conv(x)
        x = self.gap(x)                    # 3 -> 1
        return self.classifier(x)
