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


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class VoiceEmotionCNN(nn.Module):
    """
    From-scratch double-conv CNN over the log-mel spectrogram with auxiliary
    statistical features (MFCC means + stds + acoustic descriptors).

    Four conv blocks (1->64->128->256->512) collapse the (128, 130) input to
    a 512-d global-pooled embedding. The 80-d stat vector is LayerNorm'd
    (batch-independent so it behaves the same in train and eval) and
    concatenated before the classifier head.
    """

    def __init__(self, num_classes: int = 8, stat_features: int = STAT_FEATURES):
        super().__init__()
        self.block1 = _conv_block(1, 64)
        self.block2 = _conv_block(64, 128)
        self.block3 = _conv_block(128, 256)
        self.block4 = _conv_block(256, 512)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_conv = nn.Dropout2d(0.3)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.stats_norm = nn.LayerNorm(stat_features)
        self.classifier = nn.Sequential(
            nn.Linear(512 + stat_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, mel: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.block1(mel))    # (128, 130) -> (64, 65)
        x = self.pool(self.block2(x))      # -> (32, 32)
        x = self.pool(self.block3(x))      # -> (16, 16)
        x = self.pool(self.block4(x))      # -> (8, 8)
        x = self.dropout_conv(x)
        x = self.gap(x).flatten(1)
        stats = self.stats_norm(stats)
        x = torch.cat([x, stats], dim=1)
        return self.classifier(x)
