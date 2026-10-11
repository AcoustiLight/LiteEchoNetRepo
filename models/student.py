"""Lightweight student architecture recovered from the ESC-50 notebook."""

import torch.nn as nn

from .blocks import (
    LearnablePooling,
    ResidualNormalization,
    StudentConv3x3,
    StudentLightEchoBlock,
)


class StudentLightEchoNet(nn.Module):
    """Student network used by the supplied ESC-50 checkpoints.

    ``forward_with_features`` exposes the four intermediate feature locations
    used by the structurally aligned feature-distillation design.
    """

    def __init__(self, num_classes=50):
        super().__init__()

        self.input_norm = nn.BatchNorm2d(1)

        self.conv1 = StudentConv3x3(1, 16, stride=(2, 2))
        self.norm1 = nn.BatchNorm2d(16)
        self.activation1 = nn.ReLU()

        self.conv2 = StudentConv3x3(16, 32, stride=(2, 2))
        self.activation2 = nn.ReLU()

        self.stage1 = nn.Sequential(
            StudentLightEchoBlock(32, 32, stride=(1, 1)),
            StudentLightEchoBlock(32, 32, stride=(1, 1)),
            StudentLightEchoBlock(32, 32, stride=(1, 1)),
        )
        self.stage1_norm = ResidualNormalization(32)

        self.stage2 = nn.Sequential(
            StudentLightEchoBlock(32, 32, stride=(1, 1)),
            StudentLightEchoBlock(32, 32, stride=(1, 1)),
            StudentLightEchoBlock(32, 32, stride=(1, 1)),
        )
        self.stage2_norm = ResidualNormalization(32)

        self.stage3 = nn.Sequential(
            StudentLightEchoBlock(32, 64, stride=(1, 1)),
        )
        self.stage3_norm = ResidualNormalization(64)

        self.pooling = LearnablePooling(64)
        self.head_norm = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(128, num_classes)

    def _forward_backbone(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        features = []
        x = self.input_norm(x)

        x = self.conv1(x)
        x = self.activation1(x)
        x = self.norm1(x)

        x = self.conv2(x)
        x = self.activation2(x)
        features.append(x)

        x = self.stage1(x) + x
        x = self.stage1_norm(x)
        features.append(x)

        x = self.stage2(x) + x
        x = self.stage2_norm(x)
        features.append(x)

        x = self.stage3(x)
        x = self.stage3_norm(x)
        features.append(x)

        return x, features

    def forward(self, x):
        x, _ = self._forward_backbone(x)
        x = self.pooling(x)
        x = self.head_norm(x)
        x = self.dropout(x)
        return self.classifier(x)

    def forward_with_features(self, x):
        x, features = self._forward_backbone(x)
        pooled = self.pooling(x)
        output = self.head_norm(pooled)
        output = self.dropout(output)
        logits = self.classifier(output)
        return logits, features
