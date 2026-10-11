"""Building blocks recovered from the supplied ESC-50 notebook."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualNormalization(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.lambda_param = nn.Parameter(torch.ones(num_features, 1, 1))
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=True)

    def forward(self, x):
        return self.lambda_param * x + self.instance_norm(x)


class LearnablePooling(nn.Module):
    def __init__(self, in_channels, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or in_channels // 2

        self.bn_input = ResidualNormalization(in_channels)
        self.attn_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.bn_attn = ResidualNormalization(hidden_dim)
        self.attn_score = nn.Conv2d(hidden_dim, in_channels, kernel_size=1, bias=False)
        self.activation = nn.LeakyReLU(0.1, inplace=True)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x_norm = self.bn_input(x)
        attention = self.attn_conv(x_norm)
        attention = self.bn_attn(attention)
        attention = self.activation(attention)
        scores = self.attn_score(attention)

        batch_size, channels, height, width = x.size()
        spatial_weights = F.softmax(
            scores.view(batch_size, channels, -1), dim=-1
        ).view(batch_size, channels, height, width)

        attention_pooled = (x * spatial_weights).sum(dim=[2, 3])
        average_pooled = self.global_avg_pool(x).squeeze(-1).squeeze(-1)
        return torch.cat([attention_pooled, average_pooled], dim=1)


class StudentConv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1, 1), groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            groups=groups,
        )

    def forward(self, x):
        return self.conv(x)


class StudentLightEchoBlock(nn.Module):
    """Exact block ordering recovered from the supplied ESC-50 notebook.

    Note: ``expand_norm`` is retained because it exists in the trained model
    state dictionary even though the supplied forward method does not call it.
    """

    def __init__(self, in_channels, out_channels, stride, expansion_factor=6):
        super().__init__()
        self.use_skip = True

        self.input_norm = nn.BatchNorm2d(in_channels)
        self.input_scaling = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.expand_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.expand_norm = nn.BatchNorm2d(out_channels)
        self.expand_activation = nn.LeakyReLU()

        self.spatial_conv = StudentConv3x3(
            out_channels, out_channels, stride=stride, groups=out_channels
        )
        self.spatial_norm = nn.BatchNorm2d(out_channels)
        self.spatial_activation = nn.LeakyReLU()

    def forward(self, x):
        residual = self.input_scaling(x)

        out = self.input_norm(x)
        out = self.expand_conv(out)
        out = self.expand_activation(out)

        # Operation order intentionally matches the supplied notebook.
        out = self.spatial_norm(out)
        out = self.spatial_conv(out)
        out = self.spatial_activation(out)

        return out + residual
