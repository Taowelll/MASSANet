import torch
import torch.nn as nn

from model.layers import (
    EEGResidualBlock,
    MultiScaleSpectralAttention,
    TemporalSelfAttention,
    UnifiedAttention,
)


class MASSA(nn.Module):
    def __init__(self, num_channels, sampling_rate, F1=16, D=1, F2="auto", drop_out=0.1):
        super().__init__()
        del sampling_rate
        F2 = F1 * D if F2 == "auto" else F2
        self.output_channels = F2 * 2

        self.spectral_branch = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, F1, (1, 128), padding="same"),
                    nn.BatchNorm2d(F1),
                    MultiScaleSpectralAttention(F1, F1=F1, scales=[128, 64], dropout=drop_out),
                ),
                nn.Sequential(
                    nn.Conv2d(1, F1, (1, 32), padding="same"),
                    nn.BatchNorm2d(F1),
                    MultiScaleSpectralAttention(F1, F1=F1, scales=[32, 2], dropout=drop_out),
                ),
            ]
        )

        self.spatial_branch = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(F2, F2, (num_channels, 1), groups=max(1, F2 // 4)),
                    nn.BatchNorm2d(F2),
                    nn.ELU(),
                    nn.MaxPool2d((1, 32), stride=32),
                    nn.Dropout(drop_out),
                    nn.Conv2d(F2, F2, (1, 1)),
                    UnifiedAttention(mode="cs", in_channels=F2, dropout=drop_out),
                    nn.Dropout(drop_out),
                    EEGResidualBlock(F2, F2, dropout=drop_out),
                ),
                nn.Sequential(
                    nn.Conv2d(F2, F2, (num_channels, 1)),
                    nn.BatchNorm2d(F2),
                    nn.ELU(),
                    nn.MaxPool2d((1, 64), stride=25),
                    nn.Dropout(drop_out),
                    nn.Conv2d(F2, F2, (1, 1)),
                    UnifiedAttention(mode="cs", in_channels=F2, dropout=drop_out),
                    nn.Dropout(drop_out),
                    EEGResidualBlock(F2, F2, dropout=drop_out),
                ),
            ]
        )

        self.temporal_align = nn.AdaptiveMaxPool2d((1, 35))
        self.fusion_attention = UnifiedAttention(mode="m", in_channels=self.output_channels, num_heads=8, dropout=drop_out)
        self.temporal_attention = TemporalSelfAttention(in_channels=self.output_channels, num_heads=8, dropout=drop_out)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x):
        spectral_0 = self.spectral_branch[0](x)
        spectral_1 = self.spectral_branch[1](x)

        spatial_0 = self.temporal_align(self.spatial_branch[0](spectral_0))
        spatial_1 = self.temporal_align(self.spatial_branch[1](spectral_1))

        fused = torch.cat([spatial_0, spatial_1], dim=1)
        fusion_out = self.fusion_attention(fused)

        batch_size, channels, _, time_points = fused.shape
        temporal_out = self.temporal_attention(fused.view(batch_size, channels, time_points))
        temporal_out = temporal_out.view(batch_size, channels, 1, time_points)
        return fusion_out + self.dropout(temporal_out)


class Classifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Conv2d(in_channels, 16, (1, 35)),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, num_classes, (1, 1)),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        x = self.dense(x)
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)
        return x


class Net(nn.Module):
    def __init__(self, num_classes: int, num_channels: int, sampling_rate: int, dropout: float = 0.1):
        super().__init__()
        self.backbone = MASSA(
            num_channels=num_channels,
            sampling_rate=sampling_rate,
            drop_out=dropout,
        )
        self.classifier = Classifier(self.backbone.output_channels, num_classes)

    def forward(self, x):
        x = self.backbone(x.float())
        return self.classifier(x)


def get_model(args):
    return Net(
        num_classes=args.num_classes,
        num_channels=args.num_channels,
        sampling_rate=args.sampling_rate,
        dropout=float(getattr(args, "dropout", 0.1)),
    )
