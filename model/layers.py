import torch
import torch.nn.functional as F
from torch import Tensor, nn
from einops import rearrange


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.pos_encoding = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        qkv = self.qkv(x + self.pos_encoding)
        qkv = qkv.reshape(batch_size, 3, self.num_heads, self.head_dim, height * width)
        qkv = qkv.permute(1, 0, 2, 4, 3)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.dropout(torch.softmax(attn, dim=-1))
        out = attn @ v
        out = out.permute(0, 1, 3, 2).reshape(batch_size, channels, height, width)
        return self.proj(out)


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum("bhqd,bhkd->bhqk", queries, keys)
        attn = F.softmax(energy / (self.emb_size ** 0.5), dim=-1)
        attn = self.att_drop(attn)
        out = torch.einsum("bhqk,bhkd->bhqd", attn, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.projection(out)


class UnifiedAttention(nn.Module):
    def __init__(self, mode="csfm", in_channels=256, num_heads=8, num_bands=6, dropout=0.1):
        super().__init__()
        self.mode = mode.lower()
        self.num_bands = num_bands

        if "c" in self.mode:
            hidden_channels = max(8, in_channels // 16)
            self.channel_att = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, hidden_channels, 1),
                nn.ELU(),
                nn.Conv2d(hidden_channels, in_channels, 1),
                nn.Sigmoid(),
            )

        if "s" in self.mode:
            self.spatial_att = SpatialSelfAttention(
                in_channels=in_channels,
                num_heads=max(1, num_heads // 2),
                dropout=dropout,
            )

        if "m" in self.mode:
            self.norm = nn.LayerNorm(in_channels)
            self.mha = MultiHeadAttention(
                emb_size=in_channels,
                num_heads=num_heads,
                dropout=dropout,
            )

        if "f" in self.mode:
            self.band_weights = nn.Parameter(torch.ones(1, num_bands))

    def forward(self, x):
        identity = x

        if "c" in self.mode:
            x = x * self.channel_att(x)

        if "s" in self.mode:
            x = x * self.spatial_att(x)

        if "f" in self.mode:
            _, _, _, time_points = x.shape
            band_size = max(1, time_points // self.num_bands)
            weights = torch.softmax(self.band_weights, dim=-1)
            weighted = torch.zeros_like(x)
            for band_idx in range(self.num_bands):
                start = band_idx * band_size
                end = time_points if band_idx == self.num_bands - 1 else min(time_points, (band_idx + 1) * band_size)
                weighted[:, :, :, start:end] = x[:, :, :, start:end] * weights[:, band_idx].view(1, 1, 1, 1)
            x = weighted

        if "m" in self.mode:
            batch_size, channels, height, width = x.shape
            sequence = x.reshape(batch_size, channels, height * width).permute(0, 2, 1)
            sequence = self.norm(sequence)
            sequence = self.mha(sequence)
            x = sequence.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)

        return identity + x


class ChannelSelfAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super().__init__()
        hidden_channels = max(8, in_channels // reduction_ratio)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, hidden_channels, 1),
            nn.ELU(),
            nn.Conv2d(hidden_channels, in_channels, 1),
        )

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        return x * weights


class LocalSpatialAttention(nn.Module):
    def __init__(self, in_channels, kernel_size, dropout=0.1):
        super().__init__()
        self.dynamic_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=(1, kernel_size),
                padding=(0, kernel_size // 2),
                groups=in_channels,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=(1, kernel_size),
                padding=(0, kernel_size // 2),
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.dynamic_conv(x)


class MultiScaleSpectralAttention(nn.Module):
    def __init__(self, in_channels, F1=16, scales=None, dropout=0.1):
        super().__init__()
        scales = [125, 62] if scales is None else scales
        self.branches = nn.ModuleList()
        for kernel_size in scales:
            padding = (kernel_size - 1) // 2
            self.branches.append(
                nn.Sequential(
                    nn.ConstantPad2d((padding, kernel_size - 1 - padding, 0, 0), 0),
                    nn.Conv2d(in_channels, F1, (1, kernel_size), padding=0),
                    nn.BatchNorm2d(F1),
                    nn.ELU(),
                    nn.Dropout(dropout),
                    ChannelSelfAttention(F1),
                )
            )

        self.fusion = nn.Sequential(
            nn.Conv2d(F1 * len(scales), F1, kernel_size=1),
            LocalSpatialAttention(F1, kernel_size=3, dropout=dropout),
        )

    def forward(self, x):
        branch_outputs = [branch(x) for branch in self.branches]
        return self.fusion(torch.cat(branch_outputs, dim=1))


class EEGResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.conv_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, (1, 7), padding=(0, 3)),
                    nn.BatchNorm2d(out_channels),
                    nn.ELU(),
                    nn.Conv2d(out_channels, out_channels, (1, 7), padding=(0, 3)),
                    nn.BatchNorm2d(out_channels),
                    nn.ELU(),
                    nn.Dropout2d(dropout),
                ),
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, (1, 5), padding=(0, 2)),
                    nn.BatchNorm2d(out_channels),
                    nn.ELU(),
                    nn.Conv2d(out_channels, out_channels, (1, 5), padding=(0, 2)),
                    nn.BatchNorm2d(out_channels),
                    nn.ELU(),
                    nn.Dropout2d(dropout),
                ),
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, (1, 3), padding=(0, 1)),
                    nn.BatchNorm2d(out_channels),
                    nn.ELU(),
                    nn.Conv2d(out_channels, out_channels, (1, 3), padding=(0, 1)),
                    nn.BatchNorm2d(out_channels),
                    nn.ELU(),
                    nn.Dropout2d(dropout),
                ),
            ]
        )
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (1, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.final_activation = nn.ELU()

    def forward(self, x):
        identity = self.shortcut(x)
        out = x
        for layer in self.conv_layers:
            out = layer(out)
        return self.final_activation(out + identity)


class TemporalSelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Conv1d(in_channels, in_channels * 3, kernel_size=1)
        self.proj = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, channels, time_points = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.reshape(batch_size, 3, self.num_heads, self.head_dim, time_points).unbind(1)
        q = q.permute(0, 1, 3, 2)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.dropout(torch.softmax(attn, dim=-1))
        out = attn @ v
        out = out.permute(0, 1, 3, 2).reshape(batch_size, channels, time_points)
        return self.proj(out)
