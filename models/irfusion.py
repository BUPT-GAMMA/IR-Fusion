from collections import OrderedDict
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalise_state_dict(state_dict):
    if isinstance(state_dict, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in state_dict and isinstance(state_dict[key], dict):
                state_dict = state_dict[key]
                break

    normalised = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        normalised[key] = value
    return normalised


def init_weights(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)
        elif isinstance(layer, nn.BatchNorm2d):
            nn.init.constant_(layer.weight, 1.0)
            nn.init.constant_(layer.bias, 0.0)


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=3, padding=None) -> None:
        super().__init__()
        if padding is None:
            if isinstance(kernel_size, tuple):
                padding = tuple(k // 2 for k in kernel_size)
            else:
                padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_channels, out_channels, 3),
            ConvBNReLU(out_channels, out_channels, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def _split_widths(total: int, parts: int) -> Sequence[int]:
    base = total // parts
    widths = [base] * parts
    for idx in range(total - base * parts):
        widths[idx] += 1
    return widths


class _InceptionCSplitBranch(nn.Module):
    def __init__(self, stem: Sequence[nn.Module], hidden_channels: int, out1: int, out2: int, first_kernel, second_kernel) -> None:
        super().__init__()
        self.stem = nn.Sequential(*stem)
        self.branch1 = ConvBNReLU(hidden_channels, out1, first_kernel)
        self.branch2 = ConvBNReLU(hidden_channels, out2, second_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        return torch.cat((self.branch1(x), self.branch2(x)), dim=1)


class InceptionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kind: str = "A") -> None:
        super().__init__()
        kind = kind.upper()

        widths = list(_split_widths(out_channels, 4))

        if kind == "A":
            branch2 = [ConvBNReLU(in_channels, widths[1], 1, padding=0), ConvBNReLU(widths[1], widths[1], 3)]
            branch3 = [
                ConvBNReLU(in_channels, widths[2], 1, padding=0),
                ConvBNReLU(widths[2], widths[2], 3),
                ConvBNReLU(widths[2], widths[2], 3),
            ]
        elif kind == "B":
            branch2 = [
                ConvBNReLU(in_channels, widths[1], 1, padding=0),
                ConvBNReLU(widths[1], widths[1], (1, 7)),
                ConvBNReLU(widths[1], widths[1], (7, 1)),
            ]
            branch3 = [
                ConvBNReLU(in_channels, widths[2], 1, padding=0),
                ConvBNReLU(widths[2], widths[2], (7, 1)),
                ConvBNReLU(widths[2], widths[2], (1, 7)),
                ConvBNReLU(widths[2], widths[2], (7, 1)),
                ConvBNReLU(widths[2], widths[2], (1, 7)),
            ]
        else:
            c_widths = list(_split_widths(out_channels, 6))
            branch2_hidden = c_widths[1] + c_widths[2]
            branch3_hidden = c_widths[3] + c_widths[4]
            widths = [c_widths[0], branch2_hidden, branch3_hidden, c_widths[5]]
            branch2 = _InceptionCSplitBranch(
                [ConvBNReLU(in_channels, branch2_hidden, 1, padding=0)],
                branch2_hidden,
                c_widths[1],
                c_widths[2],
                (1, 3),
                (3, 1),
            )
            branch3 = _InceptionCSplitBranch(
                [
                    ConvBNReLU(in_channels, branch3_hidden, 1, padding=0),
                    ConvBNReLU(branch3_hidden, branch3_hidden, (3, 1)),
                    ConvBNReLU(branch3_hidden, branch3_hidden, (1, 3)),
                ],
                branch3_hidden,
                c_widths[3],
                c_widths[4],
                (1, 3),
                (3, 1),
            )
        self.branch1 = ConvBNReLU(in_channels, widths[0], 1, padding=0)
        self.branch2 = branch2 if isinstance(branch2, nn.Module) else nn.Sequential(*branch2)
        self.branch3 = branch3 if isinstance(branch3, nn.Module) else nn.Sequential(*branch3)
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels, widths[3], 1, padding=0),
        )
        self.project = ConvBNReLU(out_channels, out_channels, 1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat((self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)), dim=1)
        return self.project(x)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(F.adaptive_avg_pool2d(x, 1))
        max_out = self.mlp(F.adaptive_max_pool2d(x, 1))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        att = self.sigmoid(self.conv(torch.cat((avg_out, max_out), dim=1)))
        return x * att


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))


class AttentionGate(nn.Module):
    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.gate_proj = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, 1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, 1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if gate.shape[-2:] != skip.shape[-2:]:
            gate = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        att = self.relu(self.gate_proj(gate) + self.skip_proj(skip))
        return skip * self.psi(att)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kind: str, num_blocks: int = 2) -> None:
        super().__init__()
        blocks = [nn.MaxPool2d(2), InceptionBlock(in_channels, out_channels, kind=kind)]
        for _ in range(max(0, num_blocks - 1)):
            blocks.append(InceptionBlock(out_channels, out_channels, kind=kind))
        self.block = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        up_channels = in_channels

        self.attention = AttentionGate(up_channels, skip_channels, max(out_channels // 2, 1))
        self.cbam = CBAM(up_channels + skip_channels)
        self.conv = ConvBlock(up_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        skip = self.attention(x, skip)
        return self.conv(self.cbam(torch.cat((skip, x), dim=1)))


class IRFusionNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = 32, 64, 128, 256

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        self.stem = ConvBlock(self.in_channels, c1)
        self.down1 = DownBlock(c1, c2, kind="A")
        self.down2 = DownBlock(c2, c3, kind="B")
        self.down3 = DownBlock(c3, c4, kind="C")

        self.up2 = UpBlock(c4, c3, c3)
        self.up1 = UpBlock(c3, c2, c2)
        self.up0 = UpBlock(c2, c1, c1)
        self.head = nn.Conv2d(c1, self.out_channels, kernel_size=1)
        init_weights(self)

    @staticmethod
    def _pad_to_multiple(x: torch.Tensor, multiple: int = 8) -> Tuple[torch.Tensor, Tuple[int, int]]:
        height, width = x.shape[-2:]
        pad_h = (multiple - height % multiple) % multiple
        pad_w = (multiple - width % multiple) % multiple
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, (pad_h, pad_w)

    @staticmethod
    def _crop(x: torch.Tensor, pad_hw: Tuple[int, int]) -> torch.Tensor:
        pad_h, pad_w = pad_hw
        if pad_h:
            x = x[..., :-pad_h, :]
        if pad_w:
            x = x[..., :, :-pad_w]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, pad_hw = self._pad_to_multiple(x)
        x0 = self.stem(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        x = self.up2(x3, x2)
        x = self.up1(x, x1)
        x = self.up0(x, x0)
        x = self._crop(self.head(x), pad_hw)

        return F.relu(x)

    def load(self, checkpoint_path: str, strict: bool = False) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = _normalise_state_dict(checkpoint)
        missing, unexpected = self.load_state_dict(state_dict, strict=strict)
        if missing or unexpected:
            print(f"Loaded {checkpoint_path} with missing={len(missing)}, unexpected={len(unexpected)}")
