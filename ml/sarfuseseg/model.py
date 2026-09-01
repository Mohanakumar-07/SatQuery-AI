"""SAR-FuseSeg network: ResNet-18 optical encoder + ResNet-18 SAR encoder, multi-scale
concatenation fusion, U-Net-style decoder (Implementation_Plan_v1.2.md section 4.3).

Deliberately simple per the phase-1 brief: no cross-attention, no transformers, no
large encoders. Both encoders are trained from scratch (no ImageNet weights) because
the optical branch takes 4 bands and the SAR branch takes 2 — neither matches the
pretrained 3-channel RGB stem, and silently discarding/duplicating channels to reuse
ImageNet weights would be exactly the kind of unexplained shortcut this project avoids.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock

from ml.sarfuseseg.config import N_CLASSES, OPTICAL_CHANNELS, SAR_CHANNELS

_STAGE_OUT_CHANNELS = [64, 128, 256, 512]  # resnet18 layer1..layer4
_STAGE_BLOCKS = [2, 2, 2, 2]


class ResNet18Encoder(nn.Module):
    """A from-scratch ResNet-18 stem + 4 stages, returning one feature map per stage."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.in_planes = 64
        self.layer1 = self._make_stage(64, _STAGE_BLOCKS[0], stride=1)
        self.layer2 = self._make_stage(128, _STAGE_BLOCKS[1], stride=2)
        self.layer3 = self._make_stage(256, _STAGE_BLOCKS[2], stride=2)
        self.layer4 = self._make_stage(512, _STAGE_BLOCKS[3], stride=2)

    def _make_stage(self, planes: int, n_blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [BasicBlock(self.in_planes, planes, stride=stride, downsample=downsample)]
        self.in_planes = planes
        for _ in range(1, n_blocks):
            layers.append(BasicBlock(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)  # stride 4
        f1 = self.layer1(x)  # stride 4,  64ch
        f2 = self.layer2(f1)  # stride 8, 128ch
        f3 = self.layer3(f2)  # stride 16, 256ch
        f4 = self.layer4(f3)  # stride 32, 512ch
        return [f1, f2, f3, f4]


class FusionBlock(nn.Module):
    """Concatenate optical+SAR features at one scale, squeeze back with a 1x1 conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, optical_feat: torch.Tensor, sar_feat: torch.Tensor) -> torch.Tensor:
        return self.squeeze(torch.cat([optical_feat, sar_feat], dim=1))


class DecoderBlock(nn.Module):
    """Upsample x2, concat with a shallower skip, 2x (conv-bn-relu)."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SARFuseSeg(nn.Module):
    """Optical ResNet-18 + SAR ResNet-18 -> multi-scale fusion -> U-Net decoder.

    Also usable as a single-modality baseline: pass zeros (or the real tensor of the
    other branch) and set ``use_optical=False``/``use_sar=False`` to run the Experiment
    1/2 ablations without changing the checkpoint format.
    """

    def __init__(self, n_classes: int = N_CLASSES, optical_channels: int = OPTICAL_CHANNELS, sar_channels: int = SAR_CHANNELS):
        super().__init__()
        self.optical_encoder = ResNet18Encoder(optical_channels)
        self.sar_encoder = ResNet18Encoder(sar_channels)
        self.fusions = nn.ModuleList([FusionBlock(c) for c in _STAGE_OUT_CHANNELS])

        c1, c2, c3, c4 = _STAGE_OUT_CHANNELS
        self.decoder4 = DecoderBlock(c4, c3, c3)
        self.decoder3 = DecoderBlock(c3, c2, c2)
        self.decoder2 = DecoderBlock(c2, c1, c1)
        self.decoder1 = DecoderBlock(c1, 0, c1 // 2)  # no skip left (stem was stride 4 already)
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(c1 // 2, c1 // 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(c1 // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c1 // 4, c1 // 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(c1 // 4),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(c1 // 4, n_classes, kernel_size=1)

    def forward(self, optical: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        opt_feats = self.optical_encoder(optical)
        sar_feats = self.sar_encoder(sar)
        f1, f2, f3, f4 = (fusion(o, s) for fusion, o, s in zip(self.fusions, opt_feats, sar_feats))

        x = self.decoder4(f4, f3)
        x = self.decoder3(x, f2)
        x = self.decoder2(x, f1)
        x = self.decoder1(x, None)
        x = self.final_upsample(x)
        logits = self.classifier(x)
        if logits.shape[-2:] != optical.shape[-2:]:
            logits = nn.functional.interpolate(logits, size=optical.shape[-2:], mode="bilinear", align_corners=False)
        return logits
