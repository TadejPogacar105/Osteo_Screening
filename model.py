"""
Neural architecture: ResNet-50 (1-ch) + optional neck + TARR + multi heads.
Decoupled from filesystem / training — only `ModelConfig` + tensors in `forward`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


@dataclass
class ModelConfig:
    """Hyperparameters that define the network shape and optional branches."""

    dropout: float = 0.3
    use_neck: bool = True
    neck_dim: int = 128
    use_severe_head: bool = True
    use_tscore_branch: bool = True
    use_tarr: bool = True
    tarr_hidden_dim: int = 128
    tarr_adapter_dim: int = 64
    backbone_weights: ResNet50_Weights = ResNet50_Weights.DEFAULT


class BinaryScreeningModel(nn.Module):
    """
    ResNet-50 backbone adapted for single-channel grayscale input.

    Forward returns:
      - logits_screen: (B, 1)
      - logits_severe: (B, 1) or None
      - logits_tscore: (B, 1) or None (needs baseline_info when branch enabled)
    """

    class TaskAdaptiveResidualRouting(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int, adapter_dim: int):
            super().__init__()
            self.in_dim = int(in_dim)
            self.hidden_dim = int(hidden_dim)
            self.adapter_dim = int(adapter_dim)

            self.shared_proj = nn.Sequential(
                nn.Linear(self.in_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(inplace=True),
            )

            def _make_adapter():
                return nn.Sequential(
                    nn.Linear(self.hidden_dim, self.adapter_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.adapter_dim, self.hidden_dim),
                )

            self.adapter_screen = _make_adapter()
            self.adapter_severe = _make_adapter()
            self.adapter_tscore = _make_adapter()

            self.gate_screen = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.gate_severe = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.gate_tscore = nn.Linear(self.hidden_dim, self.hidden_dim)

            self.out_screen = nn.Linear(self.hidden_dim, self.in_dim)
            self.out_severe = nn.Linear(self.hidden_dim, self.in_dim)
            self.out_tscore = nn.Linear(self.hidden_dim, self.in_dim)

        def _route_one(self, h, gate_layer, adapter_layer, out_layer):
            gate = torch.sigmoid(gate_layer(h))
            delta = gate * adapter_layer(h)
            return out_layer(h + delta)

        def forward(self, z):
            h = self.shared_proj(z)
            z_screen = self._route_one(h, self.gate_screen, self.adapter_screen, self.out_screen)
            z_severe = self._route_one(h, self.gate_severe, self.adapter_severe, self.out_severe)
            z_tscore = self._route_one(h, self.gate_tscore, self.adapter_tscore, self.out_tscore)
            return z_screen, z_severe, z_tscore

    def __init__(self, model_cfg: ModelConfig | None = None):
        super().__init__()
        self.mcfg = model_cfg or ModelConfig()
        m = self.mcfg

        backbone_raw = models.resnet50(weights=m.backbone_weights)

        old_conv = backbone_raw.conv1
        new_conv = nn.Conv2d(
            1,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        backbone_raw.conv1 = new_conv

        self.backbone = nn.Sequential(*list(backbone_raw.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        backbone_dim = backbone_raw.fc.in_features

        self.use_neck = bool(m.use_neck)
        self.use_severe_head = bool(m.use_severe_head)
        self.use_tscore_branch = bool(m.use_tscore_branch)
        self.use_tarr = bool(m.use_tarr)

        if self.use_neck:
            self.neck = nn.Sequential(
                nn.Linear(backbone_dim, m.neck_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=m.dropout),
            )
            feat_dim = int(m.neck_dim)
        else:
            self.neck = nn.Identity()
            feat_dim = int(backbone_dim)

        self.head = nn.Linear(feat_dim, 1)
        self.severe_head = nn.Linear(feat_dim, 1) if self.use_severe_head else None
        self.tarr = (
            self.TaskAdaptiveResidualRouting(
                in_dim=feat_dim,
                hidden_dim=int(m.tarr_hidden_dim),
                adapter_dim=int(m.tarr_adapter_dim),
            )
            if self.use_tarr
            else None
        )

        if self.use_tscore_branch:
            self.img_adapter = nn.Sequential(
                nn.Linear(feat_dim, 64),
                nn.ReLU(inplace=True),
            )
            self.tabular_encoder = nn.Sequential(
                nn.Linear(3, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 32),
                nn.ReLU(inplace=True),
            )
            self.fusion_mlp = nn.Sequential(
                nn.Linear(64 + 32, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(p=m.dropout),
            )
            self.tscore_head = nn.Linear(64, 1)
        else:
            self.img_adapter = None
            self.tabular_encoder = None
            self.fusion_mlp = None
            self.tscore_head = None

    def forward(self, x, baseline_info=None):
        f = self.backbone(x)
        f = self.pool(f).flatten(1)
        z = self.neck(f)
        if self.use_tarr and self.tarr is not None:
            z_screen, z_severe, z_tscore = self.tarr(z)
        else:
            z_screen, z_severe, z_tscore = z, z, z
        logits_screen = self.head(z_screen)
        logits_severe = self.severe_head(z_severe) if self.severe_head is not None else None

        logits_tscore = None
        if self.use_tscore_branch and baseline_info is not None:
            z_img = self.img_adapter(z_tscore.detach())
            z_tab = self.tabular_encoder(baseline_info)
            z_fuse = self.fusion_mlp(torch.cat([z_img, z_tab], dim=1))
            logits_tscore = self.tscore_head(z_fuse)
        return logits_screen, logits_severe, logits_tscore
