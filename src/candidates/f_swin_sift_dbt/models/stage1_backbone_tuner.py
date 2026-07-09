"""Stage 1 — Domain-Adaptive Swin-Tiny Backbone Tuner.

Trained per-fold on flat 8-task CWT windows to inject "density" into the
frozen Swin-Tiny features that Stage 2 will later consume. Task-agnostic head
(no task embedding). Fully UNFROZEN backbone + adapter — everything trains.

Adapter geometry: legacy `Conv2d(4→3, K=(5,1)) + BN + ReLU` — chosen because
the K=(5,1) kernel preserves frequency resolution while sharpening
time-axis (vertical) transients. Then nearest-neighbour upscale to 224×224
to match Swin-Tiny's pretraining resolution (per SWIN_ADAPTION_GUIDE.md §2.1).

Return: logits [B, 2].
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SwinModel


SWIN_HIDDEN_DIM = 768   # microsoft/swin-tiny-patch4-window7-224


class Stage1SwinBackboneTuner(nn.Module):
    """Legacy adapter + upscale + Swin-Tiny (unfrozen) + Linear(768→2)."""

    def __init__(self, in_channels: int = 4, num_classes: int = 2):
        super().__init__()
        # Adapter: 4-channel CWT → 3-channel RGB-like input for Swin
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=(5, 1), padding=(2, 0)),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )
        # Fully unfrozen Swin-Tiny
        self.backbone = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        # Task-agnostic classification head (no task embedding at Stage 1)
        self.classifier = nn.Linear(SWIN_HIDDEN_DIM, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 4, 32, 32] → [B, 768] via adapter + 224×224 upscale + Swin pooler."""
        x = self.adapter(x)
        x = F.interpolate(x, size=(224, 224), mode="nearest")
        out = self.backbone(pixel_values=x)
        if getattr(out, "pooler_output", None) is not None:
            return out.pooler_output
        # Fallback: manual mean pool over sequence dim of last_hidden_state
        return out.last_hidden_state.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)      # [B, 768]
        return self.classifier(features)          # [B, 2]

    def export_extractor_state(self) -> dict:
        """State dict for the (adapter + backbone) pair — everything that
        Stage 2 needs to load and freeze."""
        return {
            "adapter": self.adapter.state_dict(),
            "backbone": self.backbone.state_dict(),
        }
