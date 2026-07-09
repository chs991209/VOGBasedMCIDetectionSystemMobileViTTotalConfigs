"""Distribution-Aware Gated Fusion meta-classifier.

Replaces the prior Option C surgical-fusion architecture (which monkey-patched
MobileViT's deepest self-attention to accept a padding mask + cross-trial
sequence reshape — that scheme leaked the class signal through the padding
pattern itself).

Pipeline (per task, 8 tasks total):
    [B, 10, 4, 32, 32]
        │
    reshape ────────►  [B*10, 4, 32, 32]
        │
    Conv2d adapter (4→3) + BN + ReLU                 ← trainable
    nearest-neighbour upscale to 256×256
        │
    frozen MobileViTModel (UNMODIFIED — no patching)
        │  pooler_output
        ▼
    [B*10, D]   (D = mobilevit-small pooler output = 640)
        │
    reshape ────────►  [B, 10, D]
        │
    μ_t  = .mean(dim=1)  ─────────────────►  [B, D]
    σ²_t = .var(dim=1, unbiased=False)  ───►  [B, D]
        │
    concat (μ, σ²) ───►  [B, 2D]                     ← distribution feature
        │
    per-task gate: Linear(2D → 1) → Sigmoid → α_t    ← Bayesian priority
        │
    weighted_mean_t = μ_t · α_t  ──────────►  [B, D]
        │
    concat across 8 tasks   ────────────────►  [B, 8·D]
        │
    head: Linear → GELU → Dropout → Linear(num_classes)

The frozen backbone weights are unchanged. The padding mask is gone — every
subject is guaranteed (by the dataset's strict-parity rule) to have exactly
MAX_TRIALS=10 real trials per task, so there's nothing to mask.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MobileViTModel


class DistributionAwareFusionClassifier(nn.Module):
    """Subject-level classifier over [B, NUM_TASKS, MAX_TRIALS, 4, 32, 32]
    using per-task distribution-aware Bayesian gating.

    The "256" dimensionality cited in the architecture directive was treated
    as illustrative; the actual D is `mobilevit-small`'s pooler_output width
    (= neck_hidden_sizes[-1] = 640). All downstream shapes scale with D.
    The topology — mean+variance → sigmoid gate → multiply-mean — is exact.
    """

    NUM_TASKS = 8

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 4,
        max_trials: int = 10,
        fc_hidden: int = 128,
        dropout: float = 0.5,
        shared_backbone: nn.Module = None,
    ):
        super().__init__()
        self.num_tasks = self.NUM_TASKS
        self.max_trials = max_trials

        # Standard Conv2d adapter: 4-channel CWT → 3-channel "RGB" for ViT
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )

        # Clean, unmodified, frozen MobileViTModel
        if shared_backbone is not None:
            self.backbone = shared_backbone
        else:
            self.backbone = MobileViTModel.from_pretrained("apple/mobilevit-small")
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        # Pooler output width — mobilevit-small: 640
        D = int(self.backbone.config.neck_hidden_sizes[-1])
        self.D = D

        # Per-task gates: distribution → scalar priority weight α_t ∈ (0,1)
        self.gates = nn.ModuleList(
            [nn.Linear(2 * D, 1) for _ in range(self.NUM_TASKS)]
        )

        # Final head: Linear → GELU → Dropout → Linear(num_classes)
        total = self.NUM_TASKS * D
        self.head = nn.Sequential(
            nn.Linear(total, fc_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),
        )

    def _backbone_gap(self, x_flat: torch.Tensor) -> torch.Tensor:
        """[B*T, 3, 256, 256] → [B*T, D] via MobileViT's pooler (GAP-based)."""
        out = self.backbone(x_flat)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        # Fallback: manual GAP on last_hidden_state [B, D, H', W']
        return out.last_hidden_state.mean(dim=(-2, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, NUM_TASKS, max_trials, 4, 32, 32]
        Returns logits [B, num_classes]. No padding mask — by construction
        every (subject, task) cell holds exactly `max_trials` real trials.
        """
        B, NT, T, C, H, W = x.shape
        assert NT == self.num_tasks, f"task dim {NT} != {self.num_tasks}"
        assert T == self.max_trials, f"trial dim {T} != MAX_TRIALS {self.max_trials}"

        weighted_means = []
        for t in range(self.num_tasks):
            xt = x[:, t]                                     # [B, T, C, 32, 32]
            xt_flat = xt.reshape(B * T, C, H, W)             # [B*T, C, 32, 32]
            xt_flat = self.adapter(xt_flat)                  # [B*T, 3, 32, 32]
            xt_flat = F.interpolate(xt_flat, size=(256, 256), mode="nearest")
            feats = self._backbone_gap(xt_flat)              # [B*T, D]
            feats = feats.reshape(B, T, self.D)              # [B, T, D]

            mu_t = feats.mean(dim=1)                          # [B, D]
            var_t = feats.var(dim=1, unbiased=False)          # [B, D]
            dist = torch.cat([mu_t, var_t], dim=1)            # [B, 2D]

            alpha_t = torch.sigmoid(self.gates[t](dist))      # [B, 1]
            weighted_means.append(mu_t * alpha_t)             # [B, D]

        big = torch.cat(weighted_means, dim=1)                # [B, 8·D]
        return self.head(big)                                 # [B, num_classes]
