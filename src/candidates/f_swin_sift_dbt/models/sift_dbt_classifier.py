"""SIFT-DBT (Strictly Defined Dimensions & Device-Robust)

Stage 2 classifier. Frozen (adapter + Swin-Tiny backbone) + trainable
(SharedClassifier + AttentionGate + attention_prior).

Architecture (compressed heads per Gemini's directive):
  · SharedClassifier: 1537 → classifier_hidden (default 32) → 2      (~5 K params)
  · AttentionGate   : 1539 → gate_hidden       (default 16) → 1      (~25 K params)
  · attention_prior : Learnable [num_tasks, 1] scalar per task (warm-startable)

Forward returns (final_logits, logits_task, W_task) for XAI capture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SwinModel


class SIFT_DBTClassifier(nn.Module):
    SWIN_HIDDEN_DIM = 768
    _BACKBONE_CHUNK = 256  # cap on windows per Swin forward — defense in depth

    def __init__(
        self,
        num_tasks: int = 8,
        num_classes: int = 2,
        in_channels: int = 4,
        classifier_hidden: int = 32,
        gate_hidden: int = 16,
        dropout: float = 0.5,
        attention_prior: torch.Tensor = None,
        pretrained_extractor: dict = None,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_classes = num_classes

        self.z_dim = 2 * self.SWIN_HIDDEN_DIM + 1                       # 1537
        self.gate_input_dim = self.z_dim + num_classes                  # 1539

        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=(5, 1), padding=(2, 0)),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )
        self.backbone = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

        # Load Stage 1 pretrained weights BEFORE freezing.
        if pretrained_extractor is not None:
            if "adapter" not in pretrained_extractor or "backbone" not in pretrained_extractor:
                raise KeyError(
                    "pretrained_extractor must have 'adapter' and 'backbone' keys; "
                    f"got {list(pretrained_extractor.keys())}"
                )
            self.adapter.load_state_dict(pretrained_extractor["adapter"], strict=True)
            self.backbone.load_state_dict(pretrained_extractor["backbone"], strict=True)

        # Freeze the whole feature extractor.
        for p in self.adapter.parameters():  p.requires_grad = False
        for p in self.backbone.parameters(): p.requires_grad = False

        self.shared_classifier = nn.Sequential(
            nn.Linear(self.z_dim, classifier_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

        self.attention_gate = nn.Sequential(
            nn.Linear(self.gate_input_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        )

        if attention_prior is None:
            self.attention_prior = nn.Parameter(torch.zeros(num_tasks, 1))
        else:
            self.attention_prior = nn.Parameter(
                attention_prior.reshape(num_tasks, 1).float()
            )

    # ── Freeze-safe train() override ─────────────────────────────────────────
    # nn.Module.train() recursively flips every child to train mode, which
    # would put the adapter's BatchNorm2d into training mode and drift its
    # running_mean / running_var during Stage 2 training. Force the frozen
    # feature extractor back to eval() after the recursive call.
    def train(self, mode: bool = True):
        super().train(mode)
        self.adapter.eval()
        self.backbone.eval()
        return self

    def forward(self, ragged_batch: list, ratios: torch.Tensor):
        device = self.attention_prior.device
        ratios = ratios.to(device)

        B = len(ragged_batch)
        assert ratios.shape == (B, self.num_tasks), f"Ratios shape {ratios.shape} mismatch"

        # Flatten: subject-major, task-inner. Assumes every subject has exactly
        # num_tasks tensors (guaranteed by SubjectBundleDataset admission floor).
        flat_tensors = [t for subject in ragged_batch for t in subject]
        split_sizes = [t.shape[0] for t in flat_tensors]

        concat_tensor = torch.cat(flat_tensors, dim=0).to(device)

        # Extract features via the frozen extractor. Chunk the Swin pass to
        # bound peak GPU memory on Jetson (defense-in-depth against the
        # NVML/CUDACachingAllocator crash observed at large batch sizes).
        latents = self._extract_features(concat_tensor)
        latents_split = torch.split(latents, split_sizes, dim=0)

        # Build Z_task per (subject, task) via the flat->2D index.
        task_reprs = []
        for i, task_latents in enumerate(latents_split):
            b, t = divmod(i, self.num_tasks)

            mu = task_latents.mean(dim=0)
            var = task_latents.var(dim=0, unbiased=True)      # ≥2 trials guaranteed
            ratio_val = ratios[b, t].reshape(1)

            z_task = torch.cat([mu, var, ratio_val])
            task_reprs.append(z_task)

        task_repr_batch = torch.stack(task_reprs).reshape(B, self.num_tasks, self.z_dim)

        # Per-task classification + attention-weighted late fusion.
        logits_task = self.shared_classifier(task_repr_batch)                   # [B, T, 2]

        gate_input = torch.cat([task_repr_batch, logits_task.detach()], dim=-1)  # [B, T, 1539]
        attn_scores = self.attention_gate(gate_input) + self.attention_prior.unsqueeze(0)
        W_task = F.softmax(attn_scores, dim=1)                                   # [B, T, 1]

        final_logits = (W_task * logits_task).sum(dim=1)                         # [B, 2]

        return final_logits, logits_task, W_task

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        # Frozen backbone → wrap in no_grad to skip activation checkpointing
        # and release intermediate tensors immediately after each chunk.
        with torch.no_grad():
            x = self.adapter(x)
            x = F.interpolate(x, size=(224, 224), mode="nearest")
            N = x.shape[0]
            if N <= self._BACKBONE_CHUNK:
                return self.backbone(pixel_values=x).last_hidden_state.mean(dim=1)
            outs = []
            for start in range(0, N, self._BACKBONE_CHUNK):
                chunk = x[start:start + self._BACKBONE_CHUNK]
                outs.append(self.backbone(pixel_values=chunk).last_hidden_state.mean(dim=1))
            return torch.cat(outs, dim=0)
