"""TransferMobileViTClassifier — assembles the task-conditioned classifier.

Late-fusion pipeline (only the adapter, task embedding and head are trained;
the MobileViT backbone is frozen):

    x [B, in_channels, 32, 32]
      -> ConvAdapter          -> [B, 3, 32, 32]
      -> FrozenMobileViTBackbone         -> features [B, 640]
    task_id [B] -> TaskEmbedding         -> task_emb [B, 32]
      concat(features, task_emb)         -> fused    [B, 672]
      -> Dropout -> CosineLinear         -> logits   [B, num_classes]

Optional entropy input (entropy_dim>0): the input carries `entropy_dim` extra
constant planes after the CWT channels; they are split off BEFORE the frozen
backbone (so they aren't washed out) and joined to the fused feature just before
the head.
"""
import torch
import torch.nn as nn

from four_error_using.models.layers.cosine_linear import CosineLinear
from four_error_using.models.layers.frozen_mobilevit_backbone import FrozenMobileViTBackbone
from four_error_using.models.layers.conv_adapter import ConvAdapter
from four_error_using.models.layers.task_embedding import TaskEmbedding


class TransferMobileViTClassifier(nn.Module):
    def __init__(self, num_classes=2, in_channels=4, num_tasks=8, task_emb_dim=32,
                 dropout=0.3, entropy_dim=0, backbone="mobilevit-small"):
        super().__init__()
        self.dropout_p = float(dropout)
        self.in_channels = in_channels
        self.entropy_dim = entropy_dim

        self.adapter = ConvAdapter(in_channels=in_channels)
        self.backbone = FrozenMobileViTBackbone(pretrained=backbone)
        self.task_embedding = TaskEmbedding(num_tasks=num_tasks, task_emb_dim=task_emb_dim)
        if entropy_dim > 0:
            self.entropy_norm = nn.BatchNorm1d(entropy_dim)  # standardize the scalar(s)
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_p),
            CosineLinear(in_features=self.backbone.feature_dim + task_emb_dim + entropy_dim,
                         out_features=num_classes, scale=10.0),
        )

    def forward(self, x: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        """scalogram + task id -> HC/MCI logits (adapter → frozen backbone → head)."""
        extra = None
        if self.entropy_dim > 0:
            # extra planes ride after the CWT channels; recover the scalar(s)
            extra = x[:, self.in_channels:self.in_channels + self.entropy_dim].mean(dim=(2, 3))
            extra = self.entropy_norm(extra)
            x = x[:, :self.in_channels]

        x = self.adapter(x)                              # spatial filtering -> [B, 3, 32, 32]
        features = self.backbone(x)                      # frozen features   -> [B, 640]
        task_emb = self.task_embedding(task_id)          # task context      -> [B, 32]
        parts = [features, task_emb]
        if extra is not None:
            parts.append(extra)                          # entropy joins here (post-backbone)
        fused = torch.cat(parts, dim=1)                  # [B, 672 (+entropy_dim)]
        return self.classifier(fused)                    # metric projection -> logits
