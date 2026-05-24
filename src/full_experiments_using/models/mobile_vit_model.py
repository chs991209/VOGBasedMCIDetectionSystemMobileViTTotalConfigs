import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MobileViTModel


class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, scale=10.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.scale = scale

    def forward(self, x):
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        return self.scale * F.linear(x_norm, w_norm)


class TransferMobileViTClassifier(nn.Module):
    def __init__(self, num_classes=2, in_channels=4, num_tasks=8, task_emb_dim=32, dropout=0.3):
        super().__init__()
        self.dropout_p = float(dropout)

        # 1. Spatial Sobel Asymmetric Adapter: [4, 32, 32] -> [3, 32, 32]
        # Enhances vertical transients (time-axis) while preserving frequency resolution
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=(5, 1), padding=(2, 0)),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        # 2. Frozen MobileViT Backbone
        self.backbone = MobileViTModel.from_pretrained("apple/mobilevit-small")
        for p in self.backbone.parameters():
            p.requires_grad = False

        # 3. Task Conditioning Embedding
        self.task_embedding = nn.Embedding(num_embeddings=num_tasks, embedding_dim=task_emb_dim)

        # 4. Metric Learning Head (MobileViT feature dim 640 + task emb dim 32 = 672)
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_p),
            CosineLinear(in_features=640 + task_emb_dim, out_features=num_classes, scale=10.0)
        )

    def forward(self, x: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        # Spatial filtering
        x = self.adapter(x)

        # Exact 8x scaling using nearest neighbor to preserve vertical edge gradients
        x = F.interpolate(x, size=(256, 256), mode='nearest')

        # Visual feature extraction
        outputs = self.backbone(pixel_values=x)
        features = outputs.last_hidden_state.mean(dim=[2, 3])  # [B, 640]

        # Task context injection
        task_emb = self.task_embedding(task_id)  # [B, 32]

        # Fusion
        fused_features = torch.cat([features, task_emb], dim=1)  # [B, 672]

        # Metric Projection
        return self.classifier(fused_features)