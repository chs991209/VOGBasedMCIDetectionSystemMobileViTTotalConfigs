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
    # Anti-Saccade B isolation: only 2 task types (Horizontal / Vertical anti).
    # task_emb_dim=16 keeps the embedding compact to limit overfitting at this scope.
    def __init__(self, num_classes=2, in_channels=4, num_tasks=2, task_emb_dim=16):
        super().__init__()

        # Spatial Sobel Asymmetric Adapter: [4, 32, 32] -> [3, 32, 32]
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=(5, 1), padding=(2, 0)),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        # Frozen MobileViT Backbone
        self.backbone = MobileViTModel.from_pretrained("apple/mobilevit-small")
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Task Conditioning Embedding
        self.task_embedding = nn.Embedding(num_embeddings=num_tasks, embedding_dim=task_emb_dim)

        # Metric Learning Head (MobileViT feature dim 640 + task_emb_dim → CosineLinear)
        # Dropout 0.5: aggressive regularization for the tiny anti-saccade B dataset (~3k trainable params, ~160 train windows/fold).
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
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
        task_emb = self.task_embedding(task_id)  # [B, task_emb_dim]

        # Fusion
        fused_features = torch.cat([features, task_emb], dim=1)  # [B, 640 + task_emb_dim]

        # Metric Projection
        return self.classifier(fused_features)