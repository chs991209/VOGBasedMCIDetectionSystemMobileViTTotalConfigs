"""DynamicLatentClassifier (Solution D)

각 환자의 시도(Trial) 횟수가 다른 가변 텐서를 완벽하게 지원하는 동적 집계 분류기입니다.
배치(Batch) 내의 모든 환자/태스크/트라이얼 텐서를 1차원 리스트로 평탄화(Flatten)하여
동결된 MobileViT를 한 번에 효율적으로 통과시킨 뒤, 원래의 환자 및 태스크별로 복구하여
평균과 분산을 계산하는 고급 텐서 라우팅(Advanced Tensor Routing) 기법을 사용합니다.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MobileViTModel

class DynamicLatentClassifier(nn.Module):
    def __init__(
        self,
        num_tasks: int = 2,
        vit_hidden_dim: int = 640,
        dropout_rate: float = 0.5,
        shared_backbone: nn.Module = None,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.vit_hidden_dim = vit_hidden_dim

        # [1] 구조 변환부 (Adapter)
        self.channel_adapter = nn.Conv2d(
            in_channels=4, out_channels=3, kernel_size=3, padding=1
        )

        # [2] 특징 추출부 (Frozen MobileViT Backbone)
        if shared_backbone is not None:
            self.backbone = shared_backbone
        else:
            self.backbone = MobileViTModel.from_pretrained("apple/mobilevit-small")
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        # μ ‖ σ² over trials → 2D, plus one scalar cross-axis ratio per task.
        # The +1 is the cross_axis_ratio feature (cross-energy / active-energy).
        self.fused_dim = self.vit_hidden_dim * 2 + 1

        # [3] 메타 게이팅부 (Task Gating Network)
        self.task_gate = nn.Sequential(
            nn.Linear(self.fused_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        # [4] 최종 예측부 (Final Classification Head)
        self.classifier = nn.Sequential(
            nn.Linear(self.fused_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 2),
        )

    def forward(self, ragged_batch, ratios):
        """
        ragged_batch: List[List[Tensor of [T_ij, 4, 32, 32]]] (Outer: Batch, Inner: Tasks)
        ratios:       Tensor[B, NUM_TASKS] — per-(subject, task) cross-axis energy ratio
                      (mean over the subject's trials of that task), computed at cache-build
                      time from RAW magnitudes (pre-normalization).
        Returns: (logits [B, 2], gate_weights [B, num_tasks, 1])
        """
        B = len(ragged_batch)
        device = self.channel_adapter.weight.device
        ratios = ratios.to(device)

        # --- Stage 1: Flatten Batches for Efficient Execution ---
        flat_tensors = []
        split_sizes = []

        for b in range(B):
            assert len(ragged_batch[b]) == self.num_tasks, f"Task count mismatch in batch {b}"
            for t in range(self.num_tasks):
                t_tensor = ragged_batch[b][t]
                flat_tensors.append(t_tensor)
                split_sizes.append(t_tensor.shape[0])

        # 모든 환자의 모든 트라이얼을 병합 -> [Total_Trials, 4, 32, 32]
        concat_tensor = torch.cat(flat_tensors, dim=0).to(device)
        x_adapted = self.channel_adapter(concat_tensor)

        # Frozen Backbone 통과 (그라디언트는 통과하여 어댑터에 닿음)
        self.backbone.eval()
        vit_outputs = self.backbone(pixel_values=x_adapted)
        latents = vit_outputs.pooler_output  # [Total_Trials, 640]

        # --- Stage 2: Unflatten & Dynamic Aggregation (μ ‖ σ²) ---
        # 다시 환자와 태스크 별로 잘라냄
        latents_split = torch.split(latents, split_sizes, dim=0)

        task_repr_batch = []
        split_idx = 0

        for b in range(B):
            b_task_reprs = []
            for t in range(self.num_tasks):
                task_latents = latents_split[split_idx]  # [T_ij, 640]

                mu_t = task_latents.mean(dim=0)          # [640]
                var_t = task_latents.var(dim=0, unbiased=True) # [640] (T=2 이상이므로 안전)

                b_task_reprs.append(torch.cat([mu_t, var_t], dim=-1)) # [1280]
                split_idx += 1

            task_repr_batch.append(torch.stack(b_task_reprs, dim=0)) # [Tasks, 1280]

        task_repr_batch = torch.stack(task_repr_batch, dim=0) # [B, Tasks, 1280]

        # Append the per-task cross-axis ratio as an extra feature.
        # ratios: [B, NUM_TASKS] → [B, NUM_TASKS, 1] → concat → [B, NUM_TASKS, 1281]
        task_repr_batch = torch.cat(
            [task_repr_batch, ratios.unsqueeze(-1)], dim=-1
        )

        # --- Stage 3: Dynamic Meta-Gating ---
        gate_logits = self.task_gate(task_repr_batch)         # [B, Tasks, 1]
        gate_weights = F.softmax(gate_logits, dim=1)          # over tasks

        fused_repr = (task_repr_batch * gate_weights).sum(dim=1)  # [B, 1280]

        # --- Stage 4: Diagnosis ---
        logits = self.classifier(fused_repr)                  # [B, 2]
        return logits, gate_weights