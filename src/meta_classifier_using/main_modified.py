import torch
import torch.nn as nn
from transformers import MobileViTModel


class LeakFreeMetaClassifier(nn.Module):
    def __init__(self, num_tasks=4, vit_hidden_dim=640, dropout_rate=0.5):
        super().__init__()
        self.num_tasks = num_tasks
        self.vit_hidden_dim = vit_hidden_dim

        # ---------------------------------------------------------
        # [1] 구조 변환부 (Adapter)
        # 4채널 오차 CWT를 MobileViT가 이해하는 3채널(RGB)로 변환
        # ---------------------------------------------------------
        self.channel_adapter = nn.Conv2d(
            in_channels=4,
            out_channels=3,
            kernel_size=3,
            padding=1
        )

        # ---------------------------------------------------------
        # [2] 특징 추출부 (Frozen MobileViT Backbone)
        # ---------------------------------------------------------
        self.backbone = MobileViTModel.from_pretrained("apple/mobilevit-small")
        # 백본 가중치 동결 (특징 추출기로만 사용)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 변동성(mu + var) 병합으로 인해 차원이 2배가 됨 (640 * 2 = 1280)
        self.fused_dim = self.vit_hidden_dim * 2

        # ---------------------------------------------------------
        # [3] 메타 게이팅부 (Task Gating Network)
        # 각 태스크의 (평균 || 분산) 벡터를 평가하여 가중치(0~1) 산출
        # ---------------------------------------------------------
        self.task_gate = nn.Sequential(
            nn.Linear(self.fused_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        self.gate_softmax = nn.Softmax(dim=1)  # 4개 태스크 간의 상대적 중요도 평가

        # ---------------------------------------------------------
        # [4] 최종 예측부 (Final Classification Head)
        # ---------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(self.fused_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 2)  # [HC vs MCI]
        )

    def forward(self, x):
        """
        입력 텐서 규격의 엄격한 대칭성 보장
        x shape: [Batch, 4 (Tasks), 10 (Trials), 4 (Channels), 32 (H), 32 (W)]
        """
        B, TASKS, TRIALS, C, H, W = x.shape

        # [검증 가드레일] 데이터 누수 원천 차단: 마스크가 없으므로 규격이 완벽해야 함
        assert TASKS == self.num_tasks, f"Expected {self.num_tasks} tasks, got {TASKS}"
        assert TRIALS == 10, f"Expected strictly 10 trials to prevent leakage, got {TRIALS}"

        # ---------------------------------------------------------
        # 단계 1: 독립적 순차 라우팅 (Independent Trial Routing)
        # ---------------------------------------------------------
        # 배치, 태스크, 트라이얼을 하나의 평면(Batch 차원)으로 붕괴시킴
        # shape: [B * 4 * 10, 4, 32, 32]
        x_flat = x.view(B * TASKS * TRIALS, C, H, W)

        # Adapter 통과: [B * 40, 3, 32, 32]
        x_adapted = self.channel_adapter(x_flat)

        # 동결된 백본 통과 (그라디언트 계산 생략으로 연산 효율 극대화)
        self.backbone.eval()
        with torch.no_grad():
            vit_outputs = self.backbone(pixel_values=x_adapted)
            # MobileViT의 최종 pooler_output (GAP 적용됨)
            # shape: [B * 40, 640]
            latents = vit_outputs.pooler_output

            # ---------------------------------------------------------
        # 단계 2: 물리적 변동성 계산 (Aleatoric Uncertainty)
        # 트랜스포머의 어텐션을 대체하는 우리 아키텍처의 핵심 엔진
        # ---------------------------------------------------------
        # 구조 복원: [B, 4, 10, 640]
        latents_restored = latents.view(B, TASKS, TRIALS, self.vit_hidden_dim)

        # 10번의 시도(Trials) 차원(dim=2)을 기준으로 평균과 분산 계산
        # shape: [B, 4, 640]
        mu_t = torch.mean(latents_restored, dim=2)
        var_t = torch.var(latents_restored, dim=2, unbiased=True)

        # [평균 || 분산] 병합: 모델이 '기본 오차'와 '인지적 흔들림'을 동시에 보도록 강제
        # shape: [B, 4, 1280]
        task_repr = torch.cat([mu_t, var_t], dim=-1)

        # ---------------------------------------------------------
        # 단계 3: 동적 메타 게이팅 (Dynamic Meta-Gating)
        # ---------------------------------------------------------
        # 각 태스크의 1280차원 정보를 바탕으로 게이트 점수 산출
        # shape: [B, 4, 1]
        gate_logits = self.task_gate(task_repr)
        gate_weights = self.gate_softmax(gate_logits)

        # 게이트 가중치를 각 태스크 벡터에 곱한 후 결합 (Weighted Sum)
        # shape: [B, 1280]
        fused_repr = torch.sum(task_repr * gate_weights, dim=1)

        # ---------------------------------------------------------
        # 단계 4: 최종 진단 예측
        # ---------------------------------------------------------
        # shape: [B, 2]
        logits = self.classifier(fused_repr)

        return logits, gate_weights  # gate_weights는 추후 설명력(XAI) 분석을 위해 반환