"""Candidate F — Swin-Tiny + SIFT-DBT Attention-Weighted Vote (current target).

Two-stage hybrid pipeline:
  · Stage 1 (Domain-Adaptive Pretrain): Swin-Tiny fully unfrozen, trained on
    flat 8-task CWT windows for feature learning. Task-agnostic Linear(768, 2)
    head; no task embedding. Per-fold, uses only the training-set subjects.
  · Stage 2 (Clinical Fine-Tune): Swin-Tiny frozen (params + BN stats), plus
    the pretrained adapter frozen too. Per-task μ ‖ σ² over ragged trials → a
    Shared Classifier produces per-task logits; a parallel Attention Gate
    (seeing both Z_task AND the task's own logits) produces attention weights.
    Late-fusion via weighted sum of the per-task logits → final logits.

Master Context §1/§5 deviation: Stage 2 operates on all 8 tasks
(NUM_TASKS = 8), per Gemini's SIFT-DBT directive.

Return signature from the Stage 2 forward: (final_logits, logits_task, W_task)
— enables explicit XAI narratives via the probe generator.
"""
