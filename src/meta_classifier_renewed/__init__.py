"""Meta-classifier (renewed) — LeakFreeMetaClassifier with 4-task strict parity.

Adaptation of the user-supplied `LeakFreeMetaClassifier` (main_modified.py)
into the standard `<role>_using/`-style pipeline structure.

Key differences vs `meta_classifier_using/`:
  - NUM_TASKS = 4 (HSacA, HSacR, VSacA, VSacR — the "plenty" tasks per PROBLEM.md).
  - Strict-parity rule still in force (MAX_TRIALS=10) but evaluated on 4 tasks
    instead of 8 → 32/37 subjects survive (HC=14, MCI=18).
  - Gate: softmax across tasks (attention budget = 1), shared per-task Linear.
  - Fusion: gate-weighted sum → [B, 1280] (not concat).
  - Forward returns (logits, gate_weights) for XAI inspection.
"""
