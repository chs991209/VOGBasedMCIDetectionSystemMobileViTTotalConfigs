"""Per-task attention-gate priors for Stage 2 warm-start.

The Stage 2 AttentionGate has a per-task learnable bias `attention_prior` that
is added to the gate's scores BEFORE the across-tasks softmax. Selecting a
prior scheme initialises that bias to a distribution derived from the wvote
probe evidence (`TASK_CONTRIBUTION_RATES.md`), giving the gate a warm start.

Schemes:
  · "none":  zero bias → uniform attention at init (fully learn from scratch)
  · "wvote": legacy wvote scheme  → up-weights vertical B-paradigm (5,6,7),
                                    zeroes A-paradigm (0,4)
  · "loo":   raw per-task LOO Δ AUROC × 15 → probe-optimal directional prior

The bias is ALWAYS learnable (nn.Parameter), so SGD can adjust from the prior.
Worst case: same as "none". Best case: convergence on small N benefits from
the warm start.
"""
from typing import Optional

import torch


# 8-task ordering: (0=HSacA, 1=HSacB, 2=HSacBanti, 3=HSacR,
#                   4=VSacA, 5=VSacB, 6=VSacBanti, 7=VSacR)
ATTENTION_PRIOR_SCHEMES = {
    "none": None,

    # Legacy wvote scheme (produced AUROC 0.791 in Candidate A / wvote soft-vote).
    # Post-softmax approximation: [0.049, 0.081, 0.081, 0.081, 0.049, 0.220, 0.220, 0.220]
    "wvote": [0.0, 0.5, 0.5, 0.5, 0.0, 1.5, 1.5, 1.5],

    # Raw probe-derived: LOO Δ AUROC values × 15 (scale chosen for meaningful
    # softmax spread). Post-softmax approximation:
    #   [0.069, 0.106, 0.102, 0.177, 0.110, 0.115, 0.106, 0.243]
    # Source: outputs/reports/full_experiments_using/run_20260527_205622_full_drop050_pat030_wvote/task_contribution_probe.md
    "loo": [-0.48, -0.045, -0.09, +0.465, -0.015, +0.03, -0.045, +0.78],
}


def resolve_prior(name: str, num_tasks: int = 8) -> Optional[torch.Tensor]:
    """Look up a scheme by name and return a float32 tensor of shape [num_tasks, 1],
    or None if the scheme is "none". Raises ValueError on unknown names or
    length mismatch."""
    if name not in ATTENTION_PRIOR_SCHEMES:
        raise ValueError(
            f"Unknown attention_prior scheme '{name}'. "
            f"Choices: {list(ATTENTION_PRIOR_SCHEMES.keys())}"
        )
    values = ATTENTION_PRIOR_SCHEMES[name]
    if values is None:
        return None
    if len(values) != num_tasks:
        raise ValueError(
            f"Prior '{name}' has {len(values)} values but num_tasks={num_tasks}"
        )
    return torch.tensor(values, dtype=torch.float32).reshape(num_tasks, 1)
