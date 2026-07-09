"""Datasets for the Swin SIFT-DBT candidate.

Two datasets, both consuming the 4-error CWT cache
(`outputs/cache/data_store_meta_4err.pkl` — 3-tuple events):

  · FlatWindowDataset — one item per trial-window (for Stage 1 pretraining).
    Yields (tensor [4,32,32], task_id, label, subject_id).

  · SubjectBundleDataset — one item per subject (for Stage 2 inference / gate
    training). Yields (subject_tasks: List[Tensor[T_ij, 4,32,32]], ratios [T],
    label, subject_id). Ragged trials per (subject, task).

`ragged_collate` packs a batch of subject-items into
  (bundle_batch, ratios [B, T], labels [B], sids [B]).
"""
import logging
from collections import defaultdict
from typing import Iterable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


NUM_TASKS_DEFAULT = 8
CHW = (4, 32, 32)


# ────────────────────────────────────────────────────────────────────────────
# Flat window dataset (Stage 1)
# ────────────────────────────────────────────────────────────────────────────

class FlatWindowDataset(Dataset):
    """One item per trial-window. Task-agnostic classification input for
    Stage 1's backbone tuner (window-level supervised learning).

    Item: (tensor [4, 32, 32] float32, task_id int, label int, subject_id str).
    Handles both 2-tuple (legacy cache) and 3-tuple (four_error cache) event
    tuples; the cross-axis ratio is ignored at Stage 1.
    """

    NUM_TASKS = NUM_TASKS_DEFAULT
    CHW = CHW

    def __init__(self, data_store, keep_task_ids: Optional[Iterable[int]] = None):
        self.keep_task_ids = (
            set(keep_task_ids) if keep_task_ids is not None else set(range(self.NUM_TASKS))
        )
        X, T, y, sids = [], [], [], []
        for group, subjects in data_store.items():
            label = 0 if group == "HC" else 1
            for sid, epochs in subjects.items():
                for ev in epochs:
                    tensor, task_id = ev[0], ev[1]  # ignore optional ratio (ev[2])
                    if task_id not in self.keep_task_ids:
                        continue
                    X.append(tensor)
                    T.append(int(task_id))
                    y.append(label)
                    sids.append(str(sid))
        self.X = torch.from_numpy(np.stack(X, axis=0)).float() if X else torch.empty(0, *CHW)
        self.T = torch.tensor(T, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
        self.subject_ids = sids

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.T[idx], self.y[idx], self.subject_ids[idx]


# ────────────────────────────────────────────────────────────────────────────
# Ragged subject-bundle dataset (Stage 2)
# ────────────────────────────────────────────────────────────────────────────

def ragged_collate(batch):
    """DataLoader collate that passes ragged per-subject bundles through
    without stacking. Returns:
        (bundle_batch: List[List[Tensor[T_ij, 4, 32, 32]]],
         ratios      : Tensor[B, NUM_TASKS],
         labels      : Tensor[B],
         sids        : List[str])
    """
    bundle_batch = [item[0] for item in batch]
    ratios = torch.stack([item[1] for item in batch])
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
    sids = [item[3] for item in batch]
    return bundle_batch, ratios, labels, sids


class SubjectBundleDataset(Dataset):
    """One item per subject. Ragged trials per (subject, task): no upper cap,
    no zero-padding. Bootstrap-fills tasks with exactly 1 trial to enable
    variance computation (var over 2 duplicated trials = 0.0, semantically
    "no cognitive variability observed").

    Admission floor: subjects with < `min_trials` on any kept task are dropped
    (their IDs are logged).

    Cross-axis ratio (from cache 3-tuples) is aggregated per (subject, task)
    as the mean across trials, and exposed as a companion `[NUM_TASKS]` tensor
    on each item.

    Item: (subject_tasks: List[Tensor[T_ij, 4, 32, 32]],
           ratios: Tensor[NUM_TASKS],
           label: int,
           subject_id: str).
    """

    CHW = CHW

    def __init__(
        self,
        data_store,
        keep_task_ids: Iterable[int] = tuple(range(NUM_TASKS_DEFAULT)),
        min_trials: int = 1,
    ):
        self.KEEP_TASK_IDS = tuple(keep_task_ids)
        self.NUM_TASKS = len(self.KEEP_TASK_IDS)
        self.MIN_TRIALS = int(min_trials)

        self.subject_ids = []
        self.labels = []
        self.dropped = []
        self.bundles_list = []
        self.ratios_list = []

        for group, subjects in data_store.items():
            label = 0 if group == "HC" else 1
            for sid, epochs in subjects.items():
                per_task = {tid: [] for tid in self.KEEP_TASK_IDS}
                per_task_ratios = {tid: [] for tid in self.KEEP_TASK_IDS}
                for ev in epochs:
                    if len(ev) == 3:
                        tensor, task_id, ratio = ev
                    else:
                        tensor, task_id = ev
                        ratio = 1.0
                    if task_id in per_task:
                        per_task[task_id].append(tensor)
                        per_task_ratios[task_id].append(float(ratio))

                counts = [len(per_task[tid]) for tid in self.KEEP_TASK_IDS]
                if min(counts) < self.MIN_TRIALS:
                    self.dropped.append((sid, group, counts))
                    continue

                subject_tasks, subject_ratios = [], []
                for tid in self.KEEP_TASK_IDS:
                    trials = per_task[tid]
                    ratios = per_task_ratios[tid]
                    # Bootstrap-fill to keep var() finite.
                    if len(trials) == 1:
                        trials = [trials[0], trials[0]]
                        ratios = [ratios[0], ratios[0]] if ratios else [1.0, 1.0]
                    task_tensor = torch.from_numpy(np.stack(trials, axis=0)).float()
                    subject_tasks.append(task_tensor)
                    subject_ratios.append(float(np.mean(ratios)) if ratios else 1.0)

                self.bundles_list.append(subject_tasks)
                self.ratios_list.append(torch.tensor(subject_ratios, dtype=torch.float32))
                self.subject_ids.append(sid)
                self.labels.append(label)

        if self.dropped:
            logger.info(
                "SubjectBundleDataset admission floor (min_trials=%d over %d task(s)): "
                "%d subject(s) dropped.",
                self.MIN_TRIALS, self.NUM_TASKS, len(self.dropped),
            )

        self.y = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            self.bundles_list[idx],
            self.ratios_list[idx],
            self.y[idx],
            self.subject_ids[idx],
        )

    @property
    def shape(self) -> str:
        return f"Ragged[N={len(self.labels)}, Tasks={self.NUM_TASKS}, Trials=Dynamic]"


# ────────────────────────────────────────────────────────────────────────────
# Cross-fold indexing helper
# ────────────────────────────────────────────────────────────────────────────

def flat_indices_for_subjects(flat_ds: FlatWindowDataset, subject_ids: Iterable[str]) -> list:
    """Return the list of `FlatWindowDataset` indices whose `subject_ids[i]`
    is in the given set. Used by the orchestrator to restrict Stage 1 training
    to the fold's train subjects."""
    keep = set(subject_ids)
    return [i for i, s in enumerate(flat_ds.subject_ids) if s in keep]
