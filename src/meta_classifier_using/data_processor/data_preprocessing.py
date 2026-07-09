"""SubjectBundleDataset — strict-parity subject bundles for the
Distribution-Aware Gated Fusion meta-classifier.

Each item:
    (bundle      [NUM_TASKS, MAX_TRIALS, C, 32, 32]  float32,
     label       int (0 = HC, 1 = MCI),
     subject_id  str)

No padding mask is produced or consumed anywhere — the prior Option C
zero-padding scheme allowed the network to learn the padding pattern as a
class-confound, and has been excised.

Strict-parity rule: every subject must have ≥ MAX_TRIALS valid trials for
each of the 8 tasks. Subjects falling short on any task are dropped from
the dataset entirely, with their IDs logged.
"""
import logging

import numpy as np
import torch
from torch.utils.data import Dataset


MAX_TRIALS = 10

logger = logging.getLogger(__name__)


class SubjectBundleDataset(Dataset):
    NUM_TASKS = 8
    CHW = (4, 32, 32)
    MAX_TRIALS = MAX_TRIALS

    def __init__(self, data_store):
        self.subject_ids = []
        self.labels = []
        self.dropped = []   # list[(sid, group, per_task_counts)]
        bundles_list = []

        for group, subjects in data_store.items():
            label = 0 if group == "HC" else 1
            for sid, epochs in subjects.items():
                per_task = [[] for _ in range(self.NUM_TASKS)]
                for tensor, task_id in epochs:
                    if 0 <= task_id < self.NUM_TASKS:
                        per_task[task_id].append(tensor)

                counts = [len(t) for t in per_task]
                if min(counts) < MAX_TRIALS:
                    self.dropped.append((sid, group, counts))
                    continue

                bundle = np.zeros(
                    (self.NUM_TASKS, MAX_TRIALS) + self.CHW, dtype=np.float32
                )
                for t in range(self.NUM_TASKS):
                    trials = per_task[t][:MAX_TRIALS]   # first-N truncation
                    bundle[t] = np.stack(trials, axis=0)
                bundles_list.append(bundle)
                self.subject_ids.append(sid)
                self.labels.append(label)

        if self.dropped:
            logger.info(
                "Strict-parity drop: %d subject(s) had <%d trials on ≥1 task and were excluded.",
                len(self.dropped), MAX_TRIALS,
            )
            for sid, group, counts in self.dropped:
                logger.info("  · dropped %s (%s) — per-task counts = %s", sid, group, counts)

        if bundles_list:
            self.X = torch.from_numpy(np.stack(bundles_list, axis=0))  # [N, 8, T, C, 32, 32]
        else:
            self.X = torch.empty(0, self.NUM_TASKS, MAX_TRIALS, *self.CHW, dtype=torch.float32)
        self.y = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.subject_ids[idx]

    @property
    def shape(self):
        return tuple(self.X.shape) if len(self.X) else (0,)
