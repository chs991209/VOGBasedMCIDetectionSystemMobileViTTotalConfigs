"""SubjectBundleDataset (Solution D: Dynamic Latent Aggregation)

환자의 시도(Trial) 횟수가 제각각인 현상을 그대로 수용하는 가변(Ragged) 데이터셋입니다.
- Strict-parity 상한선(MAX_TRIALS)을 완전히 제거하여 환자 데이터를 버리지 않습니다.
- MIN_TRIALS (기본값 1)을 두어, 최소한 1번이라도 성공한 환자를 모두 구제합니다.
- 단, 분산(Variance) 연산 시 NaN 발생을 막기 위해, 트라이얼이 1개뿐인 태스크는
  해당 트라이얼을 1회 복제(Bootstrap)하여 크기 2의 텐서로 만들어 분산을 0으로 산출하게 합니다.
- DataLoader가 가변 텐서를 배치로 묶을 수 있도록 `ragged_collate` 함수를 제공합니다.
"""
import logging
import numpy as np
import torch
from torch.utils.data import Dataset

KEEP_TASK_IDS = (0, 3, 4, 7)
NUM_TASKS = len(KEEP_TASK_IDS)

logger = logging.getLogger(__name__)

def ragged_collate(batch):
    """
    DataLoader 가변 텐서 배치 처리 + 교차축 비율(cross-axis ratio) 전달.
    batch: List of tuples -> [(task_tensors_list, ratios_per_task, label, sid), ...]
    """
    bundle_batch = [item[0] for item in batch]
    ratios = torch.stack([item[1] for item in batch])     # [B, NUM_TASKS]
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
    sids = [item[3] for item in batch]
    return bundle_batch, ratios, labels, sids


class SubjectBundleDataset(Dataset):
    CHW = (4, 32, 32)

    def __init__(self, data_store, keep_task_ids=KEEP_TASK_IDS, min_trials=1):
        self.KEEP_TASK_IDS = tuple(keep_task_ids)
        self.NUM_TASKS = len(self.KEEP_TASK_IDS)
        self.MIN_TRIALS = int(min_trials)

        self.subject_ids = []
        self.labels = []
        self.dropped = []   # list[(sid, group, per_task_counts)]
        self.bundles_list = [] # Ragged list: [Subject] -> [Tasks] -> [Trials, C, H, W]
        self.ratios_list = []  # parallel list: [Subject] -> Tensor[NUM_TASKS] cross-axis ratios

        for group, subjects in data_store.items():
            label = 0 if group == "HC" else 1
            for sid, epochs in subjects.items():
                # Accept either legacy 2-tuple or new 3-tuple events.
                # 3-tuple: (tensor, task_id, cross_axis_ratio)
                per_task = {tid: [] for tid in self.KEEP_TASK_IDS}
                per_task_ratios = {tid: [] for tid in self.KEEP_TASK_IDS}
                for ev in epochs:
                    if len(ev) == 3:
                        tensor, task_id, ratio = ev
                    else:
                        tensor, task_id = ev
                        ratio = 1.0  # neutral default for legacy caches
                    if task_id in per_task:
                        per_task[task_id].append(tensor)
                        per_task_ratios[task_id].append(float(ratio))

                counts = [len(per_task[tid]) for tid in self.KEEP_TASK_IDS]

                # 하나라도 MIN_TRIALS 미만이면 드롭
                if min(counts) < self.MIN_TRIALS:
                    self.dropped.append((sid, group, counts))
                    continue

                subject_tasks = []
                subject_ratios = []
                for tid in self.KEEP_TASK_IDS:
                    trials = per_task[tid]
                    ratios = per_task_ratios[tid]

                    # [Bootstrapping] 1번만 성공한 경우, 분산 연산(NaN 방지)을 위해 1회 복제
                    if len(trials) == 1:
                        trials = [trials[0], trials[0]]
                        ratios = [ratios[0], ratios[0]] if ratios else [1.0, 1.0]

                    task_tensor = torch.from_numpy(np.stack(trials, axis=0)).float() # [T, 4, 32, 32]
                    subject_tasks.append(task_tensor)
                    # Mean cross-axis ratio across this subject's trials of this task
                    subject_ratios.append(float(np.mean(ratios)) if ratios else 1.0)

                self.bundles_list.append(subject_tasks)
                self.ratios_list.append(torch.tensor(subject_ratios, dtype=torch.float32))
                self.subject_ids.append(sid)
                self.labels.append(label)

        if self.dropped:
            logger.info(
                "Admission Floor drop (over %d kept task(s) at MIN_TRIALS=%d): "
                "%d subject(s) had <%d trials on ≥1 kept task.",
                self.NUM_TASKS, self.MIN_TRIALS, len(self.dropped), self.MIN_TRIALS,
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
    def shape(self):
        return f"Ragged[N={len(self.labels)}, Tasks={self.NUM_TASKS}, Trials=Dynamic]"