"""Full-experiment (8-task) Monte-Carlo Group CV evaluator.

Mirrors the 2-experiment evaluator's layout/conventions: DI'd probe,
per-fold checkpoint dir, logger, optional AugmentedSubset on the train
split. Uses fully-qualified package imports so this works as long as
`src/` is on sys.path (set up by the entry-point caller).
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Subset

from four_error_using.data_processor.data_engineer import AugmentedSubset
from four_error_using.models.task_conditioned_classifier import TransferMobileViTClassifier
from four_error_using.model_trainers.model_trainer import ModelTrainer, _auroc

logger = logging.getLogger(__name__)


class RepetitiveGroupValidator:
    def __init__(
        self,
        dataset,
        max_epochs: int = 500,
        batch_size: int = 32,
        n_splits: int = 30,
        probe=None,
        checkpoint_dir: Optional[Path] = None,
        num_tasks: int = 8,
        augment: bool = False,
        early_stop_patience: int = 40,
        dropout: float = 0.3,
        task_weights: Optional[dict] = None,
        eval_batch_size: Optional[int] = None,
        in_channels: int = 4,
        stratified: bool = False,
        strat_test_counts: Optional[dict] = None,
    ):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # A6000 / Ampere throughput. Inputs are a fixed 256x256 after interpolate,
        # so cuDNN autotune converges to the fastest conv kernels on the first
        # fold and every fold thereafter reuses them. TF32 accelerates the frozen
        # MobileViT matmuls/convs with negligible accuracy impact. These were left
        # off on the Jetson AGX Orin (Ampere-mobile, unified memory) build.
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        self.dataset = dataset
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        # Inference runs in eval() under no_grad, so a larger batch changes nothing
        # numerically — it just saturates the 48 GB A6000. Defaults to 8x the train
        # batch (min 256) unless overridden.
        self.eval_batch_size = int(eval_batch_size) if eval_batch_size else max(256, batch_size * 8)
        self.n_splits = n_splits
        self.probe = probe
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.num_tasks = num_tasks
        self.in_channels = int(in_channels)
        self.augment = augment
        self.early_stop_patience = early_stop_patience
        self.dropout = float(dropout)
        # None → plain (unweighted) soft-vote. Dict {task_id: weight} → weighted vote.
        self.task_weights = dict(task_weights) if task_weights is not None else None

        # Fold sampling. stratified=False → GroupShuffleSplit (random subject share,
        # class ratio drifts fold to fold). stratified=True → fixed per-class TEST
        # subject counts each fold with random membership (grouped, no leakage), so
        # every fold holds the same HC/MCI train/test composition. strat_test_counts
        # maps {class_label: n_test_subjects}; None uses a sensible per-class ~30%.
        self.stratified = bool(stratified)
        self.strat_test_counts = dict(strat_test_counts) if strat_test_counts is not None else None

    def _aggregate_subject_prob(self, pairs):
        """pairs: list of (prob, task_id) for one subject's windows.

        If task_weights is set, returns the weighted mean
        Σ(w_t · p) / Σ(w_t); otherwise the plain mean. Falls back to the
        plain mean when the weighted denominator is 0 (subject has windows
        only in zero-weight tasks — rare; logged once per occurrence)."""
        if self.task_weights is None:
            return float(np.mean([p for p, _ in pairs]))
        num = sum(self.task_weights.get(t, 0.0) * p for p, t in pairs)
        den = sum(self.task_weights.get(t, 0.0) for _, t in pairs)
        if den <= 0.0:
            logger.warning("Subject had only zero-weight tasks; falling back to unweighted mean.")
            return float(np.mean([p for p, _ in pairs]))
        return float(num / den)

    def _infer_subjects(self, model, test_idx, subj_ids, fold_idx):
        model.eval()
        subj_probs: dict = defaultdict(list)  # sid -> list of (prob, task_id)
        subj_labels: dict = {}

        loader = DataLoader(
            Subset(self.dataset, test_idx),
            batch_size=self.eval_batch_size,
            shuffle=False,
            pin_memory=(self.device.type == "cuda"),
        )
        offset = 0
        with torch.no_grad():
            for inputs, tasks, labels in loader:
                inputs = inputs.to(self.device, non_blocking=True)
                tasks = tasks.to(self.device, non_blocking=True)
                probs = torch.softmax(model(inputs, tasks), dim=1)[:, 1].cpu().numpy()

                tasks_np = tasks.cpu().numpy()
                labels_np = labels.numpy()
                sid_slice = subj_ids[offset:offset + len(labels)]

                if self.probe is not None:
                    for i in range(len(labels_np)):
                        self.probe.add_window(
                            fold_idx=fold_idx,
                            subject_id=str(sid_slice[i]),
                            task_id=int(tasks_np[i]),
                            true_label=int(labels_np[i]),
                            prob=float(probs[i]),
                        )

                for sid, p, l, t in zip(sid_slice, probs, labels_np, tasks_np):
                    subj_probs[sid].append((float(p), int(t)))
                    subj_labels[sid] = int(l)
                offset += len(labels)

        return {
            sid: (subj_labels[sid], self._aggregate_subject_prob(subj_probs[sid]))
            for sid in subj_probs
        }

    def _stratified_group_splits(self, y_arr, subj_ids):
        """Yield (train_idx, test_idx) window-index arrays for each fold with a
        FIXED per-class test-subject count and random membership.

        Subjects are grouped (all a subject's windows move together → no leakage)
        and, per class label, a fixed number of subjects (self.strat_test_counts,
        defaulting to round(30% ) per class) is drawn into the test split each
        fold. A single seeded RNG advanced across folds keeps the 30 folds
        distinct yet fully reproducible."""
        # Each subject has one label; map subject -> class.
        subj_to_label = {}
        for sid, y in zip(subj_ids, y_arr):
            subj_to_label[sid] = int(y)
        per_class = defaultdict(list)
        for sid, lab in subj_to_label.items():
            per_class[lab].append(sid)
        for lab in per_class:
            per_class[lab] = np.array(sorted(per_class[lab]))  # deterministic base order

        # Resolve test counts per class (default ~30%, min 1 kept in each side).
        counts = {}
        for lab, subs in per_class.items():
            if self.strat_test_counts is not None and lab in self.strat_test_counts:
                n_test = int(self.strat_test_counts[lab])
            else:
                n_test = int(round(0.3 * len(subs)))
            n_test = max(1, min(n_test, len(subs) - 1))  # never empty train or test side
            counts[lab] = n_test

        rng = np.random.RandomState(42)
        for _ in range(self.n_splits):
            test_subjs = set()
            for lab, subs in per_class.items():
                perm = rng.permutation(subs)
                test_subjs.update(perm[:counts[lab]].tolist())
            test_mask = np.array([sid in test_subjs for sid in subj_ids])
            test_idx = np.where(test_mask)[0]
            train_idx = np.where(~test_mask)[0]
            yield train_idx, test_idx

    def run(self):
        y_arr = self.dataset.y.numpy()
        subj_ids = np.array(self.dataset.subject_ids)
        indices = np.arange(len(self.dataset))

        fold_metrics = []
        sampling = "stratified-group" if self.stratified else "grouped (unstratified)"
        logger.info(
            "MC Group CV [%s] — %d folds | device=%s | num_tasks=%d | augment=%s | dropout=%.2f | weighted_vote=%s",
            sampling, self.n_splits, self.device, self.num_tasks, self.augment, self.dropout,
            (self.task_weights if self.task_weights is not None else "off"),
        )

        if self.stratified:
            splits = list(self._stratified_group_splits(y_arr, subj_ids))
        else:
            gss = GroupShuffleSplit(n_splits=self.n_splits, test_size=0.3, random_state=42)
            splits = list(gss.split(indices, y_arr, groups=subj_ids))

        for fold, (train_idx, test_idx) in enumerate(splits):
            train_subjs = set(subj_ids[train_idx])
            test_subjs = set(subj_ids[test_idx])
            assert train_subjs.isdisjoint(test_subjs), "Subject leakage detected!"

            # Realized per-class (HC=0, MCI=1) SUBJECT counts, so the stratification
            # is verifiable in the log rather than merely asserted.
            def _cls_subj_counts(idx):
                s2l = {sid: int(lab) for sid, lab in zip(subj_ids[idx], y_arr[idx])}
                hc = sum(1 for v in s2l.values() if v == 0)
                return hc, len(s2l) - hc
            tr_hc, tr_mci = _cls_subj_counts(train_idx)
            te_hc, te_mci = _cls_subj_counts(test_idx)

            logger.info(
                "=== Fold %02d/%d | train: %d subj (HC=%d MCI=%d) / %d win | test: %d subj (HC=%d MCI=%d) / %d win ===",
                fold + 1, self.n_splits,
                len(train_subjs), tr_hc, tr_mci, len(train_idx),
                len(test_subjs), te_hc, te_mci, len(test_idx),
            )

            model = TransferMobileViTClassifier(
                num_classes=2, in_channels=self.in_channels, num_tasks=self.num_tasks,
                dropout=self.dropout,
            )
            trainer = ModelTrainer(
                model,
                device=self.device,
                checkpoint_dir=self.checkpoint_dir,
                fold_idx=fold,
            )

            train_subset = (
                AugmentedSubset(self.dataset, train_idx)
                if self.augment
                else Subset(self.dataset, train_idx)
            )
            test_subset = Subset(self.dataset, test_idx)

            trained_model = trainer.train_model(
                train_subset,
                test_subset,
                max_epochs=self.max_epochs,
                batch_size=self.batch_size,
                early_stop_patience=self.early_stop_patience,
                eval_batch_size=self.eval_batch_size,
            )

            subj_preds = self._infer_subjects(trained_model, test_idx, subj_ids, fold_idx=fold)

            true_arr = np.array([v[0] for v in subj_preds.values()])
            prob_arr = np.array([v[1] for v in subj_preds.values()])
            pred_arr = (prob_arr > 0.5).astype(int)

            tp = int(np.sum((true_arr == 1) & (pred_arr == 1)))
            tn = int(np.sum((true_arr == 0) & (pred_arr == 0)))
            fp = int(np.sum((true_arr == 0) & (pred_arr == 1)))
            fn = int(np.sum((true_arr == 1) & (pred_arr == 0)))

            acc = (tp + tn) / len(true_arr)
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            auroc = _auroc(true_arr, prob_arr)

            fold_metrics.append({'acc': acc, 'sens': sens, 'spec': spec, 'auroc': auroc})
            logger.info(
                "-> Fold %02d Result | Acc=%.3f Sens=%.3f Spec=%.3f AUROC=%.3f",
                fold + 1, acc, sens, spec, auroc,
            )

        accs = [m['acc'] for m in fold_metrics]
        senss = [m['sens'] for m in fold_metrics]
        specs = [m['spec'] for m in fold_metrics]
        aurocs = [m['auroc'] for m in fold_metrics]

        logger.info("=" * 60)
        logger.info("  MC-CV Results [%s sampling] (%d folds)", sampling, self.n_splits)
        logger.info("  Accuracy    : %.3f ± %.3f", float(np.mean(accs)), float(np.std(accs)))
        logger.info("  Sensitivity : %.3f ± %.3f", float(np.mean(senss)), float(np.std(senss)))
        logger.info("  Specificity : %.3f ± %.3f", float(np.mean(specs)), float(np.std(specs)))
        logger.info("  AUROC       : %.3f ± %.3f", float(np.mean(aurocs)), float(np.std(aurocs)))
        logger.info("=" * 60)

        if self.probe is not None:
            logger.info("Evaluator lifecycle end: writing probe report...")
            self.probe.generate_markdown_report()

        return {
            'accuracy': (float(np.mean(accs)), float(np.std(accs))),
            'sensitivity': (float(np.mean(senss)), float(np.std(senss))),
            'specificity': (float(np.mean(specs)), float(np.std(specs))),
            'auroc': (float(np.mean(aurocs)), float(np.std(aurocs))),
            'fold_metrics': fold_metrics,
        }
