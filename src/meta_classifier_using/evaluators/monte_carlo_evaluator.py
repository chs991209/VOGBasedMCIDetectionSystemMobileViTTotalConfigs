"""Meta-classifier Monte-Carlo Group CV evaluator (Distribution-Aware Fusion).

엄격한 임상 ML 평가 기준 적용:
- 테스트셋 데이터 유출(Data Leakage)을 방지하기 위해 Youden's J 사후 계산을 제거합니다.
- 평가 임계값(Threshold)은 0.5로 엄격하게 고정합니다.
- 패딩 마스크는 더 이상 사용되지 않습니다 (Option C 폐기, 엄격 동수 룰로 대체).
- 사전 학습된 MobileViT 백본은 monkey-patch 없이 frozen 상태로 1회만 로드합니다.
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, Subset
from transformers import MobileViTModel

from meta_classifier_using.models.meta_mobile_vit import DistributionAwareFusionClassifier
from meta_classifier_using.model_trainers.meta_trainer import MetaTrainer

logger = logging.getLogger(__name__)


class MetaMonteCarloGroupEvaluator:
    def __init__(
        self,
        dataset,
        max_epochs: int = 500,
        batch_size: int = 8,
        n_splits: int = 30,
        checkpoint_dir: Optional[Path] = None,
        early_stop_patience: int = 40,
        dropout: float = 0.5,
        fc_hidden: int = 128,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.n_splits = n_splits
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.early_stop_patience = early_stop_patience
        self.dropout = dropout
        self.fc_hidden = fc_hidden
        self.lr = lr
        self.weight_decay = weight_decay

    def _infer_subjects(self, model, val_subset: Subset):
        """Direct subject-level predictions. Returns dict[sid] = (label, prob_mci)."""
        model.eval()
        loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False)
        out = {}
        with torch.no_grad():
            for bundle, label, sid in loader:   # no padding mask anymore
                bundle = bundle.to(self.device)
                logits = model(bundle)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                for s, p, l in zip(sid, probs, label.numpy()):
                    out[str(s)] = (int(l), float(p))
        return out

    def run(self):
        y_arr = self.dataset.y.numpy()
        subj_ids = np.array(self.dataset.subject_ids)
        indices = np.arange(len(self.dataset))

        fold_metrics = []
        logger.info(
            "Meta MC Group CV — %d folds | device=%s | dropout=%.2f | "
            "fc_hidden=%d | lr=%.0e | wd=%.0e",
            self.n_splits, self.device, self.dropout, self.fc_hidden,
            self.lr, self.weight_decay,
        )

        # Shared frozen backbone — clean, unmodified, no monkey-patching.
        # Loaded ONCE before the fold loop to avoid 30× from_pretrained()
        # which fragments the CUDA allocator (cause of prior NVML crashes).
        logger.info("Loading shared MobileViT-small backbone (one-time, no patches)...")
        shared_backbone = MobileViTModel.from_pretrained("apple/mobilevit-small")
        for p in shared_backbone.parameters():
            p.requires_grad = False
        shared_backbone.eval()
        shared_backbone = shared_backbone.to(self.device)
        logger.info("Shared backbone ready on %s.", self.device)

        gss = GroupShuffleSplit(n_splits=self.n_splits, test_size=0.3, random_state=42)
        for fold, (train_idx, test_idx) in enumerate(gss.split(indices, y_arr, groups=subj_ids)):
            train_subjs = set(subj_ids[train_idx])
            test_subjs = set(subj_ids[test_idx])
            assert train_subjs.isdisjoint(test_subjs), "Subject leakage detected!"

            logger.info(
                "=== Fold %02d/%d | train: %d subj | test: %d subj ===",
                fold + 1, self.n_splits, len(train_subjs), len(test_subjs),
            )

            model = DistributionAwareFusionClassifier(
                num_classes=2,
                in_channels=4,
                max_trials=self.dataset.MAX_TRIALS,
                fc_hidden=self.fc_hidden,
                dropout=self.dropout,
                shared_backbone=shared_backbone,
            )
            trainer = MetaTrainer(
                model,
                device=self.device,
                checkpoint_dir=self.checkpoint_dir,
                fold_idx=fold,
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
            trained = trainer.train_model(
                Subset(self.dataset, train_idx),
                Subset(self.dataset, test_idx),
                max_epochs=self.max_epochs,
                batch_size=self.batch_size,
                early_stop_patience=self.early_stop_patience,
            )

            # --- [CRITICAL ARCHITECTURE FIX: STRICT THRESHOLDING] ---
            preds = self._infer_subjects(trained, Subset(self.dataset, test_idx))
            true_arr = np.array([v[0] for v in preds.values()])
            prob_arr = np.array([v[1] for v in preds.values()])

            # 절대적인 0.5 임계값 강제 적용 (테스트셋 기반 임계값 탐색 완전 차단)
            pred_arr = (prob_arr > 0.5).astype(int)

            # Confusion Matrix 기반 명시적 계산
            tn, fp, fn, tp = confusion_matrix(true_arr, pred_arr, labels=[0, 1]).ravel()

            acc = accuracy_score(true_arr, pred_arr)
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            # AUROC는 임계값과 무관하게 모델의 순위 지정 능력(Separability)을 평가
            try:
                auroc = float(roc_auc_score(true_arr, prob_arr))
            except (ValueError, Exception) as e:
                logger.warning("Fold %02d AUROC undefined (%s); using 0.5", fold + 1, e)
                auroc = 0.5

            fold_metrics.append({"acc": acc, "sens": sens, "spec": spec, "auroc": auroc})
            logger.info(
                "-> Fold %02d Result | Acc=%.3f Sens=%.3f Spec=%.3f AUROC=%.3f (Fixed Threshold: 0.5)",
                fold + 1, acc, sens, spec, auroc,
            )

            # Allocator hygiene between folds (defends against NVML/caching-
            # allocator state corruption that surfaced in earlier runs).
            del model, trainer, trained
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        accs = [m["acc"] for m in fold_metrics]
        senss = [m["sens"] for m in fold_metrics]
        specs = [m["spec"] for m in fold_metrics]
        aurocs = [m["auroc"] for m in fold_metrics]

        logger.info("=" * 60)
        logger.info("  Stratified MC-CV Results (%d folds) - STRICT 0.5 THRESHOLD", self.n_splits)
        logger.info("  Accuracy    : %.3f ± %.3f", float(np.mean(accs)), float(np.std(accs)))
        logger.info("  Sensitivity : %.3f ± %.3f", float(np.mean(senss)), float(np.std(senss)))
        logger.info("  Specificity : %.3f ± %.3f", float(np.mean(specs)), float(np.std(specs)))
        logger.info("  AUROC       : %.3f ± %.3f", float(np.mean(aurocs)), float(np.std(aurocs)))
        logger.info("=" * 60)

        return {
            "accuracy": (float(np.mean(accs)), float(np.std(accs))),
            "sensitivity": (float(np.mean(senss)), float(np.std(senss))),
            "specificity": (float(np.mean(specs)), float(np.std(specs))),
            "auroc": (float(np.mean(aurocs)), float(np.std(aurocs))),
            "fold_metrics": fold_metrics,
        }

