"""Stage 2 trainer — bundle-level training of the SIFT-DBT classifier.

Consumes SubjectBundleDataset via `ragged_collate`. Model returns a 3-tuple
`(final_logits, logits_task, W_task)`; loss is CE on final_logits only (the
gate + shared classifier + attention_prior are all trained by that single
subject-level supervision — logits_task and W_task are byproducts for XAI).

Best-AUROC checkpoint IS restored back into the model before returning, so
downstream inference scores the peak-generalization weights, not the
final-epoch drift-state.
"""
import logging
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from candidates.f_swin_sift_dbt.data_processor.datasets import ragged_collate

logger = logging.getLogger(__name__)


def _auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Manual AUROC (Mann-Whitney U based). Returns 0.5 on single-class input."""
    if len(y_true) == 0:
        return 0.5
    order = np.argsort(scores)[::-1]
    y_sorted = y_true[order]
    n_pos = int(np.sum(y_sorted == 1))
    n_neg = int(np.sum(y_sorted == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp = fp = 0
    tprs, fprs = [0.0], [0.0]
    for label in y_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tprs.append(tp / n_pos)
        fprs.append(fp / n_neg)
    return float(np.trapezoid(tprs, fprs))


def _labels_of(subset: Subset) -> list:
    if hasattr(subset, "y"):
        return subset.y.tolist()
    return subset.dataset.y[torch.as_tensor(subset.indices, dtype=torch.long)].tolist()


class Stage2Trainer:
    WARMUP_EPOCHS = 5
    GRAD_CLIP = 1.0

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        checkpoint_path: Optional[Path] = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
    ):
        self.device = device
        self.model = model.to(device)
        self.use_amp = device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
        self.lr = lr
        self.wd = weight_decay
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        n_trainable = sum(p.numel() for p in trainable)
        n_frozen = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        logger.info(
            "Stage 2 trainable params: %.3f M  |  frozen: %.3f M",
            n_trainable / 1e6, n_frozen / 1e6,
        )

    def _class_weights(self, train_subset: Subset) -> torch.Tensor:
        labels = _labels_of(train_subset)
        counts = Counter(labels)
        total = sum(counts.values())
        return torch.tensor(
            [total / (2 * counts.get(i, 1)) for i in range(2)],
            dtype=torch.float32,
        )

    def train(
        self,
        train_subset: Subset,
        val_subset: Subset,
        epochs: int = 500,
        batch_size: int = 8,
        patience: int = 30,
    ) -> nn.Module:
        """Fit; restore best-val-AUROC checkpoint into self.model before returning."""
        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True, drop_last=False,
            collate_fn=ragged_collate,
        )
        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False,
            collate_fn=ragged_collate,
        )

        alpha = self._class_weights(train_subset).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=alpha)
        logger.info("Stage 2 CE weights — HC: %.3f  MCI: %.3f", alpha[0], alpha[1])

        base_lr = self.optimizer.param_groups[0]["lr"]
        plateau = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.2, patience=10, min_lr=1e-6
        )

        best_auroc = -1.0
        best_epoch = 0
        no_improve = 0

        for epoch in range(epochs):
            if epoch < self.WARMUP_EPOCHS:
                lr = base_lr * (epoch + 1) / self.WARMUP_EPOCHS
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr

            # ── Train ────────────────────────────────────────────────
            self.model.train()   # NB: model overrides train() to keep frozen submods in eval()
            t_correct = t_total = 0
            for ragged_bundle, ratios, y, _sids in train_loader:
                y = y.to(self.device)
                self.optimizer.zero_grad()
                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    final_logits, _logits_task, _W_task = self.model(ragged_bundle, ratios)
                    loss = criterion(final_logits, y)
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        max_norm=self.GRAD_CLIP,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        max_norm=self.GRAD_CLIP,
                    )
                    self.optimizer.step()
                _, pred = final_logits.max(1)
                t_correct += pred.eq(y).sum().item()
                t_total += y.size(0)

            # ── Validate (compute AUROC + accuracy) ──────────────────
            self.model.eval()
            v_correct = v_total = 0
            all_probs, all_lbls = [], []
            with torch.no_grad():
                for ragged_bundle, ratios, y, _sids in val_loader:
                    y_d = y.to(self.device)
                    with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                        final_logits, _lt, _wt = self.model(ragged_bundle, ratios)
                    probs = torch.softmax(final_logits, dim=1)[:, 1].cpu().numpy()
                    all_probs.extend(probs.tolist())
                    all_lbls.extend(y.numpy().tolist())
                    _, pred = final_logits.max(1)
                    v_correct += pred.eq(y_d).sum().item()
                    v_total += y_d.size(0)

            v_auroc = _auroc(np.array(all_lbls), np.array(all_probs))
            train_acc = 100.0 * t_correct / max(t_total, 1)
            val_acc = 100.0 * v_correct / max(v_total, 1)
            cur_lr = self.optimizer.param_groups[0]["lr"]

            logger.info(
                "S2-Ep [%03d/%d] T-Acc: %5.1f%%  V-Acc: %5.1f%%  V-AUROC: %.4f  LR: %.2e",
                epoch + 1, epochs, train_acc, val_acc, v_auroc, cur_lr,
            )

            if epoch >= self.WARMUP_EPOCHS:
                plateau.step(v_auroc)

            # Best-AUROC checkpoint tracking
            if v_auroc > best_auroc:
                best_auroc = v_auroc
                best_epoch = epoch + 1
                no_improve = 0
                if self.checkpoint_path is not None:
                    torch.save(self.model.state_dict(), self.checkpoint_path)
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info(
                        "Stage 2 early stop — best AUROC %.4f at epoch %d",
                        best_auroc, best_epoch,
                    )
                    break

        # Restore best checkpoint into the model
        if self.checkpoint_path is not None and self.checkpoint_path.exists():
            self.model.load_state_dict(
                torch.load(self.checkpoint_path, map_location=self.device)
            )
            logger.info(
                "Restored best-AUROC Stage 2 state (epoch %d, AUROC %.4f)",
                best_epoch, best_auroc,
            )

        logger.info("Stage 2 done. Best val AUROC: %.4f (epoch %d)", best_auroc, best_epoch)
        return self.model
