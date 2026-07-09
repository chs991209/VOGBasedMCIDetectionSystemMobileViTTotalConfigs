"""Meta-classifier trainer — subject-level training loop.

Input batches: bundles of shape [B, 8, C, 32, 32].
Loss: CE on the model's [B, num_classes] output, with inverse-frequency class
weights computed from the training split's subject labels.
Optimizer/schedule: AdamW + 5-epoch linear warmup + ReduceLROnPlateau on
val-AUROC + grad-clip norm=1 (mirrors the full-experiments-using trainer).
Early-stopping: patience on val-AUROC. Best-AUROC checkpoint per fold.
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

logger = logging.getLogger(__name__)


def _auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.5
    order = np.argsort(scores)[::-1]
    y_sorted = y_true[order]
    n_pos, n_neg = int(np.sum(y_sorted == 1)), int(np.sum(y_sorted == 0))
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


def _labels_of(subset: Subset) -> torch.Tensor:
    return subset.dataset.y[torch.as_tensor(subset.indices, dtype=torch.long)]


class MetaTrainer:
    WARMUP_EPOCHS = 5
    GRAD_CLIP = 1.0

    def __init__(
        self,
        model: nn.Module,
        device: str = "auto",
        checkpoint_dir: Optional[Path] = None,
        fold_idx: Optional[int] = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
    ):
        if device in ("auto", "cuda"):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # HDLSS regularization: defaults lowered (lr 1e-3 → 1e-4) and weight
        # decay raised (1e-4 → 1e-2) to suppress hyper-convergence on N=37.
        self.lr = lr
        self.weight_decay = weight_decay
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)

        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.fold_idx = fold_idx

    def _checkpoint_path(self) -> Optional[Path]:
        if self.checkpoint_dir is None:
            return None
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        tag = f"fold_{self.fold_idx:02d}" if self.fold_idx is not None else "single"
        return self.checkpoint_dir / f"{tag}_best.pth"

    def train_model(
        self,
        train_subset: Subset,
        val_subset: Subset,
        max_epochs: int = 500,
        batch_size: int = 8,
        early_stop_patience: int = 40,
    ):
        logger.info(
            "Device: %s | max_epochs=%d | batch=%d | es_patience=%d | lr=%.0e | wd=%.0e",
            self.device, max_epochs, batch_size, early_stop_patience,
            self.lr, self.weight_decay,
        )

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Inverse-frequency class weights at subject level
        train_labels = _labels_of(train_subset).tolist()
        counts = Counter(train_labels)
        total = sum(counts.values())
        alpha = torch.tensor(
            [total / (2 * counts.get(i, 1)) for i in range(2)],
            dtype=torch.float32,
        ).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=alpha)
        logger.info("CE weights — HC: %.3f  MCI: %.3f", alpha[0], alpha[1])

        base_lr = self.optimizer.param_groups[0]["lr"]
        plateau = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.2, patience=10, min_lr=1e-6
        )

        best_auroc = -1.0
        best_epoch = 0
        no_improve = 0
        ckpt_path = self._checkpoint_path()

        for epoch in range(max_epochs):
            # Linear warmup
            if epoch < self.WARMUP_EPOCHS:
                lr = base_lr * (epoch + 1) / self.WARMUP_EPOCHS
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr

            # ── Train ─────────────────────────────────────────────
            self.model.train()
            t_correct = t_total = 0
            for bundle, label, _sid in train_loader:
                bundle = bundle.to(self.device)
                label = label.to(self.device)

                self.optimizer.zero_grad()
                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    logits = self.model(bundle)
                    loss = criterion(logits, label)

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

                _, pred = logits.max(1)
                t_total += label.size(0)
                t_correct += pred.eq(label).sum().item()

            # ── Validate ──────────────────────────────────────────
            self.model.eval()
            v_correct = v_total = 0
            all_probs: list = []
            all_lbls: list = []
            with torch.no_grad():
                for bundle, label, _sid in val_loader:
                    bundle_d = bundle.to(self.device)
                    label_d = label.to(self.device)
                    with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                        logits = self.model(bundle_d)
                    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    all_probs.extend(probs.tolist())
                    all_lbls.extend(label.numpy().tolist())
                    _, pred = logits.max(1)
                    v_total += label_d.size(0)
                    v_correct += pred.eq(label_d).sum().item()

            v_auroc = _auroc(np.array(all_lbls), np.array(all_probs))
            t_acc = 100.0 * t_correct / max(t_total, 1)
            v_acc = 100.0 * v_correct / max(v_total, 1)
            cur_lr = self.optimizer.param_groups[0]["lr"]

            logger.info(
                "Ep [%03d/%d] T-Acc: %5.1f%%  V-Acc: %5.1f%%  V-AUROC: %.4f  LR: %.2e",
                epoch + 1, max_epochs, t_acc, v_acc, v_auroc, cur_lr,
            )

            if epoch >= self.WARMUP_EPOCHS:
                plateau.step(v_auroc)

            # Early stop on val AUROC
            if v_auroc > best_auroc:
                best_auroc = v_auroc
                best_epoch = epoch + 1
                no_improve = 0
                if ckpt_path is not None:
                    torch.save(self.model.state_dict(), ckpt_path)
                    logger.info("  ↳ new best AUROC; saved checkpoint: %s", ckpt_path)
            else:
                no_improve += 1
                if no_improve >= early_stop_patience:
                    logger.info(
                        "Early stop — best AUROC %.4f at epoch %d (no improvement for %d epochs)",
                        best_auroc, best_epoch, early_stop_patience,
                    )
                    break

        logger.info("Done. Best AUROC: %.4f (epoch %d)", best_auroc, best_epoch)
        return self.model
