"""Stratified 30-fold subject-grouped CV for the kinematic-feature experiment.

Per-subject feature vector = task-weighted mean of the per-trial kinematics
(adopted scheme: A=0, H-saccades 0.5, V-saccades 1.5 → V is 3x H). A logistic
model is trained per fold on standardized features; folds use fixed per-class
test counts (HC=4 / MCI=8), mirroring the four_error stratified protocol.
"""
import logging

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Adopted task-weight scheme (matches detection_caller default): A excluded, V 3x H.
DEFAULT_WEIGHTS = {0: 0.0, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.0, 5: 1.5, 6: 1.5, 7: 1.5}
LABEL = {"HC": 0, "MCI": 1}


def subject_vectors(data_store, weights=DEFAULT_WEIGHTS):
    """Task-weighted mean feature vector per subject → (X, y, subject_ids)."""
    X, y, sids = [], [], []
    for group, subs in data_store.items():
        for sid, trials in subs.items():
            feats = np.stack([f for f, _ in trials]).astype(np.float64)   # [T, F]
            w = np.array([weights.get(t, 0.0) for _, t in trials])        # [T]
            if w.sum() <= 0:
                continue
            X.append((feats * w[:, None]).sum(0) / w.sum())
            y.append(LABEL[group]); sids.append(f"{group}:{sid}")
    return np.asarray(X), np.asarray(y), sids


def run_cv(data_store, feature_names, weights=DEFAULT_WEIGHTS,
           n_splits=30, test_hc=4, test_mci=8, seed=42):
    X, y, sids = subject_vectors(data_store, weights)
    hc, mci = np.where(y == 0)[0], np.where(y == 1)[0]
    logger.info("Kinematic CV [stratified-group] | subjects HC=%d MCI=%d | features=%s",
                len(hc), len(mci), feature_names)
    rng = np.random.default_rng(seed)
    accs, sens, specs, aucs, coefs = [], [], [], [], []
    TP = TN = FP = FN = 0
    for fold in range(n_splits):
        te = np.concatenate([rng.choice(hc, test_hc, replace=False),
                             rng.choice(mci, test_mci, replace=False)])
        te_set = set(te.tolist())
        tr = np.array([i for i in range(len(y)) if i not in te_set])
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(class_weight="balanced", max_iter=2000)
        clf.fit(sc.transform(X[tr]), y[tr])
        p = clf.predict_proba(sc.transform(X[te]))[:, 1]
        pred = (p >= 0.5).astype(int)
        yte = y[te]
        tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
        TP += tp; TN += tn; FP += fp; FN += fn
        acc = (tp + tn) / max(tp + tn + fp + fn, 1)
        se = tp / max(tp + fn, 1); sp = tn / max(tn + fp, 1)
        au = roc_auc_score(yte, p)
        accs.append(acc); sens.append(se); specs.append(sp); aucs.append(au)
        coefs.append(clf.coef_[0])
        logger.info("Fold %02d/%d | test: 12 subj (HC=%d MCI=%d) | Acc=%.3f Sens=%.3f Spec=%.3f AUROC=%.3f",
                    fold + 1, n_splits, test_hc, test_mci, acc, se, sp, au)

    def ms(a):
        return float(np.mean(a)), float(np.std(a))
    return dict(acc=ms(accs), sens=ms(sens), spec=ms(specs), auroc=ms(aucs),
                conf=(TP, FN, FP, TN), n_hc=int(len(hc)), n_mci=int(len(mci)),
                coef_mean=dict(zip(feature_names, np.mean(coefs, 0).round(4).tolist())))
