"""Task-contribution probe with multi-grouping comparison.

Records per-window predictions during MC inference, keyed by
(fold, subject_id, task_id). Renders a Markdown report that answers
*how much each task — and each meaningful **group** of tasks — contributes
to subject-level detection.*

Multiple comparison dimensions are supported simultaneously via
``task_groupings``. Each grouping defines a logical slicing of the task
set (e.g. by-task, by-axis, by-type, by-inhibition). For each grouping,
the report renders per-group:

- **Standalone performance** — subject-level metrics using ONLY that
  group's windows in the soft-vote.
- **Leave-One-Out (LOO) contribution** — subject-level metrics using all
  of a subject's windows EXCEPT that group's. Δ AUROC = AUROC_full −
  AUROC_without_group. Positive Δ ⇒ the group contributes positively.

A cross-grouping summary at the end ranks every group across every
grouping by mean Δ AUROC so the most/least useful slices surface at a
glance.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


_TASK_NAMES_2 = {
    0: "Horizontal Saccade B (anti)",
    1: "Vertical Saccade B (anti)",
}
_TASK_NAMES_8 = {
    0: "Horizontal Saccade A",
    1: "Horizontal Saccade B",
    2: "Horizontal Saccade B (anti)",
    3: "Horizontal Saccade R",
    4: "Vertical Saccade A",
    5: "Vertical Saccade B",
    6: "Vertical Saccade B (anti)",
    7: "Vertical Saccade R",
}


def _default_groupings(num_tasks: int, task_names: Dict[int, str]) -> Dict[str, Dict[str, List[int]]]:
    if num_tasks == 8:
        return {
            "by-task": {f"Task {t} — {task_names[t]}": [t] for t in range(8)},
            "by-axis": {
                "Horizontal axis (tasks 0–3)": [0, 1, 2, 3],
                "Vertical axis (tasks 4–7)": [4, 5, 6, 7],
            },
            "by-type": {
                "A — slow visually-guided (tasks 0, 4)": [0, 4],
                "B — gap paradigm (tasks 1, 5)": [1, 5],
                "B-anti — anti-saccade (tasks 2, 6)": [2, 6],
                "R — repetitive (tasks 3, 7)": [3, 7],
            },
            "by-inhibition": {
                "Reflexive saccades (A, B, R)": [0, 1, 3, 4, 5, 7],
                "Cognitive inhibition (B-anti)": [2, 6],
            },
        }
    # Default: 2-task / arbitrary — just per-task
    return {
        "by-task": {f"Task {t} — {task_names.get(t, f'Task {t}')}": [t] for t in range(num_tasks)},
    }


def _auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    order = np.argsort(y_prob)[::-1]
    y_sorted = y_true[order]
    n_pos, n_neg = int(np.sum(y_sorted == 1)), int(np.sum(y_sorted == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
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


def _confusion_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float, float, float]:
    if len(y_true) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    y_pred = (y_prob > 0.5).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    n = tp + tn + fp + fn
    acc = (tp + tn) / n if n else float("nan")
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return acc, sens, spec, _auroc(y_true, y_prob)


def _fmt(v: float, prec: int = 3) -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "—"
    return f"{v:.{prec}f}"


def _signed(v: float, prec: int = 3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:+.{prec}f}"


def _suggest(mean_delta: float) -> str:
    if np.isnan(mean_delta):
        return "—"
    if mean_delta >= 0.05:
        return "**up-weight (≈ 1.5×)**"
    if mean_delta >= 0.01:
        return "keep (1.0×)"
    if mean_delta > -0.01:
        return "neutral (1.0×)"
    if mean_delta > -0.05:
        return "down-weight (≈ 0.5×)"
    return "**drop / weight 0**"


class TaskWiseProbeGenerator:
    """Subject-level multi-grouping contribution probe.

    Parameters
    ----------
    num_tasks
        Number of distinct task IDs the model produces. Picks sensible
        defaults for task_names + task_groupings.
    output_path
        Where the Markdown report is written when ``generate_markdown_report``
        is called.
    task_names
        Optional explicit ``{task_id: name}`` mapping; otherwise defaults
        are used (the canonical 2-task or 8-task name table).
    task_groupings
        Optional ``{grouping_name: {group_name: [task_id, …]}}`` map.
        If omitted, ``_default_groupings`` provides sane defaults
        (per-task for 2-task; by-task + by-axis + by-type + by-inhibition
        for 8-task).
    report_style
        Kept for backwards-compat with old callers; ignored — the new
        report renders all configured groupings.
    """

    def __init__(
        self,
        num_tasks: int = 2,
        output_path: Optional[Union[str, Path]] = None,
        task_names: Optional[Dict[int, str]] = None,
        task_groupings: Optional[Dict[str, Dict[str, List[int]]]] = None,
        report_style: Optional[str] = None,  # ignored, kept for backwards compat
    ):
        self.num_tasks = num_tasks
        if task_names is not None:
            self.task_names = dict(task_names)
        elif num_tasks == 2:
            self.task_names = dict(_TASK_NAMES_2)
        elif num_tasks == 8:
            self.task_names = dict(_TASK_NAMES_8)
        else:
            self.task_names = {i: f"Task {i}" for i in range(num_tasks)}

        self.task_groupings = (
            task_groupings if task_groupings is not None
            else _default_groupings(num_tasks, self.task_names)
        )

        # fold_data[fold][subject_id][task_id] = list of (label, prob)
        self.fold_data: Dict[int, Dict[str, Dict[int, List[Tuple[int, float]]]]] = (
            defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        )
        self.completed_folds = 0
        self.output_path = Path(output_path) if output_path is not None else None

    # ──────────────────────────────────────────────────────────────────
    # Data ingest

    def add_window(
        self,
        fold_idx: int,
        subject_id: str,
        task_id: int,
        true_label: int,
        prob: float,
    ) -> None:
        self.fold_data[fold_idx][str(subject_id)][int(task_id)].append(
            (int(true_label), float(prob))
        )
        self.completed_folds = max(self.completed_folds, fold_idx + 1)

    # ──────────────────────────────────────────────────────────────────
    # Per-fold subject aggregation

    def _per_subject(self, fold_idx: int):
        """Returns dict[subject_id] -> {
            'label': int,
            'P_full': float,              # mean of ALL window probs
            'task_probs': Dict[int, List[float]]   # per-task probability lists
        }.
        """
        out = {}
        for sid, tasks in self.fold_data[fold_idx].items():
            all_probs: List[float] = []
            label = None
            per_task: Dict[int, List[float]] = {}
            for tid, recs in tasks.items():
                ps = [p for (_, p) in recs]
                ls = [l for (l, _) in recs]
                per_task[tid] = ps
                all_probs.extend(ps)
                if label is None and ls:
                    label = ls[0]
            if not all_probs or label is None:
                continue
            out[sid] = {
                "label": int(label),
                "P_full": float(np.mean(all_probs)),
                "task_probs": per_task,
            }
        return out

    # ──────────────────────────────────────────────────────────────────
    # Per-fold group-level metrics

    def _fold_full_baseline(self, fold_idx: int):
        per = self._per_subject(fold_idx)
        if not per:
            return float("nan"), float("nan"), float("nan"), float("nan"), 0
        labels = np.array([s["label"] for s in per.values()])
        probs = np.array([s["P_full"] for s in per.values()])
        a, se, sp, au = _confusion_metrics(labels, probs)
        return a, se, sp, au, len(per)

    def _fold_group_standalone(self, fold_idx: int, group_task_ids: Iterable[int]):
        per = self._per_subject(fold_idx)
        ids = set(group_task_ids)
        labels, probs = [], []
        for s in per.values():
            relevant: List[float] = []
            for t in ids:
                relevant.extend(s["task_probs"].get(t, []))
            if relevant:
                labels.append(s["label"])
                probs.append(float(np.mean(relevant)))
        if not labels:
            return float("nan"), float("nan"), float("nan"), float("nan"), 0
        a, se, sp, au = _confusion_metrics(np.array(labels), np.array(probs))
        return a, se, sp, au, len(labels)

    def _fold_group_loo(self, fold_idx: int, group_task_ids: Iterable[int]):
        """Returns (acc_no, sens_no, spec_no, auroc_no, n_no, delta_auroc)."""
        per = self._per_subject(fold_idx)
        excluded = set(group_task_ids)
        labels_no, probs_no, labels_full, probs_full = [], [], [], []
        for s in per.values():
            remaining: List[float] = []
            for t, ps in s["task_probs"].items():
                if t not in excluded:
                    remaining.extend(ps)
            if remaining:
                labels_no.append(s["label"])
                probs_no.append(float(np.mean(remaining)))
                labels_full.append(s["label"])
                probs_full.append(s["P_full"])
        if not labels_no:
            return float("nan"), float("nan"), float("nan"), float("nan"), 0, float("nan")
        a, se, sp, au_no = _confusion_metrics(np.array(labels_no), np.array(probs_no))
        au_full_same = _auroc(np.array(labels_full), np.array(probs_full))
        delta = au_full_same - au_no  # positive ⇒ group contributes
        return a, se, sp, au_no, len(labels_no), delta

    # ──────────────────────────────────────────────────────────────────
    # Markdown tables

    def _baseline_table(self) -> List[str]:
        rows = [
            "| Fold | n_subj | Acc | Sens | Spec | AUROC |",
            "| :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        a_, se_, sp_, au_ = [], [], [], []
        for f in range(self.completed_folds):
            a, se, sp, au, n = self._fold_full_baseline(f)
            rows.append(f"| {f + 1:02d} | {n} | {_fmt(a)} | {_fmt(se)} | {_fmt(sp)} | {_fmt(au)} |")
            a_.append(a); se_.append(se); sp_.append(sp); au_.append(au)
        rows.append(
            f"| **mean** | — | **{_fmt(np.nanmean(a_))}** | **{_fmt(np.nanmean(se_))}** "
            f"| **{_fmt(np.nanmean(sp_))}** | **{_fmt(np.nanmean(au_))}** |"
        )
        rows.append(
            f"| **std**  | — | {_fmt(np.nanstd(a_))} | {_fmt(np.nanstd(se_))} "
            f"| {_fmt(np.nanstd(sp_))} | {_fmt(np.nanstd(au_))} |"
        )
        return rows

    def _standalone_table(self, group_tasks: List[int]) -> Tuple[List[str], float]:
        rows = [
            "| Fold | n_subj | Acc | Sens | Spec | AUROC |",
            "| :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        a_, se_, sp_, au_ = [], [], [], []
        for f in range(self.completed_folds):
            a, se, sp, au, n = self._fold_group_standalone(f, group_tasks)
            rows.append(f"| {f + 1:02d} | {n} | {_fmt(a)} | {_fmt(se)} | {_fmt(sp)} | {_fmt(au)} |")
            a_.append(a); se_.append(se); sp_.append(sp); au_.append(au)
        mean_au = float(np.nanmean(au_)) if au_ else float("nan")
        rows.append(
            f"| **mean** | — | **{_fmt(np.nanmean(a_))}** | **{_fmt(np.nanmean(se_))}** "
            f"| **{_fmt(np.nanmean(sp_))}** | **{_fmt(mean_au)}** |"
        )
        rows.append(
            f"| **std**  | — | {_fmt(np.nanstd(a_))} | {_fmt(np.nanstd(se_))} "
            f"| {_fmt(np.nanstd(sp_))} | {_fmt(np.nanstd(au_))} |"
        )
        return rows, mean_au

    def _loo_table(self, group_tasks: List[int]) -> Tuple[List[str], float, float]:
        rows = [
            "| Fold | n_subj | AUROC w/o group | Δ AUROC = full − w/o |",
            "| :---: | :---: | :---: | :---: |",
        ]
        aus_no, deltas = [], []
        for f in range(self.completed_folds):
            _a, _se, _sp, au_no, n, delta = self._fold_group_loo(f, group_tasks)
            rows.append(f"| {f + 1:02d} | {n} | {_fmt(au_no)} | {_signed(delta)} |")
            aus_no.append(au_no); deltas.append(delta)
        mean_no = float(np.nanmean(aus_no)) if aus_no else float("nan")
        mean_delta = float(np.nanmean(deltas)) if deltas else float("nan")
        std_delta = float(np.nanstd(deltas)) if deltas else float("nan")
        rows.append(f"| **mean** | — | **{_fmt(mean_no)}** | **{_signed(mean_delta)}** |")
        rows.append(f"| **std**  | — | {_fmt(np.nanstd(aus_no))} | {_fmt(std_delta)} |")
        return rows, mean_delta, std_delta

    # ──────────────────────────────────────────────────────────────────
    # Report assembly

    def generate_markdown_report(self, filepath: Optional[Union[str, Path]] = None) -> None:
        target = Path(filepath) if filepath is not None else self.output_path
        if target is None:
            target = Path("task_wise_probe_report.md")

        md: List[str] = []
        md.append(f"# Task-Contribution Probe — Folds 01–{self.completed_folds:02d}\n")
        md.append("> **Source:** `probe_generator.py`")
        md.append(
            "> **Protocol:** Subject-level soft-vote (matches the main MC evaluator). "
            "Every group's standalone and LOO Δ AUROC are computed per fold, then "
            "summarised across folds. Positive Δ AUROC ⇒ the group contributes "
            "positively to detection."
        )
        md.append(
            f"> **Scope:** num_tasks={self.num_tasks}, "
            f"groupings={', '.join(self.task_groupings.keys())}, "
            f"completed_folds={self.completed_folds}."
        )
        md.append("")

        # 1. Full ensemble baseline
        md.append("## 1. Full Ensemble Baseline (all tasks soft-voted)\n")
        md.append(
            "Same rule as the main evaluator. Subject-level prediction = mean of "
            "all window probabilities across all of a subject's tasks. This is the "
            "headline metric that each LOO ΔAUROC below is computed against.\n"
        )
        md.extend(self._baseline_table())
        md.append("")

        # 2..N. One section per grouping
        loo_summary: List[Dict] = []  # collects (grouping, group_name, group_tasks, mean_delta, std_delta, standalone_auroc)
        section_idx = 2
        for grouping_name, groups in self.task_groupings.items():
            md.append(f"## {section_idx}. Grouping: `{grouping_name}`\n")
            md.append(
                f"This grouping slices the {self.num_tasks} tasks into "
                f"{len(groups)} group(s) and asks, per group: "
                f"*(a)* how well does this group alone classify subjects? "
                f"*(b)* how much does the full ensemble lose when this group is removed?\n"
            )

            sub_idx = ord('a')
            for group_name, group_tasks in groups.items():
                md.append(f"### {section_idx}.{chr(sub_idx)} {group_name}  (tasks {list(group_tasks)})\n")

                md.append("**Standalone — subject-level soft-vote using only this group's windows:**\n")
                standalone_rows, standalone_au = self._standalone_table(list(group_tasks))
                md.extend(standalone_rows)
                md.append("")

                md.append("**LOO contribution — subject-level soft-vote with this group's windows excluded:**\n")
                loo_rows, mean_delta, std_delta = self._loo_table(list(group_tasks))
                md.extend(loo_rows)
                md.append("")

                loo_summary.append({
                    "grouping": grouping_name,
                    "group": group_name,
                    "tasks": list(group_tasks),
                    "mean_delta": mean_delta,
                    "std_delta": std_delta,
                    "standalone_auroc": standalone_au,
                })
                sub_idx += 1

            section_idx += 1

        # Cross-grouping ranking
        md.append(f"## {section_idx}. Cross-Grouping Summary\n")
        md.append(
            "Every group from every grouping, ranked by mean Δ AUROC across folds. "
            "Use this to compare contributions across slicings at a glance — e.g., is "
            "the Anti-saccade *inhibition* group's Δ larger than the strongest *axis* "
            "or any *individual task*'s Δ?\n"
        )
        md.append("| Rank | Grouping | Group | Tasks | Mean Δ AUROC | Std Δ | Standalone AUROC | Suggested Action |")
        md.append("| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :--- |")
        ranked = sorted(
            loo_summary,
            key=lambda r: (-r["mean_delta"]) if not np.isnan(r["mean_delta"]) else float("inf"),
        )
        for rank, r in enumerate(ranked, start=1):
            md.append(
                f"| {rank} | `{r['grouping']}` | {r['group']} | "
                f"{r['tasks']} | {_signed(r['mean_delta'])} | {_fmt(r['std_delta'])} | "
                f"{_fmt(r['standalone_auroc'])} | {_suggest(r['mean_delta'])} |"
            )
        md.append("")

        md.append("---")
        md.append(
            "_Notes:_ Δ AUROC is computed per fold on the SAME subject set "
            "(only subjects with non-empty `no_group` set are counted in that fold's Δ). "
            "Means/stds are NaN-safe across folds — folds with a single-class test set "
            "have NaN AUROC and are excluded from the average. Standalone AUROC is the "
            "mean across folds of subject-level AUROC computed using only the group's "
            "windows. The Suggested Action heuristic is `mean_delta ≥ 0.05 ⇒ up-weight`, "
            "`0.01..0.05 ⇒ keep`, `±0.01 ⇒ neutral`, `−0.05..−0.01 ⇒ down-weight`, "
            "`< −0.05 ⇒ drop`."
        )

        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write("\n".join(md))
        logger.info("Task-Contribution Probe report written: %s", target)
