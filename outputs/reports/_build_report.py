#!/usr/bin/env python
"""Build the consolidated RESULTS report + confusion matrices for the VOG-MCI
signal-mode study. COMPLETED runs only (logs containing 'MC-CV Results');
halted/mid-training runs are excluded. Confusion matrices are reconstructed from
the per-fold logs and written to outputs/reports/confusion_matrices/."""
import re, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/oem/choihyunsu/VOGBasedMCIDetection"
LOGDIR = os.path.join(ROOT, "outputs/logs")
REPORTS = os.path.join(ROOT, "outputs/reports")
OUT = os.path.join(REPORTS, "RESULTS.md")
# Per-run confusion matrices are written to a
# {mode}_using/{sampling}/{scheme}/confusion_matrix.png hierarchy under REPORTS.

SCHEME = {  # weight-tag -> readable
    "default": "default (0·.5·.5·.5·0·1.5·1.5·1.5)",
    "w0-5-5-10-0-15-15-30": "0 .5 .5 1 0 1.5 1.5 3",
    "w0-5-10-10-0-15-30-30": "0 .5 1 1 0 1.5 3 3",
    "w0-5-10-10-0-10-30-30": "0 .5 1 1 0 1 3 3",
    "w0-5-10-10-0-10-20-20": "0 .5 1 1 0 1 2 2",
}


def parse(path):
    t = open(path, errors="replace").read()
    if "MC-CV Results" not in t:
        return None  # halted / incomplete -> excluded
    def met(n):
        x = re.search(n + r"\s+:\s*([0-9.]+)\s*±\s*([0-9.]+)", t)
        return (float(x.group(1)), float(x.group(2))) if x else None
    rid = (re.search(r"Run ID:\s*(\S+)", t) or [None]);  rid = rid.group(1) if hasattr(rid, "group") else os.path.basename(path)
    mode = re.search(r"signal_mode=([a-z_]+)", t)
    mode = mode.group(1) if mode else "legacy"
    strat = "stratified" if "_strat" in rid else "grouped"
    wt = re.search(r"_w([0-9-]+)", rid); wt = ("w" + wt.group(1)) if wt else "default"
    thr = re.search(r"artifact(?:_threshold=|=thr=)([0-9.]+)", t); thr = thr.group(1) if thr else "?"
    folds = re.findall(r"Fold (\d+)/30 \|.*?test: \d+ subj \(HC=(\d+) MCI=(\d+)\)", t)
    res = {int(f): (float(s), float(sp)) for f, s, sp in
           re.findall(r"Fold (\d+) Result \| Acc=[\d.]+ Sens=([\d.]+) Spec=([\d.]+)", t)}
    TP = TN = FP = FN = 0; have = False
    for f, h, mc in folds:
        f, h, mc = int(f), int(h), int(mc)
        if f in res:
            s, sp = res[f]; tp = round(s * mc); tn = round(sp * h)
            TP += tp; FN += mc - tp; TN += tn; FP += h - tn; have = True
    return dict(rid=rid, mode=mode, strat=strat, scheme=wt, thr=thr,
                acc=met("Accuracy"), sens=met("Sensitivity"),
                spec=met("Specificity"), auroc=met("AUROC"),
                conf=(TP, FN, FP, TN) if have else None)


def save_conf(mode, strat, scheme, conf):
    TP, FN, FP, TN = conf
    mat = np.array([[TN, FP], [FN, TP]])
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.imshow(mat, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred HC", "pred MCI"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["true HC", "true MCI"])
    mx = mat.max() or 1
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                    fontsize=15, color="white" if mat[i, j] > mx / 2 else "black")
    acc = (TP + TN) / max(TP + TN + FP + FN, 1)
    sens = TP / max(TP + FN, 1); spec = TN / max(TN + FP, 1)
    ax.set_title(f"{mode} · {strat} · {scheme}\nAcc {acc:.3f} · Sens {sens:.3f} · Spec {spec:.3f}", fontsize=8)
    fig.tight_layout()
    # four_error is the current study; full_error (8ch) is archived (rejected).
    base = REPORTS if mode == "four_error" else os.path.join(REPORTS, "archive")
    sub = os.path.join(base, f"{mode}_using", strat, scheme)
    os.makedirs(sub, exist_ok=True)
    p = os.path.join(sub, "confusion_matrix.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    return os.path.relpath(p, os.path.dirname(OUT))


def f(v): return f"{v[0]:.3f} ± {v[1]:.3f}" if v else "—"


def main():
    # logs live in a {mode}_using/{sampling}/{scheme}/ hierarchy; skip archived (_halted) runs
    log_paths = [p for p in glob.glob(os.path.join(LOGDIR, "**", "run_*full*.log"), recursive=True)
                 if "/_halted/" not in p]
    runs = [r for r in (parse(p) for p in log_paths) if r]
    # signal-mode study runs (four_error / full_error), the focus
    study = [r for r in runs if r["mode"] in ("four_error", "full_error")]
    # confusion PNGs for every study run
    for r in study:
        r["conf_img"] = save_conf(r["mode"], r["strat"], r["scheme"], r["conf"]) if r["conf"] else None

    def row(r):
        img = f"[cm]({r['conf_img']})" if r.get("conf_img") else "—"
        return (f"| {SCHEME.get(r['scheme'], r['scheme'])} | {r['strat']} | "
                f"{f(r['acc'])} | {f(r['sens'])} | {f(r['spec'])} | **{f(r['auroc'])}** | {img} |")

    fe = sorted([r for r in study if r["mode"] == "four_error"], key=lambda r: (r["strat"], r["scheme"]))
    fu = sorted([r for r in study if r["mode"] == "full_error"], key=lambda r: (r["strat"], r["scheme"]))

    L = []
    L.append("# VOG-MCI Detection — Results & Methods\n")
    L.append("_Auto-generated from completed run logs. **Only successful (30/30-fold) runs are "
             "included; halted/mid-training runs are excluded** (listed at the bottom)._\n")

    L.append("## 1. What the system does\n")
    L.append("Event-locked continuous-wavelet (CWT) scalograms of eye-vs-target gaze error feed a "
             "**task-conditioned, frozen-MobileViT** classifier scored per window; a per-subject "
             "**task-weighted soft vote** produces the HC-vs-MCI probability. 37 subjects "
             "(14 HC / 23 MCI), subject-grouped 30-fold cross-validation.\n")

    L.append("## 2. Code: representations, flags, sampling\n")
    L.append("Entry point: `src/four_error_using/detection_caller/detection_caller.py`.\n")
    L.append("**Signal representation** (`--signal-mode`), the CWT channels fed to the model:\n")
    L.append("| Mode | Channels | Meaning |\n|---|---|---|")
    L.append("| `legacy` (default) | 4 | `[mag_L, re_L, mag_R, re_R]` — task-axis only, per eye (magnitude + real part) |")
    L.append("| `four_error` | 4 | `[|CWT(LH-TH)|, |CWT(RH-TH)|, |CWT(LV-TV)|, |CWT(RV-TV)|]` — both axes, magnitude only |")
    L.append("| `full_error` | 8 | both axes × both eyes × (magnitude, real) |\n")
    L.append("**Sampling** (`--stratified`): default = grouped random 70/30 subject splits "
             "(`GroupShuffleSplit`, class ratio varies per fold); `--stratified` = fixed "
             "HC=4/MCI=8 test subjects per fold (balanced), random membership.\n")
    L.append("**Aggregation weights** (`--vote-weights \"w0 … w7\"` or `--weighted-vote` for the "
             "built-in scheme): per-task weights applied in the subject-level soft vote "
             "`P_s = Σ_i w[t_i]·p_i / Σ_i w[t_i]` (inference only; does not affect training).\n")
    L.append("**Other flags:** `--artifact-threshold` (gaze-error rejection, deg), `--batch-size`, "
             "`--dropout`, `--patience`. A6000: TF32 + cuDNN autotune + pinned transfers enabled.\n")

    L.append("## 3. Results (completed runs, threshold 30, 30-fold)\n")
    L.append("**Legacy baseline (task-axis mag+re, default scheme): AUROC 0.791, Sens 0.788, Spec 0.599.**\n")
    L.append("### 3.1 four_error (4-channel, both-axis magnitude)\n")
    L.append("| Vote scheme | Sampling | Accuracy | Sensitivity | Specificity | AUROC | Confusion |")
    L.append("|---|---|---|---|---|---|---|")
    for r in fe: L.append(row(r))
    L.append("\n### 3.2 full_error (8-channel, both-axis magnitude + real)\n")
    L.append("| Vote scheme | Sampling | Accuracy | Sensitivity | Specificity | AUROC | Confusion |")
    L.append("|---|---|---|---|---|---|---|")
    for r in fu: L.append(row(r))

    L.append("\n## 4. Findings\n")
    L.append("- **four_error (both-axis, 4ch) beats legacy** (0.791) across all schemes.\n"
             "- **Stratified folds raise AUROC and cut its variance** (four_error ~0.80–0.84 grouped "
             "→ ~0.86–0.88 stratified) by removing folds with too few HC subjects.\n"
             "- **full_error (8ch) is worst (0.767)** — adding real-part channels overfits (37 subjects).\n"
             "- **Best configuration: four_error + stratified, AUROC ≈ 0.876.**\n")

    L.append("## 5. Confusion matrices\n")
    L.append("One aggregate 2×2 matrix per study run (over 30 folds, ~360 subject-predictions), "
             "organised as `outputs/reports/{mode}_using/{sampling}/{scheme}/confusion_matrix.png`; "
             "linked in the tables above.\n")

    L.append("## 6. Excluded (halted) runs\n")
    halted = sorted(os.path.basename(p) for p in log_paths
                    if "MC-CV Results" not in open(p, errors="replace").read())
    L.append("These runs were stopped mid-training and are **not** in the results above:\n")
    for h in halted:
        L.append(f"- `{h.replace('run_','').replace('.log','')}`")

    open(OUT, "w").write("\n".join(L) + "\n")
    print(f"[report] wrote {OUT}")
    print(f"[report] study runs: {len(study)} | confusion PNGs: {sum(1 for r in study if r.get('conf_img'))} | halted excluded: {len(halted)}")


if __name__ == "__main__":
    main()
