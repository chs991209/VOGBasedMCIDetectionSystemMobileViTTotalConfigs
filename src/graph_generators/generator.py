"""Batch time-series visualizer for VOG CSV recordings.

Mirrors the clinical logic of `visualization.ipynb` but routes output to PNG
files under `imgs/time_series_visualized/<GROUP>/<subject_id>/<task_name>.png`
instead of `plt.show()`, and uses the project's `paths.DATA_DIR` as the
default source.

CLI:
    # render every VOG CSV under data/
    python src/graph_generators/generator.py

    # only HC subjects, only horizontal-saccade tasks, first 20 files
    python src/graph_generators/generator.py \\
        --groups HC --tasks Horizontal --limit 20
"""
import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from paths import DATA_DIR, TIME_SERIES_IMGS_DIR  # noqa: E402

logger = logging.getLogger("time_series_visualization")


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _normalize_group(parent_parent_name: str) -> str:
    """Map a raw dataset folder name like 'HC_csv_24_25' to a clean 'HC'/'MCI' label."""
    name_upper = parent_parent_name.upper()
    if "HC" in name_upper:
        return "HC"
    if "MCI" in name_upper:
        return "MCI"
    return parent_parent_name  # fallback — preserve as-is


def _load_csv_safely(file_path: Path) -> pd.DataFrame:
    """Multi-encoding tolerant loader. Matches the notebook's behaviour but
    raises on failure instead of returning None, so the caller can `try`."""
    encodings_to_try = ['utf-16', 'utf-16le', 'utf-8-sig', 'cp949']
    header_idx, raw_lines, header_columns = -1, [], []

    for enc in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=enc, errors='replace') as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                line_clean = line.replace('\x00', '').lower()
                if 'lh' in line_clean and 'rh' in line_clean and 'target' in line_clean:
                    header_idx = i
                    raw_lines = lines
                    header_columns = [c.replace('\x00', '').strip() for c in line.split(',')]
                    break
            if header_idx != -1:
                break
        except UnicodeError:
            continue

    if header_idx == -1:
        raise ValueError("missing core headers (LH/RH/target)")

    parsed = []
    for line in raw_lines[header_idx + 1:]:
        line_clean = line.replace('\x00', '').strip()
        if not line_clean:
            continue
        row = [v.strip() for v in line_clean.split(',')]
        if len(row) < len(header_columns):
            row += [''] * (len(header_columns) - len(row))
        elif len(row) > len(header_columns):
            row = row[:len(header_columns)]
        parsed.append(row)

    df = pd.DataFrame(parsed, columns=header_columns)
    df = df.apply(pd.to_numeric, errors='coerce').dropna(how='all').reset_index(drop=True)
    return df


def _resolve_direction(task_name: str, df: pd.DataFrame) -> str:
    """Determine 'Horizontal' or 'Vertical' from the task filename first
    (most reliable), falling back to data-driven detection if the name is
    ambiguous."""
    name = task_name.lower()
    if "vertical" in name:
        return "Vertical"
    if "horizontal" in name:
        return "Horizontal"
    # Fallback: prefer V if it carries any signal
    tv = next((c for c in df.columns if 'targetv' in str(c).lower() or 'target_v' in str(c).lower()), None)
    if tv and df[tv].abs().sum() > 0:
        return "Vertical"
    return "Horizontal"


# ────────────────────────────────────────────────────────────────────────────
# Visualizer (single file)
# ────────────────────────────────────────────────────────────────────────────

def visualize_vog_file(file_path: Path, out_root: Path, dpi: int = 110) -> bool:
    """Render one VOG CSV to a 3-tier PNG. Returns True on success."""
    path_obj = Path(file_path)
    if not path_obj.exists():
        logger.warning("File not found: %s", path_obj)
        return False

    # Metadata
    raw_group = path_obj.parent.parent.name if len(path_obj.parents) >= 2 else "Unknown_Group"
    group_name = _normalize_group(raw_group)
    session_id = path_obj.parent.name
    task_name = path_obj.stem.replace("PD VOG -_", "").replace("PD VOG -", "").strip()

    try:
        df = _load_csv_safely(path_obj)
    except Exception as e:
        logger.warning("Parse failed (%s): %s", path_obj.name, e)
        return False

    # Time + target column resolution
    time_col = next((c for c in df.columns if 'time' in str(c).lower() or str(c).lower() == 't'), df.columns[0])
    time_sec = df[time_col]

    direction_str = _resolve_direction(task_name, df)
    target_axis = 'v' if direction_str == "Vertical" else 'h'

    target_col = next(
        (c for c in df.columns
         if f'target{target_axis}' in str(c).lower() or f'target_{target_axis}' in str(c).lower()),
        None,
    )
    if not target_col:
        logger.warning("Target column not found for %s: %s", direction_str, path_obj.name)
        return False

    search_l = f'l{target_axis}'
    search_r = f'r{target_axis}'
    eye_col_l = next((c for c in df.columns if str(c).lower() == search_l), None)
    eye_col_r = next((c for c in df.columns if str(c).lower() == search_r), None)
    if not eye_col_l or not eye_col_r:
        logger.warning("Missing eye columns %s/%s in %s", search_l, search_r, path_obj.name)
        return False

    # Anti-saccade dynamic target inversion
    is_anti = 'anti' in task_name.lower()
    df['Expected_Target'] = -df[target_col] if is_anti else df[target_col]
    df['Error_L'] = df[eye_col_l] - df['Expected_Target']
    df['Error_R'] = df[eye_col_r] - df['Expected_Target']

    # ── Render ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    title_str = f"[{group_name}] Session: {session_id}\nTask: {task_name} ({direction_str} Analysis)"
    fig.suptitle(title_str, fontsize=16, fontweight='bold')

    # 1. Raw waveform
    axes[0].plot(time_sec, df[target_col], label=f'Target Stimulus ({target_col})',
                 color='red', linestyle='--', linewidth=2)
    if is_anti:
        axes[0].plot(time_sec, df['Expected_Target'], label='Expected Eye Position (Anti)',
                     color='magenta', linestyle=':', linewidth=2)
    axes[0].plot(time_sec, df[eye_col_l], label=f'Left Eye ({eye_col_l})',  color='blue',  alpha=0.7)
    axes[0].plot(time_sec, df[eye_col_r], label=f'Right Eye ({eye_col_r})', color='green', alpha=0.7)
    axes[0].set_title('1. Raw Waveform: Target vs Eye Movement', fontsize=14)
    axes[0].set_ylabel('Position (Degrees)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle=':', alpha=0.6)

    # 2. Tracking error
    axes[1].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[1].plot(time_sec, df['Error_L'], label='Error Left',  color='purple', alpha=0.8)
    axes[1].plot(time_sec, df['Error_R'], label='Error Right', color='orange', alpha=0.8)
    axes[1].set_title('2. Derived Feature: Eye Tracking Error (Deviation from Expected)', fontsize=14)
    axes[1].set_ylabel('Error Distance')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, linestyle=':', alpha=0.6)

    # 3. Orthogonal cross-axis variance
    cross_axis = 'h' if direction_str == "Vertical" else 'v'
    noise_l = f'l{cross_axis}'
    noise_r = f'r{cross_axis}'
    actual_noise_l = next((c for c in df.columns if str(c).lower() == noise_l), None)
    actual_noise_r = next((c for c in df.columns if str(c).lower() == noise_r), None)
    if actual_noise_l and actual_noise_r:
        axes[2].plot(time_sec, df[actual_noise_l], label=f'Left Eye ({actual_noise_l})',
                     color='gray',  alpha=0.7)
        axes[2].plot(time_sec, df[actual_noise_r], label=f'Right Eye ({actual_noise_r})',
                     color='brown', alpha=0.7)
        axes[2].set_title('3. Outlier/Noise Monitoring (Orthogonal Cross-Axis Variance)', fontsize=14)
        axes[2].set_ylabel('Position')
        axes[2].legend(loc='upper right')
        axes[2].grid(True, linestyle=':', alpha=0.6)
    else:
        axes[2].set_title(f'3. Cross-axis data not available ({noise_l}/{noise_r})', fontsize=14)
    axes[2].set_xlabel('Time (sec)', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save
    safe_task = task_name.replace('/', '_')
    out_path = out_root / group_name / session_id / f"{safe_task}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


# ────────────────────────────────────────────────────────────────────────────
# Batch + CLI
# ────────────────────────────────────────────────────────────────────────────

def _parse_csv_list(s: str) -> list:
    return [t.strip() for t in s.split(",") if t.strip()]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VOG time-series visualizer (batch).")
    p.add_argument("--data-dir", dest="data_dir", type=Path, default=DATA_DIR,
                   help="Root directory containing <group>_csv_*/<subject>/PD VOG -_*.csv files.")
    p.add_argument("--out", type=Path, default=TIME_SERIES_IMGS_DIR,
                   help="Output root for rendered PNGs (default imgs/time_series_visualized).")
    p.add_argument("--groups", type=_parse_csv_list, default=None,
                   help="Comma-separated groups to include (HC,MCI). Default: all.")
    p.add_argument("--subjects", type=_parse_csv_list, default=None,
                   help="Comma-separated subject_id substrings to include. Default: all.")
    p.add_argument("--tasks", type=_parse_csv_list, default=None,
                   help="Comma-separated task-name substrings to include "
                        "(e.g. 'Horizontal,Vertical' or 'Saccade A,Saccade R'). Default: all.")
    p.add_argument("--limit", type=int, default=None,
                   help="Stop after rendering this many files (default: render every match).")
    p.add_argument("--dpi", type=int, default=110)
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    args = _parse_args()

    if not args.data_dir.exists():
        logger.error("Data directory not found: %s", args.data_dir)
        return 1

    csv_files = list(args.data_dir.rglob("*.csv"))
    csv_files = [f for f in csv_files if "PD VOG" in f.name.upper()]

    if args.groups:
        groups_upper = {g.upper() for g in args.groups}
        csv_files = [f for f in csv_files
                     if any(g in f.parent.parent.name.upper() for g in groups_upper)]
    if args.subjects:
        csv_files = [f for f in csv_files
                     if any(s in f.parent.name for s in args.subjects)]
    if args.tasks:
        csv_files = [f for f in csv_files
                     if any(t.lower() in f.stem.lower() for t in args.tasks)]

    if args.limit:
        csv_files = csv_files[:args.limit]

    logger.info("Discovered %d CSV file(s) under %s (after filters)", len(csv_files), args.data_dir)
    logger.info("Output root: %s", args.out)

    success = 0
    for i, fp in enumerate(csv_files, 1):
        if i % 20 == 0 or i == 1:
            logger.info("[%d/%d] %s", i, len(csv_files), fp.name)
        if visualize_vog_file(fp, args.out, dpi=args.dpi):
            success += 1

    logger.info("Done. Rendered %d / %d files.", success, len(csv_files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
