from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
LOGS_DIR = OUTPUTS_DIR / "logs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
CACHE_DIR = OUTPUTS_DIR / "cache"


def ensure_output_dirs() -> None:
    for d in (CHECKPOINTS_DIR, LOGS_DIR, REPORTS_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
