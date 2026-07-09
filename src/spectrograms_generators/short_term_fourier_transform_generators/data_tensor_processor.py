import logging
import pickle
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import signal
from scipy.ndimage import zoom
from collections import defaultdict

# Bootstrap so `from paths import …` works when this file is imported.
_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

logger = logging.getLogger(__name__)


class HighFreqSpectrogramCore:
    """
    [데이터 파이프라인 코어 모듈]
    시계열 데이터 로드, 정규화, 에러 연산, STFT 기반 스펙트로그램 텐서 추출을 전담합니다.
    시각화 로직이 배제된 순수 데이터 처리 계층(Data Processing Layer)입니다.
    """
    def __init__(self, target_time_bins=200, nperseg=64, noverlap=24,
                 default_fs=120.0, target_fs=120.0, min_freq=0.0, max_freq=60.0,
                 cache_path=None):
        self.target_time_bins = target_time_bins
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.default_fs = default_fs
        self.target_fs = target_fs
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.cache_path = Path(cache_path) if cache_path is not None else None

        self.target_tasks = {
            "Horizontal": ["Horizontal Saccade A", "Horizontal Saccade B", "Horizontal Saccade B (anti)", "Horizontal Saccade R"],
            "Vertical": ["Vertical Saccade A", "Vertical Saccade B", "Vertical Saccade B (anti)", "Vertical Saccade R"]
        }

        # 텐서 스토어: [Group][Task][Eye] -> List of Tensors
        self.data_store = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    def _config_signature(self) -> dict:
        return {
            "target_time_bins": self.target_time_bins,
            "nperseg": self.nperseg,
            "noverlap": self.noverlap,
            "target_fs": self.target_fs,
            "min_freq": self.min_freq,
            "max_freq": self.max_freq,
            "target_tasks": {k: list(v) for k, v in self.target_tasks.items()},
        }

    def _try_load_cache(self) -> bool:
        if self.cache_path is None or not self.cache_path.exists():
            return False
        try:
            with open(self.cache_path, "rb") as f:
                payload = pickle.load(f)
        except Exception as e:
            logger.warning("Cache read failed (%s); reprocessing.", e)
            return False
        if payload.get("config") != self._config_signature():
            logger.info("Cache config mismatch; reprocessing.")
            return False
        ds = payload.get("data_store", {})
        self.data_store = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for grp, by_task in ds.items():
            for task, by_eye in by_task.items():
                for eye, tensors in by_eye.items():
                    self.data_store[grp][task][eye] = list(tensors)
        n = sum(len(v) for g in self.data_store.values()
                       for t in g.values()
                       for v in t.values())
        logger.info("Loaded STFT cache: %s (%d tensors).", self.cache_path, n)
        return True

    def _save_cache(self) -> None:
        if self.cache_path is None:
            return
        plain = {grp: {task: {eye: list(tensors) for eye, tensors in by_eye.items()}
                       for task, by_eye in by_task.items()}
                 for grp, by_task in self.data_store.items()}
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
            with open(tmp, "wb") as f:
                pickle.dump({"config": self._config_signature(), "data_store": plain}, f)
            tmp.replace(self.cache_path)
            logger.info("Cached STFT data_store to: %s", self.cache_path)
        except Exception as e:
            logger.warning("Cache write failed: %s", e)

    def _load_csv_safely(self, file_path: Path) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path, skipinitialspace=True)
            df.columns = [str(c).strip() for c in df.columns]
            if any('lh' in c.lower() for c in df.columns): return df
        except Exception: pass

        encodings = ['utf-16', 'utf-16le', 'utf-8-sig', 'cp949']
        header_idx, raw_lines, header_columns = -1, [], []
        for enc in encodings:
            try:
                with open(file_path, 'r', encoding=enc, errors='replace') as f: lines = f.readlines()
                for i, line in enumerate(lines):
                    line_clean = line.replace('\x00', '').lower()
                    if 'lh' in line_clean and 'rh' in line_clean:
                        header_idx, raw_lines = i, lines
                        header_columns = [col.replace('\x00', '').strip() for col in line.split(',')]
                        break
                if header_idx != -1: break
            except UnicodeError: continue

        if header_idx == -1: raise ValueError("Headers missing")

        parsed_data = []
        for line in raw_lines[header_idx + 1:]:
            line_clean = line.replace('\x00', '').strip()
            if not line_clean: continue
            row_values = [val.strip() for val in line_clean.split(',')]
            row_values += [''] * max(0, len(header_columns) - len(row_values))
            parsed_data.append(row_values[:len(header_columns)])

        df = pd.DataFrame(parsed_data, columns=header_columns)
        return df.apply(pd.to_numeric, errors='coerce').dropna(how='all').reset_index(drop=True)

    def _get_spectrogram(self, error_sig: np.ndarray, current_fs: float):
        if len(error_sig) == 0: return None

        if self.target_fs and current_fs != self.target_fs:
            num_samples = int(len(error_sig) * (self.target_fs / current_fs))
            if num_samples > 0:
                error_sig = signal.resample(error_sig, num_samples)
                current_fs = self.target_fs

        current_nperseg = min(self.nperseg, max(16, len(error_sig) // 4))
        current_noverlap = self.noverlap

        if len(error_sig) < current_nperseg:
            error_sig = np.pad(error_sig, (0, current_nperseg - len(error_sig)), mode='constant')

        f_bins, t_bins, Sxx = signal.spectrogram(
            x=error_sig, fs=current_fs, window='hann',
            nperseg=current_nperseg, noverlap=current_noverlap, detrend='constant'
        )

        current_time_bins = Sxx.shape[1]
        if current_time_bins == 0: return None
        time_zoom_factor = self.target_time_bins / current_time_bins

        valid_freq_idx = (f_bins >= self.min_freq) & (f_bins <= self.max_freq)
        Sxx_cropped = Sxx[valid_freq_idx, :]

        Sxx_final = zoom(Sxx_cropped, (1.0, time_zoom_factor), mode='nearest', order=1)
        return Sxx_final

    def process_directory(self, base_dir: Path):
        if self._try_load_cache():
            return
        csv_files = [f for f in base_dir.rglob('*.csv') if 'PD VOG' in f.name.upper()]
        loaded_cnt = 0

        for filepath in csv_files:
            clean_task = filepath.stem.replace("PD VOG -_", "").replace("PD VOG -", "").strip()
            axis_type = "Horizontal" if "Horizontal" in clean_task else "Vertical" if "Vertical" in clean_task else None
            if not axis_type or clean_task not in self.target_tasks[axis_type]: continue

            group = "HC" if "HC" in str(filepath) else "MCI" if "MCI" in str(filepath) else None
            if not group: continue

            try:
                df = self._load_csv_safely(filepath)
                is_anti = "anti" in clean_task.lower()
                axis_char = 'h' if axis_type == "Horizontal" else 'v'

                target_col = next((c for c in df.columns if f'target{axis_char}' in str(c).lower() or f'target_{axis_char}' in str(c).lower()), None)
                if not target_col: continue
                if is_anti: df[target_col] = df[target_col] * -1

                time_col = next((c for c in df.columns if 'time' in str(c).lower() or str(c).lower() == 't'), df.columns[0])
                time_val = df[time_col].dropna().values
                target_val = df[target_col].dropna().values

                current_fs = 1.0 / np.mean(np.diff(time_val)) if len(time_val) > 1 else self.default_fs

                for eye in ['L', 'R']:
                    actual_col = next((c for c in df.columns if str(c).lower() == f'{eye.lower()}{axis_char}'), None)
                    if not actual_col: continue

                    actual_val = df[actual_col].dropna().values
                    min_len = min(len(target_val), len(actual_val))

                    error_sig = actual_val[:min_len] - target_val[:min_len]
                    tensor = self._get_spectrogram(error_sig, current_fs)

                    if tensor is not None:
                        eye_full = 'Left' if eye == 'L' else 'Right'
                        self.data_store[group][clean_task][eye_full].append(tensor)
                        loaded_cnt += 1
            except Exception: pass

        logger.info("고주파(60Hz) 텐서 스토어 적재 완료 (총 %d쌍).", loaded_cnt)
        self._save_cache()