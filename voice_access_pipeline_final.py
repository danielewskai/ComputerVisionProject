from __future__ import annotations

import json
import hashlib
import math
import os
import pickle
import random
import shutil
import tempfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


# =============================================================================
# Defaults
# =============================================================================

DEFAULT_WORK_DIR = Path("voice_access_files")
DEFAULT_SEGMENT_SECONDS = 3.0
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_TRIM_SILENCE = True
DEFAULT_TRIM_TOP_DB = 25.0
DEFAULT_N_MELS = 128
DEFAULT_IMAGE_SIZE = 128
DEFAULT_EMBEDDING_DIM = 64
DEFAULT_PREEMPHASIS = 0.97
DEFAULT_TARGET_RMS_DB = -20.0
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Paths and configs
# =============================================================================



@dataclass
class ProjectPaths:
    work_dir: Path
    data_cache_dir: Path
    data_signature: str
    recording_meta_path: Path
    segment_meta_path: Path
    spectro_meta_path: Path
    duplicate_meta_path: Path
    allow_speakers_path: Path
    speaker_index_path: Path
    segment_dir: Path
    spectro_dir: Path
    model_dir: Path
    results_dir: Path
    config_dir: Path
    app_added_dir: Path

    @classmethod
    def from_work_dir(
        cls,
        work_dir: Path | str,
        data_config: Optional[Any] = None,
    ) -> "ProjectPaths":
        work_dir = Path(work_dir)
        if data_config is None:
            data_signature = "default"
        else:
            data_signature = data_config_signature(resolve_data_prep_config(data_config))
        data_cache_dir = work_dir / "data_cache" / data_signature
        return cls(
            work_dir=work_dir,
            data_cache_dir=data_cache_dir,
            data_signature=data_signature,
            recording_meta_path=data_cache_dir / "metadata_recording_level.csv",
            segment_meta_path=data_cache_dir / "metadata_segment_level.csv",
            spectro_meta_path=data_cache_dir / "metadata_spectrogram_level.csv",
            duplicate_meta_path=data_cache_dir / "metadata_duplicate_audio.csv",
            allow_speakers_path=data_cache_dir / "allow_speakers.json",
            speaker_index_path=data_cache_dir / "speaker_index.json",
            segment_dir=data_cache_dir / "segmented_audio",
            spectro_dir=data_cache_dir / "spectrograms_npy",
            model_dir=work_dir / "models",
            results_dir=work_dir / "results",
            config_dir=work_dir / "configs",
            app_added_dir=work_dir / "app_added_allow_speakers",
        )


@dataclass
class DataPrepConfig:
    source_dir: Path | str
    work_dir: Path | str = DEFAULT_WORK_DIR
    allow_speakers: Optional[List[str]] = None
    allow_speaker_prefix: Optional[str] = None
    n_allow_speakers: int = 5
    seed: int = 123
    train_ratio: float = 0.7
    valid_ratio: float = 0.1
    target_allow_speakers_in_valid: int = 2
    sample_rate: Optional[int] = DEFAULT_SAMPLE_RATE
    segment_seconds: float = DEFAULT_SEGMENT_SECONDS
    keep_remainder: bool = False
    trim_silence: bool = DEFAULT_TRIM_SILENCE
    trim_top_db: float = DEFAULT_TRIM_TOP_DB
    remove_dc_offset: bool = True
    normalize_peak: bool = True
    peak_target: float = 0.98
    normalize_rms: bool = True
    target_rms_db: float = DEFAULT_TARGET_RMS_DB
    preemphasis: float = DEFAULT_PREEMPHASIS
    image_size: int = DEFAULT_IMAGE_SIZE
    n_mels: int = DEFAULT_N_MELS
    spectrogram_mode: str = "pcen"  # pcen, logmel
    spec_norm_min_percentile: float = 1.0
    spec_norm_max_percentile: float = 99.0
    mel_fmin: float = 20.0
    mel_fmax: Optional[float] = None
    deduplicate_audio: bool = True
    duplicate_hash_max_seconds: Optional[float] = None


@dataclass
class ModelConfig:
    model_name: str = "rescnn"  # plaincnn, rescnn, mobilecnn, transfercnn
    base_channels: int = 32
    dropout: float = 0.35
    use_batchnorm: bool = True
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    pretrained: bool = False


@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 15
    lr: float = 3e-4
    optimizer_name: str = "adamw"
    weight_decay: float = 1e-4
    early_stopping_patience: int = 6
    lr_scheduler_patience: int = 2
    lr_scheduler_factor: float = 0.5
    max_grad_norm: float = 5.0
    aggregation_method: str = "top3mean"
    threshold_false_accept_weight: float = 2.0
    threshold_false_reject_weight: float = 1.0
    threshold_penalty_gap: float = 0.5
    augment: bool = True
    device: str = field(default_factory=lambda: DEFAULT_DEVICE)
    num_workers: Optional[int] = None


@dataclass
class PrototypeConfig:
    segment_aggregation_method: str = "mean"
    similarity_threshold: Optional[float] = None


@dataclass
class NoiseStudyConfig:
    noise_dir: Optional[Path | str] = None
    noise_kind: str = "real"  # real, white, pink
    snr_levels: List[float] = field(default_factory=lambda: [20.0, 10.0, 5.0, 0.0])
    include_clean: bool = True
    split: str = "test"
    n_recordings_per_split: Optional[int] = None
    seed: int = 123


@dataclass
class ExperimentConfig:
    experiment_name: str
    data: DataPrepConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    prototype: PrototypeConfig = field(default_factory=PrototypeConfig)


# =============================================================================
# Utilities
# =============================================================================



def get_paths(config_or_work_dir: DataPrepConfig | ExperimentConfig | Path | str) -> ProjectPaths:
    if isinstance(config_or_work_dir, ExperimentConfig):
        return ProjectPaths.from_work_dir(config_or_work_dir.data.work_dir, data_config=config_or_work_dir.data)
    if isinstance(config_or_work_dir, DataPrepConfig):
        return ProjectPaths.from_work_dir(config_or_work_dir.work_dir, data_config=config_or_work_dir)
    return ProjectPaths.from_work_dir(config_or_work_dir)


def _canonical_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {str(k): _canonical_jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical_jsonable(v) for v in value]
    return value


def data_config_cache_payload(config: DataPrepConfig) -> Dict[str, Any]:
    return {
        "source_dir": str(Path(config.source_dir).resolve()),
        "allow_speakers": None if config.allow_speakers is None else sorted(config.allow_speakers),
        "n_allow_speakers": int(config.n_allow_speakers),
        "seed": int(config.seed),
        "train_ratio": float(config.train_ratio),
        "valid_ratio": float(config.valid_ratio),
        "target_allow_speakers_in_valid": int(config.target_allow_speakers_in_valid),
        "sample_rate": config.sample_rate,
        "segment_seconds": float(config.segment_seconds),
        "keep_remainder": bool(config.keep_remainder),
        "trim_silence": bool(config.trim_silence),
        "trim_top_db": float(config.trim_top_db),
        "remove_dc_offset": bool(config.remove_dc_offset),
        "normalize_peak": bool(config.normalize_peak),
        "peak_target": float(config.peak_target),
        "normalize_rms": bool(config.normalize_rms),
        "target_rms_db": float(config.target_rms_db),
        "preemphasis": float(config.preemphasis),
        "image_size": int(config.image_size),
        "n_mels": int(config.n_mels),
        "spectrogram_mode": str(config.spectrogram_mode),
        "spec_norm_min_percentile": float(config.spec_norm_min_percentile),
        "spec_norm_max_percentile": float(config.spec_norm_max_percentile),
        "mel_fmin": float(config.mel_fmin),
        "mel_fmax": None if config.mel_fmax is None else float(config.mel_fmax),
        "deduplicate_audio": bool(config.deduplicate_audio),
        "duplicate_hash_max_seconds": None if config.duplicate_hash_max_seconds is None else float(config.duplicate_hash_max_seconds),
    }


def data_config_signature(config: DataPrepConfig | Dict[str, Any]) -> str:
    cfg = resolve_data_prep_config(config)
    payload = _canonical_jsonable(data_config_cache_payload(cfg))
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def data_artifacts_exist(config: DataPrepConfig | ExperimentConfig) -> bool:
    paths = get_paths(config)
    return (
        paths.recording_meta_path.exists()
        and paths.segment_meta_path.exists()
        and paths.spectro_meta_path.exists()
    )


def set_seed(seed: int = 123) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_dirs(paths: ProjectPaths, clean_intermediate: bool = False) -> None:
    for p in [
        paths.work_dir,
        paths.data_cache_dir,
        paths.model_dir,
        paths.results_dir,
        paths.config_dir,
        paths.segment_dir,
        paths.spectro_dir,
        paths.app_added_dir,
    ]:
        p.mkdir(parents=True, exist_ok=True)

    if clean_intermediate:
        for p in [paths.segment_dir, paths.spectro_dir]:
            if p.exists():
                shutil.rmtree(p)
            p.mkdir(parents=True, exist_ok=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")


def _to_plain_python(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_plain_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_python(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _safe_torch_load(checkpoint_path: Path | str, map_location: Any) -> Dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, TypeError):
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)



def l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    denom = float(np.linalg.norm(x))
    return x / max(denom, eps)



def cosine_similarity_np(a: np.ndarray, b: np.ndarray) -> float:
    a = l2_normalize_np(a)
    b = l2_normalize_np(b)
    return float(np.dot(a, b))



def clone_experiment(
    base_config: ExperimentConfig,
    *,
    experiment_name: str,
    data_updates: Optional[Dict[str, Any]] = None,
    model_updates: Optional[Dict[str, Any]] = None,
    train_updates: Optional[Dict[str, Any]] = None,
    prototype_updates: Optional[Dict[str, Any]] = None,
) -> ExperimentConfig:
    data = replace(base_config.data, **(data_updates or {}))
    model = replace(base_config.model, **(model_updates or {}))
    train = replace(base_config.train, **(train_updates or {}))
    prototype = replace(base_config.prototype, **(prototype_updates or {}))
    return ExperimentConfig(experiment_name=experiment_name, data=data, model=model, train=train, prototype=prototype)


# =============================================================================
# Config persistence
# =============================================================================



def _experiment_checkpoint_path(paths: ProjectPaths, experiment_name: str) -> Path:
    return paths.model_dir / f"{experiment_name}.pt"


def get_experiment_checkpoint_path(
    config_or_work_dir: ExperimentConfig | Path | str,
    experiment_name: Optional[str] = None,
) -> Path:
    if isinstance(config_or_work_dir, ExperimentConfig):
        paths = get_paths(config_or_work_dir)
        experiment_name = experiment_name or config_or_work_dir.experiment_name
    else:
        if experiment_name is None:
            raise ValueError("experiment_name must be provided when passing only work_dir.")
        paths = get_paths(config_or_work_dir)
    return _experiment_checkpoint_path(paths, experiment_name)


def _experiment_config_path(paths: ProjectPaths, experiment_name: str) -> Path:
    return paths.config_dir / f"{experiment_name}.json"



def save_experiment_config(config: ExperimentConfig, paths: Optional[ProjectPaths] = None) -> Path:
    paths = paths or get_paths(config)
    out_path = _experiment_config_path(paths, config.experiment_name)
    payload = asdict(config)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)
    return out_path



def load_experiment_config(work_dir: Path | str, experiment_name: str) -> ExperimentConfig:
    paths = get_paths(work_dir)
    with open(_experiment_config_path(paths, experiment_name), "r", encoding="utf-8") as f:
        payload = json.load(f)
    return ExperimentConfig(
        experiment_name=payload["experiment_name"],
        data=DataPrepConfig(**payload["data"]),
        model=ModelConfig(**payload["model"]),
        train=TrainConfig(**payload["train"]),
        prototype=PrototypeConfig(**payload["prototype"]),
    )


# =============================================================================
# Waveform and spectrogram preprocessing
# =============================================================================


def remove_dc_offset_waveform(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if len(y) == 0:
        return y
    return (y - float(np.mean(y))).astype(np.float32)


def peak_normalize_waveform(y: np.ndarray, target_peak: float = 0.98) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak <= 1e-8:
        return y
    return (y * (target_peak / peak)).astype(np.float32)


def rms_normalize_waveform(y: np.ndarray, target_rms_db: float = DEFAULT_TARGET_RMS_DB) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if len(y) == 0:
        return y
    rms = float(np.sqrt(np.mean(y ** 2)))
    if rms <= 1e-8:
        return y
    target_rms = float(10.0 ** (target_rms_db / 20.0))
    y = y * (target_rms / rms)
    peak = float(np.max(np.abs(y)))
    if peak > 1.0:
        y = y / peak
    return y.astype(np.float32)


def apply_preemphasis_waveform(y: np.ndarray, coeff: float = DEFAULT_PREEMPHASIS) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if len(y) <= 1 or coeff <= 0:
        return y
    out = np.empty_like(y)
    out[0] = y[0]
    out[1:] = y[1:] - coeff * y[:-1]
    return out.astype(np.float32)


def robust_scale_spectrogram(
    arr: np.ndarray,
    min_percentile: float = 1.0,
    max_percentile: float = 99.0,
) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    lo = float(np.percentile(arr, min_percentile))
    hi = float(np.percentile(arr, max_percentile))
    if not np.isfinite(lo):
        lo = float(np.min(arr))
    if not np.isfinite(hi):
        hi = float(np.max(arr))
    if hi <= lo + 1e-8:
        hi = lo + 1e-8
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo)
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def preprocess_waveform(
    y: np.ndarray,
    sr: int,
    *,
    sample_rate: Optional[int] = DEFAULT_SAMPLE_RATE,
    trim_silence: bool = DEFAULT_TRIM_SILENCE,
    trim_top_db: float = DEFAULT_TRIM_TOP_DB,
    remove_dc_offset: bool = True,
    normalize_peak: bool = True,
    peak_target: float = 0.98,
    normalize_rms: bool = True,
    target_rms_db: float = DEFAULT_TARGET_RMS_DB,
    preemphasis: float = DEFAULT_PREEMPHASIS,
) -> Tuple[np.ndarray, int]:
    y = np.asarray(y, dtype=np.float32)
    if sample_rate is not None and sr != sample_rate:
        y = librosa.resample(y, orig_sr=sr, target_sr=sample_rate).astype(np.float32)
        sr = int(sample_rate)
    if trim_silence:
        y = trim_silence_waveform(y, top_db=trim_top_db)
    if remove_dc_offset:
        y = remove_dc_offset_waveform(y)
    if normalize_peak:
        y = peak_normalize_waveform(y, target_peak=peak_target)
    if normalize_rms:
        y = rms_normalize_waveform(y, target_rms_db=target_rms_db)
    if preemphasis > 0:
        y = apply_preemphasis_waveform(y, coeff=preemphasis)
    y = np.clip(y, -1.0, 1.0).astype(np.float32)
    return y, sr


def resolve_data_prep_config(data_config: Any) -> DataPrepConfig:
    if isinstance(data_config, DataPrepConfig):
        return data_config
    if isinstance(data_config, dict):
        return DataPrepConfig(**data_config)
    raise TypeError(f"Unsupported data_config type: {type(data_config)}")



def fingerprint_audio_file(audio_path: Path | str, config: DataPrepConfig) -> Dict[str, Any]:
    audio_path = Path(audio_path)
    y_raw, sr_raw = librosa.load(audio_path, sr=None, mono=True)
    y_raw = y_raw.astype(np.float32)
    y, sr = preprocess_waveform(
        y_raw,
        sr_raw,
        sample_rate=config.sample_rate,
        trim_silence=config.trim_silence,
        trim_top_db=config.trim_top_db,
        remove_dc_offset=config.remove_dc_offset,
        normalize_peak=config.normalize_peak,
        peak_target=config.peak_target,
        normalize_rms=config.normalize_rms,
        target_rms_db=config.target_rms_db,
        preemphasis=config.preemphasis,
    )
    if config.duplicate_hash_max_seconds is not None and config.duplicate_hash_max_seconds > 0:
        max_len = int(round(float(config.duplicate_hash_max_seconds) * sr))
        y = y[:max_len]
    digest = hashlib.sha1()
    digest.update(np.asarray([sr], dtype=np.int32).tobytes())
    digest.update(np.asarray(y, dtype=np.float32).tobytes())
    return {
        "fingerprint": digest.hexdigest(),
        "n_samples": int(len(y)),
        "seconds": float(len(y) / max(sr, 1)),
        "sample_rate": int(sr),
    }



def deduplicate_speaker_to_wavs(
    speaker_to_wavs: Dict[str, List[str]],
    config: DataPrepConfig,
) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    fingerprint_to_primary: Dict[str, Dict[str, Any]] = {}
    filtered_mapping: Dict[str, List[str]] = {speaker: [] for speaker in speaker_to_wavs}
    duplicate_rows: List[Dict[str, Any]] = []

    for speaker in sorted(speaker_to_wavs):
        for audio_path in sorted(speaker_to_wavs[speaker]):
            fp_info = fingerprint_audio_file(audio_path, config)
            fingerprint = fp_info["fingerprint"]
            if fingerprint not in fingerprint_to_primary:
                fingerprint_to_primary[fingerprint] = {
                    "speaker": speaker,
                    "audio_path": str(audio_path),
                    **fp_info,
                }
                filtered_mapping[speaker].append(str(audio_path))
                continue

            primary = fingerprint_to_primary[fingerprint]
            duplicate_rows.append(
                {
                    "speaker": speaker,
                    "audio_path": str(audio_path),
                    "duplicate_of_speaker": primary["speaker"],
                    "duplicate_of_audio_path": primary["audio_path"],
                    "fingerprint": fingerprint,
                    "processed_seconds": fp_info["seconds"],
                    "sample_rate": fp_info["sample_rate"],
                    "n_samples": fp_info["n_samples"],
                }
            )

    filtered_mapping = {
        speaker: wavs for speaker, wavs in filtered_mapping.items() if len(wavs) > 0
    }
    duplicate_df = pd.DataFrame(
        duplicate_rows,
        columns=[
            "speaker",
            "audio_path",
            "duplicate_of_speaker",
            "duplicate_of_audio_path",
            "fingerprint",
            "processed_seconds",
            "sample_rate",
            "n_samples",
        ],
    )
    return filtered_mapping, duplicate_df


# =============================================================================
# Metadata and data preparation
# =============================================================================


def load_speaker_to_wavs(source_dir: Path | str) -> Dict[str, List[str]]:
    source_dir = Path(source_dir)
    mapping: Dict[str, List[str]] = {}
    for speaker_dir in sorted(p for p in source_dir.iterdir() if p.is_dir()):
        wavs = sorted(str(p) for p in speaker_dir.rglob("*.wav"))
        if wavs:
            mapping[speaker_dir.name] = wavs
    return mapping



def _split_files_within_speaker(
    files: Sequence[str],
    rng: random.Random,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.1,
    min_valid: int = 1,
    min_test: int = 1,
) -> Tuple[List[str], List[str], List[str]]:
    files = list(files)
    rng.shuffle(files)

    n = len(files)

    if n == 0:
        return [], [], []

    if n == 1:
        return [files[0]], [], []

    if n == 2:
        if rng.random() < 0.5:
            return [files[0]], [files[1]], []
        else:
            return [files[0]], [], [files[1]]

    # n >= 3:
    # Always assign at least one file to train, validation, and test when possible.
    train_files = [files[0]]
    valid_files = [files[1]]
    test_files = [files[2]]

    remaining = files[3:]

    for f in remaining:
        u = rng.random()
        if u < train_ratio:
            train_files.append(f)
        elif u < train_ratio + valid_ratio:
            valid_files.append(f)
        else:
            test_files.append(f)

    return train_files, valid_files, test_files

def _split_speakers(
    speakers: Sequence[str],
    rng: random.Random,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.1,
) -> Tuple[List[str], List[str], List[str]]:
    speakers = list(speakers)
    rng.shuffle(speakers)
    n = len(speakers)

    if n == 0:
        return [], [], []
    if n == 1:
        return speakers, [], []
    if n == 2:
        return [speakers[0]], [], [speakers[1]]

    n_train = max(1, int(round(n * train_ratio)))
    n_valid = int(round(n * valid_ratio))

    if n_train >= n:
        n_train = n - 1
    if n_train + n_valid >= n:
        n_valid = max(0, n - n_train - 1)
    if n >= 3 and n_valid == 0:
        n_valid = 1
        if n_train + n_valid >= n:
            n_train = n - n_valid - 1

    train_speakers = speakers[:n_train]
    valid_speakers = speakers[n_train:n_train + n_valid]
    test_speakers = speakers[n_train + n_valid:]

    if len(test_speakers) == 0:
        test_speakers = [train_speakers.pop()]
    return train_speakers, valid_speakers, test_speakers



def pick_allow_speakers_for_validation(
    allow_speakers: Sequence[str],
    speaker_to_wavs: Dict[str, List[str]],
    target_allow_speakers_in_valid: int,
    rng: random.Random,
) -> set[str]:
    eligible = [s for s in allow_speakers if len(speaker_to_wavs[s]) >= 3]
    if len(eligible) == 0 or target_allow_speakers_in_valid <= 0:
        return set()
    if len(eligible) <= target_allow_speakers_in_valid:
        return set(eligible)
    return set(rng.sample(eligible, target_allow_speakers_in_valid))



def build_recording_metadata(config: DataPrepConfig, paths: Optional[ProjectPaths] = None) -> pd.DataFrame:
    paths = paths or get_paths(config)
    source_dir = Path(config.source_dir)
    rng = random.Random(config.seed)

    speaker_to_wavs = load_speaker_to_wavs(source_dir)
    if config.deduplicate_audio:
        speaker_to_wavs, duplicate_df = deduplicate_speaker_to_wavs(speaker_to_wavs, config)
    else:
        duplicate_df = pd.DataFrame(
            columns=[
                "speaker",
                "audio_path",
                "duplicate_of_speaker",
                "duplicate_of_audio_path",
                "fingerprint",
                "processed_seconds",
                "sample_rate",
                "n_samples",
            ]
        )
    duplicate_df.to_csv(paths.duplicate_meta_path, index=False)

    all_speakers = sorted(speaker_to_wavs.keys())
    if len(all_speakers) < 2:
        raise ValueError("Need at least 2 speakers in source_dir after duplicate filtering")

    if config.allow_speakers is None:
        if config.allow_speaker_prefix is None:
            allow_candidates = all_speakers
        else:
            allow_candidates = [
                s for s in all_speakers
                if s.startswith(config.allow_speaker_prefix)
            ]

        if config.n_allow_speakers <= 0:
            raise ValueError("n_allow_speakers must be positive when allow_speakers is None")

        if config.n_allow_speakers > len(allow_candidates):
            raise ValueError(
                f"n_allow_speakers={config.n_allow_speakers} is larger than the number "
                f"of available allow candidates={len(allow_candidates)} "
                f"with prefix={config.allow_speaker_prefix!r}"
            )

        allow_speakers = sorted(rng.sample(allow_candidates, config.n_allow_speakers))
    else:
        allow_speakers = sorted(config.allow_speakers)

    missing = [s for s in allow_speakers if s not in speaker_to_wavs]
    if missing:
        raise ValueError(f"Allow speakers not found in source_dir after duplicate filtering: {missing}")

    not_allow_speakers = [s for s in all_speakers if s not in allow_speakers]
    if len(not_allow_speakers) == 0:
        raise ValueError("Need at least one not_allow speaker")

    allow_speakers_with_valid = pick_allow_speakers_for_validation(
        allow_speakers=allow_speakers,
        speaker_to_wavs=speaker_to_wavs,
        target_allow_speakers_in_valid=config.target_allow_speakers_in_valid,
        rng=rng,
    )

    rows: List[Dict[str, Any]] = []

    for speaker in allow_speakers:
        wavs = speaker_to_wavs[speaker]

        train_wavs, valid_wavs, test_wavs = _split_files_within_speaker(
            wavs,
            rng=rng,
            train_ratio=config.train_ratio,
            valid_ratio=config.valid_ratio,
            min_valid=1,
            min_test=1,
        )

        for split, split_wavs in [("train", train_wavs), ("valid", valid_wavs), ("test", test_wavs)]:
            for audio_path in split_wavs:
                rel = Path(audio_path).relative_to(source_dir)
                recording_id = str(rel).replace("\\", "__").replace("/", "__").replace(".wav", "")
                rows.append(
                    {
                        "recording_id": recording_id,
                        "speaker": speaker,
                        "label": 1,
                        "class_name": "allow",
                        "split": split,
                        "audio_path": str(audio_path),
                    }
                )

    train_speakers, valid_speakers, test_speakers = _split_speakers(
        not_allow_speakers,
        rng=rng,
        train_ratio=config.train_ratio,
        valid_ratio=config.valid_ratio,
    )
    speaker_to_split: Dict[str, str] = {}
    for s in train_speakers:
        speaker_to_split[s] = "train"
    for s in valid_speakers:
        speaker_to_split[s] = "valid"
    for s in test_speakers:
        speaker_to_split[s] = "test"

    for speaker in not_allow_speakers:
        split = speaker_to_split[speaker]
        for audio_path in speaker_to_wavs[speaker]:
            rel = Path(audio_path).relative_to(source_dir)
            recording_id = str(rel).replace("\\", "__").replace("/", "__").replace(".wav", "")
            rows.append(
                {
                    "recording_id": recording_id,
                    "speaker": speaker,
                    "label": 0,
                    "class_name": "not_allow",
                    "split": split,
                    "audio_path": str(audio_path),
                }
            )

    meta = pd.DataFrame(rows).sort_values(["split", "label", "speaker", "audio_path"]).reset_index(drop=True)
    meta.to_csv(paths.recording_meta_path, index=False)

    with open(paths.allow_speakers_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "allow_speakers": allow_speakers,
                "allow_speakers_with_valid_target": sorted(allow_speakers_with_valid),
                "n_duplicates_removed": int(len(duplicate_df)),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return meta



def validate_recording_metadata(meta: pd.DataFrame, target_allow_speakers_in_valid: int = 2) -> None:
    assert set(meta["label"].unique()) <= {0, 1}
    assert set(meta["split"].unique()) <= {"train", "valid", "test"}

    dup = meta.duplicated(subset=["audio_path"]).sum()
    assert dup == 0, f"Duplicate audio_path rows found: {dup}"

    split_counts = meta.groupby("audio_path")["split"].nunique()
    assert int((split_counts > 1).sum()) == 0, "Same audio_path appears in multiple splits"

    not_allow = meta[meta["label"] == 0].copy()
    speaker_split_counts = not_allow.groupby("speaker")["split"].nunique()
    assert int((speaker_split_counts > 1).sum()) == 0, "not_allow speaker appears in multiple splits"

    allow_valid_speakers = meta.loc[(meta["label"] == 1) & (meta["split"] == "valid"), "speaker"].nunique()
    possible_allow_valid_speakers = sum(meta.loc[meta["label"] == 1].groupby("speaker")["audio_path"].nunique() >= 3)
    expected_min = min(target_allow_speakers_in_valid, possible_allow_valid_speakers)
    assert allow_valid_speakers >= expected_min, (
        f"Expected at least {expected_min} allow speakers in valid, got {allow_valid_speakers}"
    )


# =============================================================================
# Audio segmentation and spectrograms
# =============================================================================


def _pad_or_trim_segment(y: np.ndarray, target_len: int) -> np.ndarray:
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    elif len(y) > target_len:
        y = y[:target_len]
    return y.astype(np.float32)



def trim_silence_waveform(y: np.ndarray, top_db: float = DEFAULT_TRIM_TOP_DB) -> np.ndarray:
    if len(y) == 0:
        return y.astype(np.float32)
    yt, _ = librosa.effects.trim(y, top_db=top_db)
    if len(yt) == 0:
        return y.astype(np.float32)
    return yt.astype(np.float32)



def split_waveform_array_to_segments(
    y: np.ndarray,
    sr: int,
    segment_seconds: float = DEFAULT_SEGMENT_SECONDS,
    keep_remainder: bool = False,
) -> List[np.ndarray]:
    segment_len = int(round(segment_seconds * sr))
    if len(y) <= segment_len:
        return [_pad_or_trim_segment(y, segment_len)]

    segments: List[np.ndarray] = []
    for start in range(0, len(y), segment_len):
        end = start + segment_len
        seg = y[start:end]
        if len(seg) < segment_len:
            if not keep_remainder:
                break
            seg = _pad_or_trim_segment(seg, segment_len)
        segments.append(seg.astype(np.float32))
    if not segments:
        segments = [_pad_or_trim_segment(y, segment_len)]
    return segments



def segment_recordings(config: DataPrepConfig, paths: Optional[ProjectPaths] = None) -> pd.DataFrame:
    paths = paths or get_paths(config)
    rec_meta = pd.read_csv(paths.recording_meta_path)
    rows: List[Dict[str, Any]] = []

    for row in tqdm(list(rec_meta.itertuples(index=False)), desc="Segmenting recordings"):
        y_raw, sr_raw = librosa.load(row.audio_path, sr=None, mono=True)
        y_raw = y_raw.astype(np.float32)
        original_seconds = len(y_raw) / max(sr_raw, 1)

        y, sr = preprocess_waveform(
            y_raw,
            sr_raw,
            sample_rate=config.sample_rate,
            trim_silence=config.trim_silence,
            trim_top_db=config.trim_top_db,
            remove_dc_offset=config.remove_dc_offset,
            normalize_peak=config.normalize_peak,
            peak_target=config.peak_target,
            normalize_rms=config.normalize_rms,
            target_rms_db=config.target_rms_db,
            preemphasis=config.preemphasis,
        )
        processed_seconds = len(y) / max(sr, 1)

        segments = split_waveform_array_to_segments(
            y,
            sr,
            segment_seconds=config.segment_seconds,
            keep_remainder=config.keep_remainder,
        )

        for seg_idx, seg in enumerate(segments):
            segment_name = f"{row.recording_id}__seg{seg_idx:04d}.wav"
            out_path = paths.segment_dir / row.split / row.class_name / row.speaker / segment_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out_path, seg, sr)

            rows.append(
                {
                    "recording_id": row.recording_id,
                    "speaker": row.speaker,
                    "label": int(row.label),
                    "class_name": row.class_name,
                    "split": row.split,
                    "audio_path": row.audio_path,
                    "segment_index": seg_idx,
                    "segment_path": str(out_path),
                    "sr": sr,
                    "segment_seconds": config.segment_seconds,
                    "original_seconds": original_seconds,
                    "processed_seconds": processed_seconds,
                }
            )

    seg_meta = pd.DataFrame(rows).sort_values(["split", "label", "speaker", "recording_id", "segment_index"]).reset_index(drop=True)
    seg_meta.to_csv(paths.segment_meta_path, index=False)
    return seg_meta



def mel_spectrogram_float32(
    y: np.ndarray,
    sr: int,
    image_size: int = DEFAULT_IMAGE_SIZE,
    n_mels: int = DEFAULT_N_MELS,
    n_fft: int = 1024,
    hop_length: int = 256,
    fmin: float = 20.0,
    fmax: Optional[float] = None,
    spectrogram_mode: str = "pcen",
    spec_norm_min_percentile: float = 1.0,
    spec_norm_max_percentile: float = 99.0,
) -> np.ndarray:
    mode = str(spectrogram_mode).lower()
    if mode == "pcen":
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=1.0,
            fmin=fmin,
            fmax=fmax,
        )
        spec = librosa.pcen(mel + 1e-6, sr=sr, hop_length=hop_length)
        spec = robust_scale_spectrogram(
            spec,
            min_percentile=spec_norm_min_percentile,
            max_percentile=spec_norm_max_percentile,
        )
    elif mode == "logmel":
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
            fmin=fmin,
            fmax=fmax,
        )
        mel_db = librosa.power_to_db(mel + 1e-10, ref=np.max)
        spec = robust_scale_spectrogram(
            mel_db,
            min_percentile=spec_norm_min_percentile,
            max_percentile=spec_norm_max_percentile,
        )
    else:
        raise ValueError(f"Unknown spectrogram_mode: {spectrogram_mode}")

    x = torch.from_numpy(spec.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)
    arr = x.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
    return np.clip(arr, 0.0, 1.0)



def build_spectrogram_arrays(config: DataPrepConfig, paths: Optional[ProjectPaths] = None) -> pd.DataFrame:
    paths = paths or get_paths(config)
    seg_meta = pd.read_csv(paths.segment_meta_path)
    rows: List[Dict[str, Any]] = []

    for row in tqdm(list(seg_meta.itertuples(index=False)), desc="Building spectrograms"):
        y, sr = librosa.load(row.segment_path, sr=None, mono=True)
        arr = mel_spectrogram_float32(
            y,
            sr=sr,
            image_size=config.image_size,
            n_mels=config.n_mels,
            fmin=config.mel_fmin,
            fmax=config.mel_fmax,
            spectrogram_mode=config.spectrogram_mode,
            spec_norm_min_percentile=config.spec_norm_min_percentile,
            spec_norm_max_percentile=config.spec_norm_max_percentile,
        )

        npy_name = Path(row.segment_path).with_suffix(".npy").name
        out_path = paths.spectro_dir / row.split / row.class_name / row.speaker / npy_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, arr.astype(np.float32))

        rows.append(
            {
                "recording_id": row.recording_id,
                "speaker": row.speaker,
                "label": int(row.label),
                "class_name": row.class_name,
                "split": row.split,
                "audio_path": row.audio_path,
                "segment_index": int(row.segment_index),
                "segment_path": row.segment_path,
                "spectrogram_path": str(out_path),
            }
        )

    spec_meta = pd.DataFrame(rows).sort_values(["split", "label", "speaker", "recording_id", "segment_index"]).reset_index(drop=True)
    spec_meta.to_csv(paths.spectro_meta_path, index=False)
    return spec_meta



def prepare_data_artifacts(
    config: DataPrepConfig,
    clean_intermediate: bool = False,
    rebuild_recording_meta: bool = True,
    rebuild_segments: bool = True,
    rebuild_spectrograms: bool = True,
) -> Dict[str, pd.DataFrame]:
    set_seed(config.seed)
    paths = get_paths(config)
    prepare_dirs(paths, clean_intermediate=clean_intermediate)

    if rebuild_recording_meta or not paths.recording_meta_path.exists():
        recording_meta = build_recording_metadata(config=config, paths=paths)
    else:
        recording_meta = pd.read_csv(paths.recording_meta_path)
    validate_recording_metadata(recording_meta, target_allow_speakers_in_valid=config.target_allow_speakers_in_valid)

    if rebuild_segments or not paths.segment_meta_path.exists():
        segment_meta = segment_recordings(config=config, paths=paths)
    else:
        segment_meta = pd.read_csv(paths.segment_meta_path)

    if rebuild_spectrograms or not paths.spectro_meta_path.exists():
        spectro_meta = build_spectrogram_arrays(config=config, paths=paths)
    else:
        spectro_meta = pd.read_csv(paths.spectro_meta_path)

    return {
        "recording_meta": recording_meta,
        "segment_meta": segment_meta,
        "spectrogram_meta": spectro_meta,
    }


# =============================================================================
# Dataset and loaders
# =============================================================================


def apply_specaugment_np(
    arr: np.ndarray,
    max_time_masks: int = 2,
    max_freq_masks: int = 2,
    time_mask_fraction: float = 0.10,
    freq_mask_fraction: float = 0.10,
) -> np.ndarray:
    arr = arr.copy()
    h, w = arr.shape

    n_time_masks = np.random.randint(0, max_time_masks + 1)
    for _ in range(n_time_masks):
        max_width = max(2, int(w * time_mask_fraction))
        width = np.random.randint(1, max_width + 1)
        start = np.random.randint(0, max(1, w - width + 1))
        arr[:, start:start + width] = 0.0

    n_freq_masks = np.random.randint(0, max_freq_masks + 1)
    for _ in range(n_freq_masks):
        max_height = max(2, int(h * freq_mask_fraction))
        height = np.random.randint(1, max_height + 1)
        start = np.random.randint(0, max(1, h - height + 1))
        arr[start:start + height, :] = 0.0

    return arr



def apply_robustness_augment_np(
    arr: np.ndarray,
    use_specaugment: bool = True,
    noise_std_max: float = 0.03,
    gain_db: float = 3.0,
    shift_fraction: float = 0.08,
) -> np.ndarray:
    arr = arr.copy().astype(np.float32)
    h, w = arr.shape

    if np.random.rand() < 0.5:
        max_shift = max(1, int(w * shift_fraction))
        shift = np.random.randint(-max_shift, max_shift + 1)
        arr = np.roll(arr, shift, axis=1)
        if shift > 0:
            arr[:, :shift] = 0.0
        elif shift < 0:
            arr[:, shift:] = 0.0

    if np.random.rand() < 0.5:
        gain = 10.0 ** (np.random.uniform(-gain_db, gain_db) / 20.0)
        arr = np.clip(arr * gain, 0.0, 1.0)

    if np.random.rand() < 0.5:
        noise_std = np.random.uniform(0.0, noise_std_max)
        noise = np.random.normal(0.0, noise_std, size=arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0.0, 1.0)

    if use_specaugment:
        arr = apply_specaugment_np(arr)

    return arr



def load_spectrogram_as_tensor(
    image_path: str | Path,
    image_size: int = DEFAULT_IMAGE_SIZE,
    mean: Optional[float] = None,
    std: Optional[float] = None,
    augment: bool = False,
) -> torch.Tensor:
    image_path = Path(image_path)
    if image_path.suffix.lower() == ".npy":
        arr = np.load(image_path).astype(np.float32)
    else:
        img = Image.open(image_path).convert("L")
        arr = np.asarray(img, dtype=np.float32) / 255.0

    if arr.shape != (image_size, image_size):
        x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)
        arr = x.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

    if augment:
        arr = apply_robustness_augment_np(arr, use_specaugment=True)

    x = torch.from_numpy(arr).unsqueeze(0)
    if mean is not None and std is not None:
        x = (x - float(mean)) / max(float(std), 1e-8)
    return x.float()


class SpectrogramDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_size: int = DEFAULT_IMAGE_SIZE,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        augment: bool = False,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        x = load_spectrogram_as_tensor(
            row["spectrogram_path"],
            image_size=self.image_size,
            mean=self.mean,
            std=self.std,
            augment=self.augment,
        )
        y = torch.tensor(float(row["label"]), dtype=torch.float32)
        return x, y, row["recording_id"], row["speaker"], row["audio_path"]



def compute_train_mean_std(train_df: pd.DataFrame, image_size: int = DEFAULT_IMAGE_SIZE) -> Tuple[float, float]:
    sum_val = 0.0
    sum_sq = 0.0
    n_pix = 0
    for image_path in train_df["spectrogram_path"]:
        arr = np.load(image_path).astype(np.float32)
        if arr.shape != (image_size, image_size):
            x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
            x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)
            arr = x.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
        sum_val += float(arr.sum())
        sum_sq += float((arr ** 2).sum())
        n_pix += arr.size
    mean = sum_val / max(n_pix, 1)
    var = sum_sq / max(n_pix, 1) - mean ** 2
    std = math.sqrt(max(var, 1e-12))
    return mean, std



def make_loaders(
    paths: ProjectPaths,
    image_size: int = DEFAULT_IMAGE_SIZE,
    batch_size: int = 32,
    augment: bool = True,
    num_workers: Optional[int] = None,
    device: str = "cpu",
    normalizer_mean: Optional[float] = None,
    normalizer_std: Optional[float] = None,
):
    meta = pd.read_csv(paths.spectro_meta_path)
    train_df = meta[meta["split"] == "train"].copy()
    valid_df = meta[meta["split"] == "valid"].copy()
    test_df = meta[meta["split"] == "test"].copy()

    if normalizer_mean is None or normalizer_std is None:
        train_mean, train_std = compute_train_mean_std(train_df, image_size=image_size)
    else:
        train_mean, train_std = float(normalizer_mean), float(normalizer_std)

    train_ds = SpectrogramDataset(train_df, image_size=image_size, mean=train_mean, std=train_std, augment=augment)
    train_eval_ds = SpectrogramDataset(train_df, image_size=image_size, mean=train_mean, std=train_std, augment=False)
    valid_ds = SpectrogramDataset(valid_df, image_size=image_size, mean=train_mean, std=train_std, augment=False)
    test_ds = SpectrogramDataset(test_df, image_size=image_size, mean=train_mean, std=train_std, augment=False)

    if num_workers is None:
        num_workers = min(4, os.cpu_count() or 1)

    loader_kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": str(device).startswith("cuda"),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    train_eval_loader = DataLoader(train_eval_ds, shuffle=False, **loader_kwargs)
    valid_loader = DataLoader(valid_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    stats = {
        "train_mean": float(train_mean),
        "train_std": float(train_std),
        "n_train_segments": int(len(train_df)),
        "n_valid_segments": int(len(valid_df)),
        "n_test_segments": int(len(test_df)),
        "n_train_recordings": int(train_df["recording_id"].nunique()),
        "n_valid_recordings": int(valid_df["recording_id"].nunique()),
        "n_test_recordings": int(test_df["recording_id"].nunique()),
    }
    return train_loader, train_eval_loader, valid_loader, test_loader, stats


# =============================================================================
# Models
# =============================================================================


class BaseVoiceModel(nn.Module):
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.forward_features(x)
        logits = self.classifier(emb)
        return logits.squeeze(1)


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, use_batchnorm: bool = True):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=not use_batchnorm)
        self.bn = nn.BatchNorm2d(out_ch) if use_batchnorm else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.0, use_batchnorm: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=not use_batchnorm)
        self.bn1 = nn.BatchNorm2d(out_ch) if use_batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=not use_batchnorm)
        self.bn2 = nn.BatchNorm2d(out_ch) if use_batchnorm else nn.Identity()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=not use_batchnorm),
                nn.BatchNorm2d(out_ch) if use_batchnorm else nn.Identity(),
            )
        else:
            self.skip = nn.Identity()

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.act(out)
        return out


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.0, use_batchnorm: bool = True):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_ch) if use_batchnorm else nn.Identity()
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_ch) if use_batchnorm else nn.Identity()
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.act(self.dw_bn(self.depthwise(x)))
        x = self.dropout(x)
        x = self.act(self.pw_bn(self.pointwise(x)))
        return x


class PlainCNN(BaseVoiceModel):
    def __init__(self, base_channels: int = 32, dropout: float = 0.35, use_batchnorm: bool = True, embedding_dim: int = DEFAULT_EMBEDDING_DIM):
        super().__init__()
        c = base_channels
        self.features = nn.Sequential(
            ConvBNAct(1, c, kernel_size=5, stride=1, use_batchnorm=use_batchnorm),
            ConvBNAct(c, c, stride=1, use_batchnorm=use_batchnorm),
            ConvBNAct(c, 2 * c, stride=2, use_batchnorm=use_batchnorm),
            nn.Dropout2d(dropout * 0.3),
            ConvBNAct(2 * c, 2 * c, stride=1, use_batchnorm=use_batchnorm),
            ConvBNAct(2 * c, 4 * c, stride=2, use_batchnorm=use_batchnorm),
            nn.Dropout2d(dropout * 0.5),
            ConvBNAct(4 * c, 4 * c, stride=1, use_batchnorm=use_batchnorm),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * c, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, 1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_features(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.embedding_head(x)


class ResCNN(BaseVoiceModel):
    def __init__(self, base_channels: int = 32, dropout: float = 0.35, use_batchnorm: bool = True, embedding_dim: int = DEFAULT_EMBEDDING_DIM):
        super().__init__()
        c = base_channels
        self.stem = ConvBNAct(1, c, kernel_size=5, stride=1, use_batchnorm=use_batchnorm)
        self.features = nn.Sequential(
            ResidualBlock(c, c, stride=1, dropout=0.0, use_batchnorm=use_batchnorm),
            ResidualBlock(c, 2 * c, stride=2, dropout=dropout * 0.3, use_batchnorm=use_batchnorm),
            ResidualBlock(2 * c, 2 * c, stride=1, dropout=dropout * 0.3, use_batchnorm=use_batchnorm),
            ResidualBlock(2 * c, 4 * c, stride=2, dropout=dropout * 0.5, use_batchnorm=use_batchnorm),
            ResidualBlock(4 * c, 4 * c, stride=1, dropout=dropout * 0.5, use_batchnorm=use_batchnorm),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * c, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, 1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        return self.embedding_head(x)


class MobileCNN(BaseVoiceModel):
    def __init__(self, base_channels: int = 24, dropout: float = 0.35, use_batchnorm: bool = True, embedding_dim: int = DEFAULT_EMBEDDING_DIM):
        super().__init__()
        c = base_channels
        self.stem = ConvBNAct(1, c, kernel_size=5, stride=1, use_batchnorm=use_batchnorm)
        self.features = nn.Sequential(
            DepthwiseSeparableBlock(c, c, stride=1, dropout=0.0, use_batchnorm=use_batchnorm),
            DepthwiseSeparableBlock(c, 2 * c, stride=2, dropout=dropout * 0.3, use_batchnorm=use_batchnorm),
            DepthwiseSeparableBlock(2 * c, 2 * c, stride=1, dropout=dropout * 0.3, use_batchnorm=use_batchnorm),
            DepthwiseSeparableBlock(2 * c, 4 * c, stride=2, dropout=dropout * 0.5, use_batchnorm=use_batchnorm),
            DepthwiseSeparableBlock(4 * c, 4 * c, stride=1, dropout=dropout * 0.5, use_batchnorm=use_batchnorm),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * c, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, 1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        return self.embedding_head(x)


class TransferCNN(BaseVoiceModel):
    def __init__(self, pretrained: bool = False, dropout: float = 0.35, embedding_dim: int = DEFAULT_EMBEDDING_DIM):
        super().__init__()
        try:
            from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        except Exception as e:  # pragma: no cover
            raise ImportError("torchvision is required for transfercnn") from e

        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        first = backbone.features[0][0]
        backbone.features[0][0] = nn.Conv2d(
            1,
            first.out_channels,
            kernel_size=first.kernel_size,
            stride=first.stride,
            padding=first.padding,
            bias=False,
        )
        self.backbone = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        last_ch = 576
        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(last_ch, embedding_dim),
            nn.Hardswish(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward_features(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        return self.embedding_head(x)



def build_model(model_config: ModelConfig) -> nn.Module:
    model_name = model_config.model_name.lower()
    if model_name == "plaincnn":
        return PlainCNN(
            base_channels=model_config.base_channels,
            dropout=model_config.dropout,
            use_batchnorm=model_config.use_batchnorm,
            embedding_dim=model_config.embedding_dim,
        )
    if model_name == "rescnn":
        return ResCNN(
            base_channels=model_config.base_channels,
            dropout=model_config.dropout,
            use_batchnorm=model_config.use_batchnorm,
            embedding_dim=model_config.embedding_dim,
        )
    if model_name == "mobilecnn":
        return MobileCNN(
            base_channels=model_config.base_channels,
            dropout=model_config.dropout,
            use_batchnorm=model_config.use_batchnorm,
            embedding_dim=model_config.embedding_dim,
        )
    if model_name == "transfercnn":
        return TransferCNN(
            pretrained=model_config.pretrained,
            dropout=model_config.dropout,
            embedding_dim=model_config.embedding_dim,
        )
    raise ValueError(f"Unknown model_name: {model_config.model_name}")


# =============================================================================
# Metrics and evaluation helpers
# =============================================================================


def collect_segment_predictions(model: nn.Module, loader: DataLoader, device: str) -> pd.DataFrame:
    model.eval()
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for x, y, recording_ids, speakers, audio_paths in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            y_np = y.numpy()
            for i in range(len(probs)):
                rows.append(
                    {
                        "recording_id": recording_ids[i],
                        "speaker": speakers[i],
                        "audio_path": audio_paths[i],
                        "y_true": int(y_np[i]),
                        "p_allow_segment": float(probs[i]),
                    }
                )
    return pd.DataFrame(rows)



def aggregate_recording_predictions(segment_df: pd.DataFrame, method: str = "mean", top_k: int = 3) -> pd.DataFrame:
    if len(segment_df) == 0:
        return pd.DataFrame(columns=["recording_id", "speaker", "audio_path", "y_true", "p_allow"])

    rows: List[Dict[str, Any]] = []
    for recording_id, g in segment_df.groupby("recording_id"):
        probs = g["p_allow_segment"].to_numpy(dtype=float)
        if method == "mean":
            p_allow = float(np.mean(probs))
        elif method == "median":
            p_allow = float(np.median(probs))
        elif method == "top3mean":
            k = min(top_k, len(probs))
            p_allow = float(np.mean(np.sort(probs)[-k:]))
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        rows.append(
            {
                "recording_id": recording_id,
                "speaker": g["speaker"].iloc[0],
                "audio_path": g["audio_path"].iloc[0],
                "y_true": int(g["y_true"].iloc[0]),
                "p_allow": p_allow,
                "n_segments": int(len(g)),
            }
        )

    return pd.DataFrame(rows).sort_values(["y_true", "speaker", "recording_id"]).reset_index(drop=True)



def binary_metrics_from_probs(y_true: np.ndarray, score: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score).astype(float)
    y_pred = (score >= threshold).astype(int)

    n_total = len(y_true)
    n_allow = int((y_true == 1).sum())
    n_not_allow = int((y_true == 0).sum())
    false_accept = int(((y_true == 0) & (y_pred == 1)).sum())
    false_reject = int(((y_true == 1) & (y_pred == 0)).sum())

    far = false_accept / max(n_not_allow, 1)
    frr = false_reject / max(n_allow, 1)
    acc = float((y_pred == y_true).mean()) if n_total > 0 else float("nan")

    return {
        "n_total": int(n_total),
        "n_allow": int(n_allow),
        "n_not_allow": int(n_not_allow),
        "false_accept": int(false_accept),
        "false_reject": int(false_reject),
        "far": float(far),
        "frr": float(frr),
        "acc": float(acc),
        "balanced_error": float(0.5 * (far + frr)),
        "gap_far_frr": float(abs(far - frr)),
    }



def find_best_threshold(
    y_true: np.ndarray,
    score: np.ndarray,
    penalty_gap: float = 0.5,
    false_accept_weight: float = 2.0,
    false_reject_weight: float = 1.0,
    min_threshold: float = 0.01,
    max_threshold: float = 0.99,
    n_grid: int = 99,
) -> Dict[str, float]:
    rows: List[Dict[str, float]] = []
    for t in np.linspace(float(min_threshold), float(max_threshold), int(n_grid)):
        m = binary_metrics_from_probs(y_true, score, threshold=float(t))

        score_value = (
            false_accept_weight * m["far"]
            + false_reject_weight * m["frr"]
            + penalty_gap * m["gap_far_frr"]
        )

        rows.append({"threshold": float(t), "score": float(score_value), **m})

    return (
        pd.DataFrame(rows)
        .sort_values(["score", "balanced_error", "gap_far_frr", "threshold"])
        .iloc[0]
        .to_dict()
    )


def evaluate_binary_model(model: nn.Module, loader: DataLoader, device: str, threshold: float, aggregation_method: str = "mean") -> Dict[str, Any]:
    seg_df = collect_segment_predictions(model, loader, device=device)
    rec_df = aggregate_recording_predictions(seg_df, method=aggregation_method)

    segment_metrics = binary_metrics_from_probs(
        y_true=seg_df["y_true"].to_numpy(),
        score=seg_df["p_allow_segment"].to_numpy(),
        threshold=threshold,
    )
    recording_metrics = binary_metrics_from_probs(
        y_true=rec_df["y_true"].to_numpy(),
        score=rec_df["p_allow"].to_numpy(),
        threshold=threshold,
    )
    return {
        "segment_df": seg_df,
        "recording_df": rec_df,
        "segment_metrics": segment_metrics,
        "recording_metrics": recording_metrics,
    }


# =============================================================================
# Training
# =============================================================================


def compute_loader_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str) -> float:
    model.eval()
    loss_sum = 0.0
    n_seen = 0
    with torch.no_grad():
        for x, y, *_ in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += loss.item() * x.size(0)
            n_seen += x.size(0)
    return loss_sum / max(n_seen, 1)



def _build_optimizer(model: nn.Module, train_config: TrainConfig) -> torch.optim.Optimizer:
    name = train_config.optimizer_name.lower()
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay, momentum=0.9)
    raise ValueError(f"Unknown optimizer_name: {train_config.optimizer_name}")


def _model_configs_are_weight_compatible(target_cfg: ModelConfig, source_payload: Dict[str, Any]) -> bool:
    if not source_payload:
        return True
    source_cfg = ModelConfig(**source_payload)
    comparable_attrs = ["model_name", "base_channels", "use_batchnorm", "embedding_dim"]
    return all(getattr(target_cfg, attr) == getattr(source_cfg, attr) for attr in comparable_attrs)


def initialize_model_from_checkpoint(
    model: nn.Module,
    target_model_config: ModelConfig,
    init_checkpoint_path: Path | str,
    device: str,
) -> Dict[str, Any]:
    init_checkpoint_path = Path(init_checkpoint_path)
    if not init_checkpoint_path.exists():
        raise FileNotFoundError(f"Initialization checkpoint does not exist: {init_checkpoint_path}")
    ckpt = _safe_torch_load(init_checkpoint_path, map_location=device)
    source_payload = ckpt.get("model_config", {})
    if not _model_configs_are_weight_compatible(target_model_config, source_payload):
        raise ValueError(
            "Initialization checkpoint is not weight-compatible with the target model config. "
            f"Checkpoint model_config={source_payload}, target={asdict(target_model_config)}"
        )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return {
        "init_checkpoint_path": str(init_checkpoint_path),
        "init_best_valid_loss": ckpt.get("best_valid_loss"),
        "init_best_valid_balanced_accuracy": ckpt.get("best_valid_balanced_accuracy"),
        "init_model_config": source_payload,
    }


def _save_training_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    *,
    image_size: Optional[int],
    train_mean: Optional[float],
    train_std: Optional[float],
    threshold: float,
    aggregation_method: str,
    model_config: Optional[dict],
    data_config: Optional[dict],
    best_valid_loss: float,
    best_epoch: int,
    best_bacc: float,
    best_thr_info: Dict[str, Any],
    threshold_search_config: Optional[Dict[str, Any]] = None,
    initialization_info: Optional[Dict[str, Any]] = None,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "image_size": image_size,
            "train_mean": train_mean,
            "train_std": train_std,
            "threshold": threshold,
            "aggregation_method": aggregation_method,
            "model_config": _to_plain_python(model_config or {}),
            "data_config": _to_plain_python(data_config or {}),
            "best_valid_loss": best_valid_loss,
            "best_epoch": best_epoch,
            "best_valid_balanced_accuracy": best_bacc,
            "best_valid_threshold_info": best_thr_info,
            "threshold_search_config": _to_plain_python(threshold_search_config or {}),
            "initialization_info": _to_plain_python(initialization_info),
        },
        checkpoint_path,
    )





def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: str,
    train_config: TrainConfig,
    checkpoint_path: Optional[Path] = None,
    train_mean: Optional[float] = None,
    train_std: Optional[float] = None,
    image_size: Optional[int] = None,
    model_config: Optional[dict] = None,
    data_config: Optional[dict] = None,
    init_checkpoint_path: Optional[Path | str] = None,
    target_model_config: Optional[ModelConfig] = None,
) -> pd.DataFrame:
    model = model.to(device)
    initialization_info: Optional[Dict[str, Any]] = None
    if init_checkpoint_path is not None:
        if target_model_config is None:
            raise ValueError("target_model_config must be provided when init_checkpoint_path is used.")
        initialization_info = initialize_model_from_checkpoint(
            model=model,
            target_model_config=target_model_config,
            init_checkpoint_path=init_checkpoint_path,
            device=device,
        )

    train_targets: List[float] = []
    for _, y, *_ in train_loader:
        train_targets.extend(y.tolist())
    train_targets = np.asarray(train_targets, dtype=np.float32)

    n_pos = float((train_targets == 1).sum())
    n_neg = float((train_targets == 0).sum())
    pos_weight_value = n_neg / max(n_pos, 1.0)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    threshold_search_config = {
        "false_accept_weight": float(train_config.threshold_false_accept_weight),
        "false_reject_weight": float(train_config.threshold_false_reject_weight),
        "penalty_gap": float(train_config.threshold_penalty_gap),
    }
    optimizer = _build_optimizer(model, train_config)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=train_config.lr_scheduler_factor,
        patience=train_config.lr_scheduler_patience,
    )

    history_rows: List[Dict[str, Any]] = []
    best_valid_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_threshold = 0.5
    best_bacc = 0.0

    if initialization_info is not None:
        best_valid_loss = compute_loader_loss(model, valid_loader, criterion, device=device)
        valid_seg_df = collect_segment_predictions(model, valid_loader, device=device)
        valid_rec_df = aggregate_recording_predictions(valid_seg_df, method=train_config.aggregation_method)
        best_thr_info = find_best_threshold(
            y_true=valid_rec_df["y_true"].to_numpy(),
            score=valid_rec_df["p_allow"].to_numpy(),
            **threshold_search_config,
        )
        best_threshold = float(best_thr_info["threshold"])
        best_bacc = float(1.0 - best_thr_info["balanced_error"])
        if checkpoint_path is not None:
            _save_training_checkpoint(
                model=model,
                checkpoint_path=checkpoint_path,
                image_size=image_size,
                train_mean=train_mean,
                train_std=train_std,
                threshold=best_threshold,
                aggregation_method=train_config.aggregation_method,
                model_config=model_config,
                data_config=data_config,
                best_valid_loss=best_valid_loss,
                best_epoch=best_epoch,
                best_bacc=best_bacc,
                best_thr_info=best_thr_info,
                threshold_search_config=threshold_search_config,
                initialization_info=initialization_info,
            )
        history_rows.append(
            {
                "epoch": 0,
                "train_loss": float("nan"),
                "valid_loss": float(best_valid_loss),
                "pos_weight": float(pos_weight_value),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "best_threshold": float(best_threshold),
                "best_valid_balanced_accuracy": float(best_bacc),
                "threshold_false_accept_weight": float(train_config.threshold_false_accept_weight),
                "threshold_false_reject_weight": float(train_config.threshold_false_reject_weight),
                "threshold_penalty_gap": float(train_config.threshold_penalty_gap),
                "initialized_from_checkpoint": str(init_checkpoint_path),
            }
        )

    epoch_bar = tqdm(range(1, train_config.epochs + 1), desc="Training epochs", leave=True)

    for epoch in epoch_bar:
        model.train()
        loss_sum = 0.0
        n_seen = 0

        for x, y, *_ in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            if train_config.max_grad_norm is not None and train_config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)

            optimizer.step()
            loss_sum += loss.item() * x.size(0)
            n_seen += x.size(0)

        train_loss = loss_sum / max(n_seen, 1)
        valid_loss = compute_loader_loss(model, valid_loader, criterion, device=device)
        scheduler.step(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            valid_seg_df = collect_segment_predictions(model, valid_loader, device=device)
            valid_rec_df = aggregate_recording_predictions(valid_seg_df, method=train_config.aggregation_method)
            best_thr_info = find_best_threshold(
                y_true=valid_rec_df["y_true"].to_numpy(),
                score=valid_rec_df["p_allow"].to_numpy(),
                **threshold_search_config,
            )
            best_threshold = float(best_thr_info["threshold"])
            best_bacc = float(1.0 - best_thr_info["balanced_error"])

            if checkpoint_path is not None:
                _save_training_checkpoint(
                    model=model,
                    checkpoint_path=checkpoint_path,
                    image_size=image_size,
                    train_mean=train_mean,
                    train_std=train_std,
                    threshold=best_threshold,
                    aggregation_method=train_config.aggregation_method,
                    model_config=model_config,
                    data_config=data_config,
                    best_valid_loss=best_valid_loss,
                    best_epoch=best_epoch,
                    best_bacc=best_bacc,
                    best_thr_info=best_thr_info,
                    threshold_search_config=threshold_search_config,
                    initialization_info=initialization_info,
                )
        else:
            epochs_without_improvement += 1

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "valid_loss": float(valid_loss),
                "pos_weight": float(pos_weight_value),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "best_threshold": float(best_threshold),
                "best_valid_balanced_accuracy": float(best_bacc),
                "threshold_false_accept_weight": float(train_config.threshold_false_accept_weight),
                "threshold_false_reject_weight": float(train_config.threshold_false_reject_weight),
                "threshold_penalty_gap": float(train_config.threshold_penalty_gap),
                "initialized_from_checkpoint": None if init_checkpoint_path is None else str(init_checkpoint_path),
            }
        )

        epoch_bar.set_postfix(train_loss=f"{train_loss:.4f}", valid_loss=f"{valid_loss:.4f}", best=f"{best_valid_loss:.4f}")

        if epochs_without_improvement >= train_config.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    return pd.DataFrame(history_rows)




def train_experiment(
    config: ExperimentConfig,
    rebuild_data: bool = False,
    clean_intermediate: bool = False,
    init_checkpoint_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    set_seed(config.data.seed)
    paths = get_paths(config)

    prepare_data_artifacts(
        config=config.data,
        clean_intermediate=clean_intermediate,
        rebuild_recording_meta=rebuild_data,
        rebuild_segments=rebuild_data,
        rebuild_spectrograms=rebuild_data,
    )

    train_loader, train_eval_loader, valid_loader, _, stats = make_loaders(
        paths=paths,
        image_size=config.data.image_size,
        batch_size=config.train.batch_size,
        augment=config.train.augment,
        num_workers=config.train.num_workers,
        device=config.train.device,
    )

    model = build_model(config.model)
    checkpoint_path = _experiment_checkpoint_path(paths, config.experiment_name)
    history = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=config.train.device,
        train_config=config.train,
        checkpoint_path=checkpoint_path,
        train_mean=stats["train_mean"],
        train_std=stats["train_std"],
        image_size=config.data.image_size,
        model_config=asdict(config.model),
        data_config=asdict(config.data),
        init_checkpoint_path=init_checkpoint_path,
        target_model_config=config.model,
    )

    history.to_csv(paths.results_dir / f"{config.experiment_name}_history.csv", index=False)
    save_experiment_config(config, paths=paths)
    return history




def load_checkpoint_bundle(
    config: ExperimentConfig,
    checkpoint_path: Optional[Path | str] = None,
) -> Tuple[nn.Module, Dict[str, Any], ProjectPaths]:
    paths = get_paths(config)
    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else _experiment_checkpoint_path(paths, config.experiment_name)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    ckpt = _safe_torch_load(checkpoint_path, map_location=config.train.device)
    model_cfg = ModelConfig(**ckpt.get("model_config", asdict(config.model)))
    model = build_model(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(config.train.device)
    model.eval()
    return model, ckpt, paths




def evaluate_saved_experiment(
    config: ExperimentConfig,
    checkpoint_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    model, ckpt, paths = load_checkpoint_bundle(config, checkpoint_path=checkpoint_path)

    _, train_eval_loader, valid_loader, test_loader, _ = make_loaders(
        paths=paths,
        image_size=int(ckpt.get("image_size", config.data.image_size)),
        batch_size=config.train.batch_size,
        augment=False,
        num_workers=config.train.num_workers,
        device=config.train.device,
        normalizer_mean=ckpt.get("train_mean"),
        normalizer_std=ckpt.get("train_std"),
    )

    aggregation_method = str(ckpt.get("aggregation_method", config.train.aggregation_method))
    threshold = float(ckpt.get("threshold", 0.5))

    train_eval = evaluate_binary_model(model, train_eval_loader, device=config.train.device, threshold=threshold, aggregation_method=aggregation_method)
    valid_eval = evaluate_binary_model(model, valid_loader, device=config.train.device, threshold=threshold, aggregation_method=aggregation_method)
    test_eval = evaluate_binary_model(model, test_loader, device=config.train.device, threshold=threshold, aggregation_method=aggregation_method)

    results = pd.DataFrame(
        [
            {
                "experiment_name": config.experiment_name,
                "mode": "binary",
                "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else str(_experiment_checkpoint_path(paths, config.experiment_name)),
                "threshold": threshold,
                "aggregation_method": aggregation_method,
                "split": "train",
                **{f"segment_{k}": v for k, v in train_eval["segment_metrics"].items()},
                **{f"recording_{k}": v for k, v in train_eval["recording_metrics"].items()},
            },
            {
                "experiment_name": config.experiment_name,
                "mode": "binary",
                "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else str(_experiment_checkpoint_path(paths, config.experiment_name)),
                "threshold": threshold,
                "aggregation_method": aggregation_method,
                "split": "valid",
                **{f"segment_{k}": v for k, v in valid_eval["segment_metrics"].items()},
                **{f"recording_{k}": v for k, v in valid_eval["recording_metrics"].items()},
            },
            {
                "experiment_name": config.experiment_name,
                "mode": "binary",
                "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else str(_experiment_checkpoint_path(paths, config.experiment_name)),
                "threshold": threshold,
                "aggregation_method": aggregation_method,
                "split": "test",
                **{f"segment_{k}": v for k, v in test_eval["segment_metrics"].items()},
                **{f"recording_{k}": v for k, v in test_eval["recording_metrics"].items()},
            },
        ]
    )
    results.to_csv(paths.results_dir / f"{config.experiment_name}_evaluation.csv", index=False)
    return results


# =============================================================================
# Bayesian optimization
# =============================================================================



def optimize_experiment_with_optuna(
    base_config: ExperimentConfig,
    n_trials: int = 20,
    rebuild_data_for_first_trial: bool = False,
    init_checkpoint_path: Optional[Path | str] = None,
    allow_data_tuning: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    try:
        import optuna
    except Exception as e:  # pragma: no cover
        raise ImportError("optuna is required for Bayesian optimization") from e

    paths = get_paths(base_config)
    trial_rows: List[Dict[str, Any]] = []

    def objective(trial: Any) -> float:
        data_updates: Dict[str, Any] = {}
        if allow_data_tuning:
            data_updates.update(
                {
                    "segment_seconds": trial.suggest_categorical("segment_seconds", [2.0, 3.0, 4.0]),
                    "trim_top_db": trial.suggest_float("trim_top_db", 15.0, 35.0),
                }
            )

        model_updates: Dict[str, Any] = {
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        }
        if init_checkpoint_path is None:
            model_updates["base_channels"] = trial.suggest_categorical("base_channels", [16, 24, 32, 48])

        train_updates: Dict[str, Any] = {
            "lr": trial.suggest_float("lr", 1e-4, 2e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 5e-4, log=True),
            "threshold_false_accept_weight": trial.suggest_float("threshold_false_accept_weight", 1.0, 5.0),
            "threshold_false_reject_weight": trial.suggest_float("threshold_false_reject_weight", 0.5, 2.0),
            "threshold_penalty_gap": trial.suggest_float("threshold_penalty_gap", 0.0, 0.75),
        }

        cfg = clone_experiment(
            base_config,
            experiment_name=f"{base_config.experiment_name}_trial_{trial.number:03d}",
            data_updates=data_updates or None,
            model_updates=model_updates or None,
            train_updates=train_updates or None,
        )
        train_experiment(
            cfg,
            rebuild_data=allow_data_tuning and rebuild_data_for_first_trial and trial.number == 0,
            clean_intermediate=False,
            init_checkpoint_path=init_checkpoint_path,
        )
        results = evaluate_saved_experiment(cfg)
        prototype_summary = build_speaker_index_for_experiment(cfg)

        valid_metrics = prototype_summary["valid_metrics"]

        objective_value = float(
            valid_metrics["far"]
            + 0.9 * valid_metrics["frr"]
            + 0.2 * valid_metrics["gap_far_frr"]
        )
        checkpoint_path = _experiment_checkpoint_path(get_paths(cfg), cfg.experiment_name)
        trial_rows.append({
            "trial_number": trial.number,
            "experiment_name": cfg.experiment_name,
            "checkpoint_path": str(checkpoint_path),
            "objective": objective_value,
            "prototype_valid_far": valid_metrics["far"],
            "prototype_valid_frr": valid_metrics["frr"],
            "prototype_valid_balanced_error": valid_metrics["balanced_error"],
            "prototype_valid_gap_far_frr": valid_metrics["gap_far_frr"],
            **trial.params,
        })
        return objective_value

    sampler = optuna.samplers.TPESampler(seed=base_config.data.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    trials_df = pd.DataFrame(trial_rows).sort_values(["objective", "trial_number"]).reset_index(drop=True)
    trials_df.to_csv(paths.results_dir / f"{base_config.experiment_name}_optuna_trials.csv", index=False)

    best_row = trials_df.iloc[0].to_dict()
    best_summary = {
        "trial_number": int(best_row["trial_number"]),
        "experiment_name": str(best_row["experiment_name"]),
        "checkpoint_path": str(best_row["checkpoint_path"]),
        "objective": float(best_row["objective"]),
        "params": study.best_trial.params,
    }
    return trials_df, best_summary


# =============================================================================
# Prototype index and inference
# =============================================================================


def extract_embedding_batch(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if not hasattr(model, "forward_features"):
        raise AttributeError("Model does not expose forward_features().")
    emb = model.forward_features(x)
    return F.normalize(emb, p=2, dim=1)



def collect_segment_embeddings(model: nn.Module, loader: DataLoader, device: str) -> pd.DataFrame:
    model.eval()
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for x, y, recording_ids, speakers, audio_paths in loader:
            x = x.to(device)
            emb = extract_embedding_batch(model, x).detach().cpu().numpy()
            y_np = y.numpy()
            for i in range(len(emb)):
                rows.append(
                    {
                        "recording_id": recording_ids[i],
                        "speaker": speakers[i],
                        "audio_path": audio_paths[i],
                        "y_true": int(y_np[i]),
                        "embedding": emb[i].astype(np.float32),
                    }
                )
    return pd.DataFrame(rows)



def aggregate_recording_embeddings(segment_emb_df: pd.DataFrame, method: str = "mean") -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for recording_id, g in segment_emb_df.groupby("recording_id"):
        embs = np.stack(g["embedding"].to_list(), axis=0)
        if method == "mean":
            rec_emb = embs.mean(axis=0)
        elif method == "median":
            rec_emb = np.median(embs, axis=0)
        else:
            raise ValueError(f"Unknown embedding aggregation method: {method}")
        rec_emb = l2_normalize_np(rec_emb)
        rows.append(
            {
                "recording_id": recording_id,
                "speaker": g["speaker"].iloc[0],
                "audio_path": g["audio_path"].iloc[0],
                "y_true": int(g["y_true"].iloc[0]),
                "embedding": rec_emb,
                "n_segments": int(len(g)),
            }
        )
    return pd.DataFrame(rows).sort_values(["y_true", "speaker", "recording_id"]).reset_index(drop=True)



def build_allow_speaker_prototypes(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    aggregation_method: str = "mean",
) -> Dict[str, np.ndarray]:
    seg_emb_df = collect_segment_embeddings(model, loader, device=device)
    rec_emb_df = aggregate_recording_embeddings(seg_emb_df, method=aggregation_method)
    allow_df = rec_emb_df[rec_emb_df["y_true"] == 1].copy()

    prototypes: Dict[str, np.ndarray] = {}
    for speaker, g in allow_df.groupby("speaker"):
        embs = np.stack(g["embedding"].to_list(), axis=0)
        proto = l2_normalize_np(embs.mean(axis=0))
        prototypes[speaker] = proto.astype(np.float32)
    return prototypes



def save_speaker_index(path: Path | str, prototypes: Dict[str, np.ndarray], similarity_threshold: float) -> None:
    payload = {
        "similarity_threshold": float(similarity_threshold),
        "prototypes": {speaker: vec.tolist() for speaker, vec in prototypes.items()},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)



def load_speaker_index(path: Path | str) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"similarity_threshold": 0.5, "prototypes": {}}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    prototypes = {speaker: l2_normalize_np(np.asarray(vec, dtype=np.float32)) for speaker, vec in payload.get("prototypes", {}).items()}
    return {
        "similarity_threshold": float(payload.get("similarity_threshold", 0.5)),
        "prototypes": prototypes,
    }



def match_embedding_to_speakers(embedding: np.ndarray, prototypes: Dict[str, np.ndarray]) -> Dict[str, Any]:
    if not prototypes:
        return {"best_speaker": None, "best_similarity": float("-inf"), "all_similarities": {}}
    sims = {speaker: cosine_similarity_np(embedding, proto) for speaker, proto in prototypes.items()}
    best_speaker = max(sims, key=sims.get)
    best_similarity = float(sims[best_speaker])
    return {"best_speaker": best_speaker, "best_similarity": best_similarity, "all_similarities": sims}



def compute_similarity_threshold_from_valid(
    model: nn.Module,
    valid_loader: DataLoader,
    device: str,
    prototypes: Dict[str, np.ndarray],
    aggregation_method: str = "mean",
    penalty_gap: float = 0.5,
    false_accept_weight: float = 2.0,
    false_reject_weight: float = 1.0,
) -> float:
    seg_emb_df = collect_segment_embeddings(model, valid_loader, device=device)
    rec_emb_df = aggregate_recording_embeddings(seg_emb_df, method=aggregation_method)

    scores = []
    for row in rec_emb_df.itertuples(index=False):
        match = match_embedding_to_speakers(row.embedding, prototypes)
        scores.append(match["best_similarity"])

    score_arr = np.asarray(scores, dtype=float)
    y_true = rec_emb_df["y_true"].to_numpy(dtype=int)
    best = find_best_threshold(
        y_true=y_true,
        score=score_arr,
        penalty_gap=penalty_gap,
        false_accept_weight=false_accept_weight,
        false_reject_weight=false_reject_weight,
    )
    return float(best["threshold"])



def build_speaker_index_for_experiment(config: ExperimentConfig, use_valid_for_threshold: bool = True) -> Dict[str, Any]:
    paths = get_paths(config)
    checkpoint_path = _experiment_checkpoint_path(paths, config.experiment_name)
    summary = build_speaker_index_from_checkpoint(
        data_config=config.data,
        train_config=config.train,
        prototype_config=config.prototype,
        checkpoint_path=checkpoint_path,
        speaker_index_path=paths.speaker_index_path,
        use_valid_for_threshold=use_valid_for_threshold,
    )
    pd.DataFrame([summary["valid_metrics"]]).to_csv(
        paths.results_dir / f"{config.experiment_name}_prototype_valid_metrics.csv", index=False
    )
    pd.DataFrame([summary["test_metrics"]]).to_csv(
        paths.results_dir / f"{config.experiment_name}_prototype_test_metrics.csv", index=False
    )
    return summary



def evaluate_prototype_with_loader(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    prototypes: Dict[str, np.ndarray],
    similarity_threshold: float,
    aggregation_method: str = "mean",
) -> Dict[str, Any]:
    seg_emb_df = collect_segment_embeddings(model, loader, device=device)
    rec_emb_df = aggregate_recording_embeddings(seg_emb_df, method=aggregation_method)

    scores = []
    rows = []
    for row in rec_emb_df.itertuples(index=False):
        match = match_embedding_to_speakers(row.embedding, prototypes)
        scores.append(match["best_similarity"])
        rows.append(
            {
                "recording_id": row.recording_id,
                "speaker": row.speaker,
                "y_true": row.y_true,
                "score": match["best_similarity"],
                "predicted_label": int(match["best_similarity"] >= similarity_threshold),
                "best_speaker": match["best_speaker"],
            }
        )
    record_df = pd.DataFrame(rows)
    metrics = binary_metrics_from_probs(
        y_true=record_df["y_true"].to_numpy(),
        score=record_df["score"].to_numpy(),
        threshold=similarity_threshold,
    )
    return {"recording_df": record_df, "recording_metrics": metrics}



def resolve_inference_data_config(
    ckpt: Dict[str, Any],
    *,
    segment_seconds: Optional[float] = None,
    trim_silence: Optional[bool] = None,
    trim_top_db: Optional[float] = None,
    n_mels: Optional[int] = None,
) -> DataPrepConfig:
    data_payload = ckpt.get("data_config", {}) or {}
    cfg = resolve_data_prep_config({"source_dir": str(DEFAULT_WORK_DIR), **data_payload})
    if segment_seconds is not None:
        cfg.segment_seconds = float(segment_seconds)
    if trim_silence is not None:
        cfg.trim_silence = bool(trim_silence)
    if trim_top_db is not None:
        cfg.trim_top_db = float(trim_top_db)
    if n_mels is not None:
        cfg.n_mels = int(n_mels)
    return cfg


def load_checkpoint_and_model(checkpoint_path: Path | str, device: str = DEFAULT_DEVICE) -> Tuple[nn.Module, dict]:
    checkpoint_path = Path(checkpoint_path)
    ckpt = _safe_torch_load(checkpoint_path, map_location=device)
    model_cfg = ModelConfig(**ckpt.get("model_config", {}))
    model = build_model(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, ckpt



def split_waveform_to_segments(
    wav_path: Path | str,
    data_config: Optional[DataPrepConfig] = None,
    segment_seconds: float = DEFAULT_SEGMENT_SECONDS,
    trim_silence: bool = DEFAULT_TRIM_SILENCE,
    trim_top_db: float = DEFAULT_TRIM_TOP_DB,
) -> Tuple[List[np.ndarray], int]:
    wav_path = Path(wav_path)
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    if data_config is None:
        data_config = DataPrepConfig(source_dir=wav_path.parent, segment_seconds=segment_seconds, trim_silence=trim_silence, trim_top_db=trim_top_db)
    y, sr = preprocess_waveform(
        y,
        sr,
        sample_rate=data_config.sample_rate,
        trim_silence=data_config.trim_silence,
        trim_top_db=data_config.trim_top_db,
        remove_dc_offset=data_config.remove_dc_offset,
        normalize_peak=data_config.normalize_peak,
        peak_target=data_config.peak_target,
        normalize_rms=data_config.normalize_rms,
        target_rms_db=data_config.target_rms_db,
        preemphasis=data_config.preemphasis,
    )
    segments = split_waveform_array_to_segments(
        y,
        sr,
        segment_seconds=data_config.segment_seconds,
        keep_remainder=False,
    )
    return segments, sr



def _waveform_segment_to_model_tensor(
    seg: np.ndarray,
    sr: int,
    image_size: int,
    train_mean: float,
    train_std: float,
    n_mels: int,
    spectrogram_mode: str = "pcen",
    spec_norm_min_percentile: float = 1.0,
    spec_norm_max_percentile: float = 99.0,
    mel_fmin: float = 20.0,
    mel_fmax: Optional[float] = None,
) -> torch.Tensor:
    arr = mel_spectrogram_float32(
        seg,
        sr=sr,
        image_size=image_size,
        n_mels=n_mels,
        spectrogram_mode=spectrogram_mode,
        spec_norm_min_percentile=spec_norm_min_percentile,
        spec_norm_max_percentile=spec_norm_max_percentile,
        fmin=mel_fmin,
        fmax=mel_fmax,
    )
    x = torch.from_numpy(arr).unsqueeze(0)
    x = (x - train_mean) / max(train_std, 1e-8)
    return x.unsqueeze(0)



def compute_embeddings_for_wav_file(
    wav_path: Path | str,
    checkpoint_path: Path | str,
    device: str = DEFAULT_DEVICE,
    segment_seconds: float = DEFAULT_SEGMENT_SECONDS,
    trim_silence: bool = DEFAULT_TRIM_SILENCE,
    trim_top_db: float = DEFAULT_TRIM_TOP_DB,
    n_mels: int = DEFAULT_N_MELS,
) -> Dict[str, Any]:
    model, ckpt = load_checkpoint_and_model(checkpoint_path, device=device)
    image_size = int(ckpt.get("image_size", DEFAULT_IMAGE_SIZE))
    train_mean = float(ckpt.get("train_mean", 0.0))
    train_std = float(ckpt.get("train_std", 1.0))
    data_cfg = resolve_inference_data_config(
        ckpt,
        segment_seconds=segment_seconds,
        trim_silence=trim_silence,
        trim_top_db=trim_top_db,
        n_mels=n_mels,
    )

    segments, sr = split_waveform_to_segments(wav_path, data_config=data_cfg)
    segment_embeddings: List[np.ndarray] = []
    with torch.no_grad():
        for seg in segments:
            x = _waveform_segment_to_model_tensor(
                seg,
                sr,
                image_size,
                train_mean,
                train_std,
                data_cfg.n_mels,
                spectrogram_mode=data_cfg.spectrogram_mode,
                spec_norm_min_percentile=data_cfg.spec_norm_min_percentile,
                spec_norm_max_percentile=data_cfg.spec_norm_max_percentile,
                mel_fmin=data_cfg.mel_fmin,
                mel_fmax=data_cfg.mel_fmax,
            ).to(device)
            emb = extract_embedding_batch(model, x).squeeze(0).detach().cpu().numpy().astype(np.float32)
            segment_embeddings.append(emb)

    recording_embedding = l2_normalize_np(np.mean(np.stack(segment_embeddings, axis=0), axis=0))
    return {
        "sample_rate": sr,
        "n_segments": len(segment_embeddings),
        "segment_embeddings": segment_embeddings,
        "recording_embedding": recording_embedding,
    }



def predict_wav_file_binary(
    wav_path: Path | str,
    checkpoint_path: Path | str,
    device: str = DEFAULT_DEVICE,
    segment_seconds: float = DEFAULT_SEGMENT_SECONDS,
    trim_silence: bool = DEFAULT_TRIM_SILENCE,
    trim_top_db: float = DEFAULT_TRIM_TOP_DB,
    n_mels: int = DEFAULT_N_MELS,
) -> Dict[str, Any]:
    model, ckpt = load_checkpoint_and_model(checkpoint_path, device=device)
    threshold = float(ckpt.get("threshold", 0.5))
    aggregation_method = str(ckpt.get("aggregation_method", "mean"))
    image_size = int(ckpt.get("image_size", DEFAULT_IMAGE_SIZE))
    train_mean = float(ckpt.get("train_mean", 0.0))
    train_std = float(ckpt.get("train_std", 1.0))
    data_cfg = resolve_inference_data_config(
        ckpt,
        segment_seconds=segment_seconds,
        trim_silence=trim_silence,
        trim_top_db=trim_top_db,
        n_mels=n_mels,
    )

    segments, sr = split_waveform_to_segments(wav_path, data_config=data_cfg)
    probs = []
    with torch.no_grad():
        for seg in segments:
            x = _waveform_segment_to_model_tensor(
                seg,
                sr,
                image_size,
                train_mean,
                train_std,
                data_cfg.n_mels,
                spectrogram_mode=data_cfg.spectrogram_mode,
                spec_norm_min_percentile=data_cfg.spec_norm_min_percentile,
                spec_norm_max_percentile=data_cfg.spec_norm_max_percentile,
                mel_fmin=data_cfg.mel_fmin,
                mel_fmax=data_cfg.mel_fmax,
            ).to(device)
            prob = torch.sigmoid(model(x)).item()
            probs.append(float(prob))

    if aggregation_method == "mean":
        recording_prob = float(np.mean(probs))
    elif aggregation_method == "median":
        recording_prob = float(np.median(probs))
    elif aggregation_method == "top3mean":
        recording_prob = float(np.mean(np.sort(np.asarray(probs))[-min(3, len(probs)):]))
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")

    return {
        "n_segments": len(probs),
        "segment_probs_allow": probs,
        "recording_prob_allow": recording_prob,
        "threshold": threshold,
        "predicted_label": int(recording_prob >= threshold),
        "predicted_class_name": "allow" if recording_prob >= threshold else "not_allow",
    }



def predict_wav_file_with_prototypes(
    wav_path: Path | str,
    checkpoint_path: Path | str,
    speaker_index_path: Path | str,
    device: str = DEFAULT_DEVICE,
    segment_seconds: float = DEFAULT_SEGMENT_SECONDS,
    trim_silence: bool = DEFAULT_TRIM_SILENCE,
    trim_top_db: float = DEFAULT_TRIM_TOP_DB,
    n_mels: int = DEFAULT_N_MELS,
) -> Dict[str, Any]:
    emb_info = compute_embeddings_for_wav_file(
        wav_path=wav_path,
        checkpoint_path=checkpoint_path,
        device=device,
        segment_seconds=segment_seconds,
        trim_silence=trim_silence,
        trim_top_db=trim_top_db,
        n_mels=n_mels,
    )
    index_payload = load_speaker_index(speaker_index_path)
    match = match_embedding_to_speakers(emb_info["recording_embedding"], index_payload["prototypes"])
    threshold = float(index_payload["similarity_threshold"])
    pred = int(match["best_similarity"] >= threshold)
    return {
        "n_segments": emb_info["n_segments"],
        "best_speaker": match["best_speaker"],
        "best_similarity": float(match["best_similarity"]),
        "threshold": threshold,
        "predicted_label": pred,
        "predicted_class_name": "allow" if pred == 1 else "not_allow",
        "all_similarities": match["all_similarities"],
    }



def add_new_allowed_speaker_recordings(
    new_speaker_dir: Path | str,
    source_dir: Path | str,
    speaker_name: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    new_speaker_dir = Path(new_speaker_dir)
    source_dir = Path(source_dir)
    if speaker_name is None:
        speaker_name = new_speaker_dir.name
    target_dir = source_dir / speaker_name
    if target_dir.exists() and any(target_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Target speaker directory already exists: {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    for wav_path in sorted(new_speaker_dir.rglob("*.wav")):
        shutil.copy2(wav_path, target_dir / wav_path.name)
    return target_dir



def extend_allow_speakers(config: DataPrepConfig, new_speaker_name: str) -> DataPrepConfig:
    current_allow = set(config.allow_speakers or [])
    paths = get_paths(config)
    if paths.allow_speakers_path.exists():
        with open(paths.allow_speakers_path, "r", encoding="utf-8") as f:
            current_allow |= set(json.load(f).get("allow_speakers", []))
    current_allow.add(new_speaker_name)
    return replace(config, allow_speakers=sorted(current_allow), n_allow_speakers=len(current_allow))



def build_speaker_index_from_checkpoint(
    data_config: DataPrepConfig,
    train_config: TrainConfig,
    prototype_config: PrototypeConfig,
    checkpoint_path: Path | str,
    speaker_index_path: Optional[Path | str] = None,
    use_valid_for_threshold: bool = True,
) -> Dict[str, Any]:
    paths = get_paths(data_config)
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    model, ckpt = load_checkpoint_and_model(checkpoint_path, device=train_config.device)
    speaker_index_path = Path(speaker_index_path or paths.speaker_index_path)

    _, train_eval_loader, valid_loader, test_loader, _ = make_loaders(
        paths=paths,
        image_size=int(ckpt.get("image_size", data_config.image_size)),
        batch_size=train_config.batch_size,
        augment=False,
        num_workers=train_config.num_workers,
        device=train_config.device,
        normalizer_mean=ckpt.get("train_mean"),
        normalizer_std=ckpt.get("train_std"),
    )

    prototypes = build_allow_speaker_prototypes(
        model=model,
        loader=train_eval_loader,
        device=train_config.device,
        aggregation_method=prototype_config.segment_aggregation_method,
    )

    if prototype_config.similarity_threshold is not None:
        similarity_threshold = float(prototype_config.similarity_threshold)
    elif use_valid_for_threshold:
        similarity_threshold = compute_similarity_threshold_from_valid(
            model=model,
            valid_loader=valid_loader,
            device=train_config.device,
            prototypes=prototypes,
            aggregation_method=prototype_config.segment_aggregation_method,
            penalty_gap=train_config.threshold_penalty_gap,
            false_accept_weight=train_config.threshold_false_accept_weight,
            false_reject_weight=train_config.threshold_false_reject_weight,
        )
    else:
        similarity_threshold = 0.5

    save_speaker_index(speaker_index_path, prototypes, similarity_threshold)

    valid_eval = evaluate_prototype_with_loader(
        model=model,
        loader=valid_loader,
        device=train_config.device,
        prototypes=prototypes,
        similarity_threshold=similarity_threshold,
        aggregation_method=prototype_config.segment_aggregation_method,
    )
    test_eval = evaluate_prototype_with_loader(
        model=model,
        loader=test_loader,
        device=train_config.device,
        prototypes=prototypes,
        similarity_threshold=similarity_threshold,
        aggregation_method=prototype_config.segment_aggregation_method,
    )
    summary = {
        "n_prototypes": len(prototypes),
        "similarity_threshold": similarity_threshold,
        "threshold_search_config": {
            "false_accept_weight": float(train_config.threshold_false_accept_weight),
            "false_reject_weight": float(train_config.threshold_false_reject_weight),
            "penalty_gap": float(train_config.threshold_penalty_gap),
        },
        "valid_metrics": valid_eval["recording_metrics"],
        "test_metrics": test_eval["recording_metrics"],
        "checkpoint_path": str(checkpoint_path),
        "speaker_index_path": str(speaker_index_path),
        "operational_mode": "prototype",
    }
    return summary



def add_new_allowed_speaker_with_prototype(
    base_config: ExperimentConfig,
    checkpoint_path: Path | str,
    new_speaker_dir: Path | str,
    speaker_name: Optional[str] = None,
    overwrite_source: bool = False,
    persist_updated_config: bool = True,
) -> Dict[str, Any]:
    copied_dir = add_new_allowed_speaker_recordings(
        new_speaker_dir=new_speaker_dir,
        source_dir=base_config.data.source_dir,
        speaker_name=speaker_name,
        overwrite=overwrite_source,
    )
    new_name = copied_dir.name

    paths = get_paths(base_config)

    index_payload = load_speaker_index(paths.speaker_index_path)
    prototypes = index_payload["prototypes"]
    similarity_threshold = index_payload["similarity_threshold"]

    wav_paths = sorted(copied_dir.rglob("*.wav"))
    if len(wav_paths) == 0:
        raise ValueError(f"No .wav files found for new speaker: {copied_dir}")

    recording_embeddings = []
    total_segments = 0

    for wav_path in wav_paths:
        emb_info = compute_embeddings_for_wav_file(
            wav_path=wav_path,
            checkpoint_path=checkpoint_path,
            device=base_config.train.device,
            segment_seconds=base_config.data.segment_seconds,
            trim_silence=base_config.data.trim_silence,
            trim_top_db=base_config.data.trim_top_db,
            n_mels=base_config.data.n_mels,
        )
        recording_embeddings.append(emb_info["recording_embedding"])
        total_segments += int(emb_info["n_segments"])

    new_proto = l2_normalize_np(np.mean(np.stack(recording_embeddings, axis=0), axis=0))
    prototypes[new_name] = new_proto.astype(np.float32)

    save_speaker_index(paths.speaker_index_path, prototypes, similarity_threshold)

    if paths.allow_speakers_path.exists():
        with open(paths.allow_speakers_path, "r", encoding="utf-8") as f:
            allow_payload = json.load(f)
    else:
        allow_payload = {"allow_speakers": []}

    allow_set = set(allow_payload.get("allow_speakers", []))
    allow_set.add(new_name)
    allow_payload["allow_speakers"] = sorted(allow_set)

    with open(paths.allow_speakers_path, "w", encoding="utf-8") as f:
        json.dump(allow_payload, f, ensure_ascii=False, indent=2)

    return {
        "new_speaker_name": new_name,
        "copied_dir": str(copied_dir),
        "speaker_index_path": str(paths.speaker_index_path),
        "checkpoint_path": str(Path(checkpoint_path)),
        "n_recordings": len(wav_paths),
        "n_segments": total_segments,
        "similarity_threshold": similarity_threshold,
        "n_prototypes": len(prototypes),
        "operational_mode": "prototype_incremental",
        "updated_allow_speakers": sorted(allow_set),
    }


def predict_wav_file_operational(
    wav_path: Path | str,
    checkpoint_path: Path | str,
    speaker_index_path: Path | str,
    device: str = DEFAULT_DEVICE,
    segment_seconds: float = DEFAULT_SEGMENT_SECONDS,
    trim_silence: bool = DEFAULT_TRIM_SILENCE,
    trim_top_db: float = DEFAULT_TRIM_TOP_DB,
    n_mels: int = DEFAULT_N_MELS,
) -> Dict[str, Any]:
    """Final-product inference path: one best operational mode based on speaker prototypes."""
    pred = predict_wav_file_with_prototypes(
        wav_path=wav_path,
        checkpoint_path=checkpoint_path,
        speaker_index_path=speaker_index_path,
        device=device,
        segment_seconds=segment_seconds,
        trim_silence=trim_silence,
        trim_top_db=trim_top_db,
        n_mels=n_mels,
    )
    pred["operational_mode"] = "prototype"
    pred["decision_margin"] = float(pred["best_similarity"] - pred["threshold"])
    return pred



def enroll_new_allowed_speaker_operational(
    base_config: ExperimentConfig,
    checkpoint_path: Path | str,
    new_speaker_dir: Path | str,
    speaker_name: Optional[str] = None,
    overwrite_source: bool = False,
) -> Dict[str, Any]:
    return add_new_allowed_speaker_with_prototype(
        base_config=base_config,
        checkpoint_path=checkpoint_path,
        new_speaker_dir=new_speaker_dir,
        speaker_name=speaker_name,
        overwrite_source=overwrite_source,
        persist_updated_config=True,
    )


# =============================================================================
# Noise study
# =============================================================================


def list_noise_files(noise_dir: Path | str) -> List[Path]:
    noise_dir = Path(noise_dir)
    patterns = ["*.wav", "*.flac", "*.mp3", "*.ogg", "*.m4a"]
    files: List[Path] = []
    for pat in patterns:
        files.extend(sorted(noise_dir.rglob(pat)))
    return sorted(set(files))



def generate_colored_noise(length: int, rng: np.random.Generator, kind: str = "white") -> np.ndarray:
    if kind == "white":
        return rng.normal(0.0, 1.0, size=length).astype(np.float32)
    if kind == "pink":
        n_freq = length // 2 + 1
        real = rng.normal(size=n_freq)
        imag = rng.normal(size=n_freq)
        spectrum = real + 1j * imag
        freqs = np.fft.rfftfreq(length)
        freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
        spectrum /= np.sqrt(freqs)
        noise = np.fft.irfft(spectrum, n=length)
        return noise.astype(np.float32)
    raise ValueError(f"Unknown synthetic noise kind: {kind}")




def create_synthetic_noise_files(
    output_dir: Path | str,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration_seconds: float = 30.0,
    n_files_per_kind: int = 5,
    kinds: Sequence[str] = ("white", "pink"),
    seed: int = 123,
) -> List[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    length = int(round(float(duration_seconds) * int(sample_rate)))
    out_paths: List[Path] = []

    for kind in kinds:
        kind = str(kind).lower()
        kind_dir = output_dir / kind
        kind_dir.mkdir(parents=True, exist_ok=True)
        for i in range(int(n_files_per_kind)):
            y = generate_colored_noise(length, rng=rng, kind=kind)
            peak = float(np.max(np.abs(y))) if len(y) else 0.0
            if peak > 1e-8:
                y = 0.8 * y / peak
            out_path = kind_dir / f"{kind}_noise_{i:03d}.wav"
            sf.write(out_path, y.astype(np.float32), int(sample_rate))
            out_paths.append(out_path)
    return out_paths

def sample_real_noise_segment(noise_files: Sequence[Path], target_len: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    if len(noise_files) == 0:
        raise ValueError("No noise files available")
    noise_path = noise_files[int(rng.integers(0, len(noise_files)))]
    y_noise, _ = librosa.load(noise_path, sr=sr, mono=True)
    y_noise = y_noise.astype(np.float32)
    if len(y_noise) == 0:
        return rng.normal(0.0, 1.0, size=target_len).astype(np.float32)
    if len(y_noise) < target_len:
        reps = int(np.ceil(target_len / max(len(y_noise), 1)))
        y_noise = np.tile(y_noise, reps)
    start_max = max(len(y_noise) - target_len, 0)
    start = int(rng.integers(0, start_max + 1)) if start_max > 0 else 0
    return y_noise[start:start + target_len].astype(np.float32)



def mix_noise_at_snr(y_signal: np.ndarray, y_noise: np.ndarray, snr_db: float) -> np.ndarray:
    y_signal = y_signal.astype(np.float32)
    y_noise = y_noise.astype(np.float32)
    if len(y_noise) != len(y_signal):
        y_noise = _pad_or_trim_segment(y_noise, len(y_signal))

    signal_power = float(np.mean(y_signal ** 2))
    noise_power = float(np.mean(y_noise ** 2))
    if signal_power <= 1e-12 or noise_power <= 1e-12:
        return y_signal.copy()

    desired_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    scale = math.sqrt(desired_noise_power / max(noise_power, 1e-12))
    mixed = y_signal + scale * y_noise
    peak = float(np.max(np.abs(mixed)))
    if peak > 1.0:
        mixed = mixed / peak
    return mixed.astype(np.float32)



def add_background_noise(
    y: np.ndarray,
    sr: int,
    noise_config: NoiseStudyConfig,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if noise_config.noise_kind == "real":
        if noise_config.noise_dir is None:
            raise ValueError("noise_dir must be provided when noise_kind='real'")
        noise_files = list_noise_files(noise_config.noise_dir)
        y_noise = sample_real_noise_segment(noise_files, len(y), sr, rng)
    elif noise_config.noise_kind in {"white", "pink"}:
        y_noise = generate_colored_noise(len(y), rng=rng, kind=noise_config.noise_kind)
    else:
        raise ValueError(f"Unknown noise_kind: {noise_config.noise_kind}")
    return mix_noise_at_snr(y, y_noise, snr_db=snr_db)



def predict_binary_from_waveform(
    model: nn.Module,
    ckpt: Dict[str, Any],
    y: np.ndarray,
    sr: int,
    device: str,
    segment_seconds: float,
    trim_silence: bool,
    trim_top_db: float,
    n_mels: int,
) -> Dict[str, Any]:
    data_cfg = resolve_inference_data_config(
        ckpt,
        segment_seconds=segment_seconds,
        trim_silence=trim_silence,
        trim_top_db=trim_top_db,
        n_mels=n_mels,
    )
    y, sr = preprocess_waveform(
        y,
        sr,
        sample_rate=data_cfg.sample_rate,
        trim_silence=data_cfg.trim_silence,
        trim_top_db=data_cfg.trim_top_db,
        remove_dc_offset=data_cfg.remove_dc_offset,
        normalize_peak=data_cfg.normalize_peak,
        peak_target=data_cfg.peak_target,
        normalize_rms=data_cfg.normalize_rms,
        target_rms_db=data_cfg.target_rms_db,
        preemphasis=data_cfg.preemphasis,
    )
    segments = split_waveform_array_to_segments(y, sr, segment_seconds=data_cfg.segment_seconds, keep_remainder=False)

    image_size = int(ckpt.get("image_size", DEFAULT_IMAGE_SIZE))
    train_mean = float(ckpt.get("train_mean", 0.0))
    train_std = float(ckpt.get("train_std", 1.0))
    threshold = float(ckpt.get("threshold", 0.5))
    aggregation_method = str(ckpt.get("aggregation_method", "mean"))

    probs = []
    with torch.no_grad():
        for seg in segments:
            x = _waveform_segment_to_model_tensor(
                seg,
                sr,
                image_size,
                train_mean,
                train_std,
                data_cfg.n_mels,
                spectrogram_mode=data_cfg.spectrogram_mode,
                spec_norm_min_percentile=data_cfg.spec_norm_min_percentile,
                spec_norm_max_percentile=data_cfg.spec_norm_max_percentile,
                mel_fmin=data_cfg.mel_fmin,
                mel_fmax=data_cfg.mel_fmax,
            ).to(device)
            probs.append(float(torch.sigmoid(model(x)).item()))

    if aggregation_method == "mean":
        score = float(np.mean(probs))
    elif aggregation_method == "median":
        score = float(np.median(probs))
    elif aggregation_method == "top3mean":
        score = float(np.mean(np.sort(np.asarray(probs))[-min(3, len(probs)):]))
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")

    return {"score": score, "threshold": threshold, "predicted_label": int(score >= threshold)}



def predict_prototype_from_waveform(
    model: nn.Module,
    ckpt: Dict[str, Any],
    y: np.ndarray,
    sr: int,
    device: str,
    speaker_index: Dict[str, Any],
    segment_seconds: float,
    trim_silence: bool,
    trim_top_db: float,
    n_mels: int,
) -> Dict[str, Any]:
    data_cfg = resolve_inference_data_config(
        ckpt,
        segment_seconds=segment_seconds,
        trim_silence=trim_silence,
        trim_top_db=trim_top_db,
        n_mels=n_mels,
    )
    y, sr = preprocess_waveform(
        y,
        sr,
        sample_rate=data_cfg.sample_rate,
        trim_silence=data_cfg.trim_silence,
        trim_top_db=data_cfg.trim_top_db,
        remove_dc_offset=data_cfg.remove_dc_offset,
        normalize_peak=data_cfg.normalize_peak,
        peak_target=data_cfg.peak_target,
        normalize_rms=data_cfg.normalize_rms,
        target_rms_db=data_cfg.target_rms_db,
        preemphasis=data_cfg.preemphasis,
    )
    segments = split_waveform_array_to_segments(y, sr, segment_seconds=data_cfg.segment_seconds, keep_remainder=False)

    image_size = int(ckpt.get("image_size", DEFAULT_IMAGE_SIZE))
    train_mean = float(ckpt.get("train_mean", 0.0))
    train_std = float(ckpt.get("train_std", 1.0))

    segment_embeddings = []
    with torch.no_grad():
        for seg in segments:
            x = _waveform_segment_to_model_tensor(
                seg,
                sr,
                image_size,
                train_mean,
                train_std,
                data_cfg.n_mels,
                spectrogram_mode=data_cfg.spectrogram_mode,
                spec_norm_min_percentile=data_cfg.spec_norm_min_percentile,
                spec_norm_max_percentile=data_cfg.spec_norm_max_percentile,
                mel_fmin=data_cfg.mel_fmin,
                mel_fmax=data_cfg.mel_fmax,
            ).to(device)
            emb = extract_embedding_batch(model, x).squeeze(0).detach().cpu().numpy().astype(np.float32)
            segment_embeddings.append(emb)

    rec_emb = l2_normalize_np(np.mean(np.stack(segment_embeddings, axis=0), axis=0))
    match = match_embedding_to_speakers(rec_emb, speaker_index["prototypes"])
    threshold = float(speaker_index["similarity_threshold"])
    return {
        "score": float(match["best_similarity"]),
        "threshold": threshold,
        "predicted_label": int(match["best_similarity"] >= threshold),
        "best_speaker": match["best_speaker"],
    }



def evaluate_experiment_under_noise(
    config: ExperimentConfig,
    noise_config: NoiseStudyConfig,
    modes: Sequence[str] = ("binary", "prototype"),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    model, ckpt, paths = load_checkpoint_bundle(config)
    rec_meta = pd.read_csv(paths.recording_meta_path)
    rec_meta = rec_meta[rec_meta["split"] == noise_config.split].copy().reset_index(drop=True)

    if noise_config.n_recordings_per_split is not None and noise_config.n_recordings_per_split < len(rec_meta):
        rng_local = np.random.default_rng(noise_config.seed)
        chosen = rng_local.choice(len(rec_meta), size=noise_config.n_recordings_per_split, replace=False)
        rec_meta = rec_meta.iloc[np.sort(chosen)].reset_index(drop=True)

    speaker_index = load_speaker_index(paths.speaker_index_path)
    if "prototype" in modes and len(speaker_index["prototypes"]) == 0:
        build_speaker_index_for_experiment(config)
        speaker_index = load_speaker_index(paths.speaker_index_path)

    conditions: List[Tuple[str, Optional[float]]] = []
    if noise_config.include_clean:
        conditions.append(("clean", None))
    for snr in noise_config.snr_levels:
        conditions.append(("noisy", float(snr)))

    raw_rows: List[Dict[str, Any]] = []
    for row in tqdm(list(rec_meta.itertuples(index=False)), desc="Noise evaluation"):
        y, sr = librosa.load(row.audio_path, sr=None, mono=True)
        y = y.astype(np.float32)

        for condition, snr in conditions:
            if condition == "clean":
                y_eval = y
            else:
                seed_value = abs(hash((row.recording_id, snr, noise_config.seed))) % (2**32)
                rng = np.random.default_rng(seed_value)
                y_eval = add_background_noise(y, sr, noise_config=noise_config, snr_db=float(snr), rng=rng)

            if "binary" in modes:
                pred = predict_binary_from_waveform(
                    model=model,
                    ckpt=ckpt,
                    y=y_eval,
                    sr=sr,
                    device=config.train.device,
                    segment_seconds=config.data.segment_seconds,
                    trim_silence=config.data.trim_silence,
                    trim_top_db=config.data.trim_top_db,
                    n_mels=config.data.n_mels,
                )
                raw_rows.append(
                    {
                        "mode": "binary",
                        "split": noise_config.split,
                        "noise_kind": "clean" if condition == "clean" else noise_config.noise_kind,
                        "snr_db": np.nan if snr is None else float(snr),
                        "recording_id": row.recording_id,
                        "speaker": row.speaker,
                        "y_true": int(row.label),
                        "score": float(pred["score"]),
                        "threshold": float(pred["threshold"]),
                        "predicted_label": int(pred["predicted_label"]),
                    }
                )

            if "prototype" in modes:
                pred = predict_prototype_from_waveform(
                    model=model,
                    ckpt=ckpt,
                    y=y_eval,
                    sr=sr,
                    device=config.train.device,
                    speaker_index=speaker_index,
                    segment_seconds=config.data.segment_seconds,
                    trim_silence=config.data.trim_silence,
                    trim_top_db=config.data.trim_top_db,
                    n_mels=config.data.n_mels,
                )
                raw_rows.append(
                    {
                        "mode": "prototype",
                        "split": noise_config.split,
                        "noise_kind": "clean" if condition == "clean" else noise_config.noise_kind,
                        "snr_db": np.nan if snr is None else float(snr),
                        "recording_id": row.recording_id,
                        "speaker": row.speaker,
                        "y_true": int(row.label),
                        "score": float(pred["score"]),
                        "threshold": float(pred["threshold"]),
                        "predicted_label": int(pred["predicted_label"]),
                    }
                )

    raw_df = pd.DataFrame(raw_rows)
    summary_rows: List[Dict[str, Any]] = []
    for (mode, noise_kind, snr_db), g in raw_df.groupby(["mode", "noise_kind", "snr_db"], dropna=False):
        metrics = binary_metrics_from_probs(g["y_true"].to_numpy(), g["score"].to_numpy(), float(g["threshold"].iloc[0]))
        summary_rows.append(
            {
                "experiment_name": config.experiment_name,
                "mode": mode,
                "split": noise_config.split,
                "noise_kind": noise_kind,
                "snr_db": snr_db,
                **metrics,
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(["mode", "snr_db"], na_position="first").reset_index(drop=True)

    raw_df.to_csv(paths.results_dir / f"{config.experiment_name}_noise_eval_raw_{noise_config.split}.csv", index=False)
    summary_df.to_csv(paths.results_dir / f"{config.experiment_name}_noise_eval_summary_{noise_config.split}.csv", index=False)
    return raw_df, summary_df

# =============================================================================
# Experiment comparison
# =============================================================================

def run_full_experiment(
    config: ExperimentConfig,
    rebuild_data: bool = False,
    clean_intermediate: bool = False,
    init_checkpoint_path: Optional[Path | str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    history = train_experiment(
        config=config,
        rebuild_data=rebuild_data,
        clean_intermediate=clean_intermediate,
        init_checkpoint_path=init_checkpoint_path,
    )
    binary_results = evaluate_saved_experiment(config=config)
    prototype_summary = build_speaker_index_for_experiment(config=config)
    return history, binary_results, prototype_summary




def compare_experiments(
    experiment_configs: Sequence[ExperimentConfig],
    rebuild_data_for_first: bool = False,
    clean_intermediate_for_first: bool = False,
) -> pd.DataFrame:
    all_rows: List[Dict[str, Any]] = []
    for i, cfg in enumerate(experiment_configs):
        _, binary_results, prototype_summary = run_full_experiment(
            config=cfg,
            rebuild_data=rebuild_data_for_first if i == 0 else False,
            clean_intermediate=clean_intermediate_for_first if i == 0 else False,
            init_checkpoint_path=None,
        )
        valid_row = binary_results[binary_results["split"] == "valid"].iloc[0].to_dict()
        test_row = binary_results[binary_results["split"] == "test"].iloc[0].to_dict()
        checkpoint_path = _experiment_checkpoint_path(get_paths(cfg), cfg.experiment_name)
        all_rows.append(
            {
                "experiment_name": cfg.experiment_name,
                "model_name": cfg.model.model_name,
                "checkpoint_path": str(checkpoint_path),
                "data_signature": get_paths(cfg).data_signature,
                "selection_split": "valid",
                "binary_valid_recording_balanced_error": valid_row["recording_balanced_error"],
                "binary_valid_recording_far": valid_row["recording_far"],
                "binary_valid_recording_frr": valid_row["recording_frr"],
                "binary_valid_recording_acc": valid_row["recording_acc"],
                "prototype_valid_balanced_error": prototype_summary["valid_metrics"]["balanced_error"],
                "prototype_valid_far": prototype_summary["valid_metrics"]["far"],
                "prototype_valid_frr": prototype_summary["valid_metrics"]["frr"],
                "binary_test_recording_balanced_error": test_row["recording_balanced_error"],
                "binary_test_recording_far": test_row["recording_far"],
                "binary_test_recording_frr": test_row["recording_frr"],
                "binary_test_recording_acc": test_row["recording_acc"],
                "prototype_test_balanced_error": prototype_summary["test_metrics"]["balanced_error"],
                "prototype_test_far": prototype_summary["test_metrics"]["far"],
                "prototype_test_frr": prototype_summary["test_metrics"]["frr"],
                "prototype_similarity_threshold": prototype_summary["similarity_threshold"],
                "n_prototypes": prototype_summary["n_prototypes"],
            }
        )
    comparison_df = pd.DataFrame(all_rows).sort_values(
        ["prototype_valid_balanced_error", "binary_valid_recording_balanced_error", "experiment_name"]
    ).reset_index(drop=True)
    paths = get_paths(experiment_configs[0])
    comparison_df.to_csv(paths.results_dir / "comparison_results.csv", index=False)
    return comparison_df


__all__ = [
    "ProjectPaths",
    "DataPrepConfig",
    "ModelConfig",
    "TrainConfig",
    "PrototypeConfig",
    "NoiseStudyConfig",
    "ExperimentConfig",
    "get_paths",
    "set_seed",
    "prepare_dirs",
    "clone_experiment",
    "save_experiment_config",
    "load_experiment_config",
    "get_experiment_checkpoint_path",
    "prepare_data_artifacts",
    "data_artifacts_exist",
    "data_config_signature",
    "make_loaders",
    "train_experiment",
    "evaluate_saved_experiment",
    "run_full_experiment",
    "compare_experiments",
    "optimize_experiment_with_optuna",
    "build_speaker_index_for_experiment",
    "build_speaker_index_from_checkpoint",
    "load_speaker_index",
    "predict_wav_file_binary",
    "predict_wav_file_with_prototypes",
    "add_new_allowed_speaker_with_prototype",
    "predict_wav_file_operational",
    "enroll_new_allowed_speaker_operational",
    "evaluate_experiment_under_noise",
    "create_synthetic_noise_files",
    "mix_noise_at_snr",
    "add_background_noise",
]
