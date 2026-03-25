from __future__ import annotations

import copy
import json
import math
import os
import random
import shutil
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
# Configs and paths
# =============================================================================


@dataclass
class ProjectPaths:
    work_dir: Path
    recording_meta_path: Path
    segment_meta_path: Path
    spectro_meta_path: Path
    allow_speakers_path: Path
    segment_dir: Path
    spectro_dir: Path
    model_dir: Path
    results_dir: Path
    config_dir: Path

    @classmethod
    def from_work_dir(cls, work_dir: Path | str) -> "ProjectPaths":
        work_dir = Path(work_dir)
        return cls(
            work_dir=work_dir,
            recording_meta_path=work_dir / "metadata_recording_level.csv",
            segment_meta_path=work_dir / "metadata_segment_level.csv",
            spectro_meta_path=work_dir / "metadata_spectrogram_level.csv",
            allow_speakers_path=work_dir / "allow_speakers.json",
            segment_dir=work_dir / "segmented_audio",
            spectro_dir=work_dir / "spectrograms_npy",
            model_dir=work_dir / "models",
            results_dir=work_dir / "results",
            config_dir=work_dir / "configs",
        )


@dataclass
class DataPrepConfig:
    source_dir: Path | str
    work_dir: Path | str = Path("voice_access_files")
    allow_speakers: Optional[List[str]] = None
    seed: int = 123
    train_ratio: float = 0.7
    valid_ratio: float = 0.1
    target_allow_speakers_in_valid: int = 2
    segment_seconds: float = 3.0
    keep_remainder: bool = False
    trim_silence: bool = True
    trim_top_db: float = 25.0
    image_size: int = 128
    n_mels: int = 128


@dataclass
class ModelConfig:
    dropout: float = 0.35
    use_batchnorm: bool = True


@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 15
    lr: float = 3e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 6
    lr_scheduler_patience: int = 2
    lr_scheduler_factor: float = 0.5
    max_grad_norm: float = 5.0
    aggregation_method: str = "top3mean"
    augment: bool = True
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    num_workers: Optional[int] = None


@dataclass
class ExperimentConfig:
    experiment_name: str
    data: DataPrepConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# =============================================================================
# Basic utilities
# =============================================================================


def get_paths(config_or_work_dir: DataPrepConfig | ExperimentConfig | Path | str) -> ProjectPaths:
    if isinstance(config_or_work_dir, ExperimentConfig):
        return ProjectPaths.from_work_dir(config_or_work_dir.data.work_dir)
    if isinstance(config_or_work_dir, DataPrepConfig):
        return ProjectPaths.from_work_dir(config_or_work_dir.work_dir)
    return ProjectPaths.from_work_dir(config_or_work_dir)


def set_seed(seed: int = 123) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_dirs(paths: ProjectPaths, clean_intermediate: bool = False) -> None:
    paths.work_dir.mkdir(parents=True, exist_ok=True)
    paths.model_dir.mkdir(parents=True, exist_ok=True)
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    paths.config_dir.mkdir(parents=True, exist_ok=True)

    if clean_intermediate:
        for p in [paths.segment_dir, paths.spectro_dir]:
            if p.exists():
                shutil.rmtree(p)

    paths.segment_dir.mkdir(parents=True, exist_ok=True)
    paths.spectro_dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Metadata building
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
    min_valid: int = 0,
    min_test: int = 1,
) -> Tuple[List[str], List[str], List[str]]:
    files = list(files)
    rng.shuffle(files)
    n = len(files)

    if n == 0:
        return [], [], []
    if n == 1:
        if min_valid > 0:
            return [], [files[0]], []
        return [files[0]], [], []
    if n == 2:
        if min_valid > 0:
            return [files[0]], [files[1]], []
        return [files[0]], [], [files[1]]

    n_train = max(1, int(round(n * train_ratio)))
    n_valid = int(round(n * valid_ratio))
    n_test = n - n_train - n_valid

    if n_valid < min_valid:
        take = min_valid - n_valid
        n_valid += take
        n_train -= take

    if n_test < min_test:
        take = min_test - n_test
        n_test += take
        n_train -= take

    while n_train < 1:
        if n_valid > min_valid:
            n_valid -= 1
            n_train += 1
        elif n_test > min_test:
            n_test -= 1
            n_train += 1
        else:
            break

    if n_train < 1:
        n_train = 1
        remaining = n - n_train
        n_valid = min(n_valid, remaining)
        n_test = n - n_train - n_valid

    train_files = files[:n_train]
    valid_files = files[n_train:n_train + n_valid]
    test_files = files[n_train + n_valid:n_train + n_valid + n_test]

    if len(test_files) == 0 and min_test > 0 and len(train_files) >= 2:
        test_files = [train_files.pop()]
    if len(valid_files) == 0 and min_valid > 0 and len(train_files) >= 2:
        valid_files = [train_files.pop()]

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
    eligible = [s for s in allow_speakers if len(speaker_to_wavs[s]) >= 2]
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
    all_speakers = sorted(speaker_to_wavs.keys())
    if not all_speakers:
        raise ValueError(f"No speakers with .wav files found in {source_dir}")

    allow_speakers = config.allow_speakers
    if allow_speakers is None:
        allow_speakers = all_speakers[:5]

    allow_speakers = sorted(allow_speakers)
    missing = [s for s in allow_speakers if s not in speaker_to_wavs]
    if missing:
        raise ValueError(f"Allow speakers not found in source_dir: {missing}")

    not_allow_speakers = [s for s in all_speakers if s not in allow_speakers]
    allow_speakers_with_valid = pick_allow_speakers_for_validation(
        allow_speakers=allow_speakers,
        speaker_to_wavs=speaker_to_wavs,
        target_allow_speakers_in_valid=config.target_allow_speakers_in_valid,
        rng=rng,
    )

    rows: List[Dict[str, Any]] = []

    for speaker in allow_speakers:
        wavs = speaker_to_wavs[speaker]
        force_valid = speaker in allow_speakers_with_valid
        train_wavs, valid_wavs, test_wavs = _split_files_within_speaker(
            wavs,
            rng=rng,
            train_ratio=config.train_ratio,
            valid_ratio=config.valid_ratio,
            min_valid=1 if force_valid else 0,
            min_test=0 if force_valid else 1,
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
    possible_allow_valid_speakers = sum(meta.loc[meta["label"] == 1].groupby("speaker")["audio_path"].nunique() >= 2)
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


def trim_silence_waveform(y: np.ndarray, top_db: float = 25.0) -> np.ndarray:
    if len(y) == 0:
        return y.astype(np.float32)
    yt, _ = librosa.effects.trim(y, top_db=top_db)
    if len(yt) == 0:
        return y.astype(np.float32)
    return yt.astype(np.float32)


def segment_recordings(config: DataPrepConfig, paths: Optional[ProjectPaths] = None) -> pd.DataFrame:
    paths = paths or get_paths(config)
    rec_meta = pd.read_csv(paths.recording_meta_path)
    rows: List[Dict[str, Any]] = []

    for row in tqdm(list(rec_meta.itertuples(index=False)), desc="Segmenting recordings"):
        y, sr = librosa.load(row.audio_path, sr=None, mono=True)
        y = y.astype(np.float32)

        original_seconds = len(y) / max(sr, 1)
        if config.trim_silence:
            y = trim_silence_waveform(y, top_db=config.trim_top_db)
        trimmed_seconds = len(y) / max(sr, 1)

        segment_len = int(round(config.segment_seconds * sr))

        if len(y) <= segment_len:
            seg = _pad_or_trim_segment(y, segment_len)
            segment_name = f"{row.recording_id}__seg0000.wav"
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
                    "segment_index": 0,
                    "segment_path": str(out_path),
                    "sr": sr,
                    "segment_seconds": config.segment_seconds,
                    "original_seconds": original_seconds,
                    "trimmed_seconds": trimmed_seconds,
                }
            )
            continue

        seg_idx = 0
        for start in range(0, len(y), segment_len):
            end = start + segment_len
            seg = y[start:end]

            if len(seg) < segment_len:
                if not config.keep_remainder:
                    break
                seg = _pad_or_trim_segment(seg, segment_len)

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
                    "trimmed_seconds": trimmed_seconds,
                }
            )
            seg_idx += 1

    seg_meta = pd.DataFrame(rows).sort_values(["split", "label", "speaker", "recording_id", "segment_index"]).reset_index(drop=True)
    seg_meta.to_csv(paths.segment_meta_path, index=False)
    return seg_meta


def mel_spectrogram_float32(
    y: np.ndarray,
    sr: int,
    image_size: int = 128,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 256,
    fmin: float = 20.0,
    fmax: Optional[float] = None,
) -> np.ndarray:
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
    mel_db = np.clip(mel_db, a_min=-80.0, a_max=0.0)
    mel_db = (mel_db + 80.0) / 80.0

    x = torch.from_numpy(mel_db.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)
    arr = x.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
    return np.clip(arr, 0.0, 1.0)


def build_spectrogram_arrays(config: DataPrepConfig, paths: Optional[ProjectPaths] = None) -> pd.DataFrame:
    paths = paths or get_paths(config)
    seg_meta = pd.read_csv(paths.segment_meta_path)
    rows: List[Dict[str, Any]] = []

    for row in tqdm(list(seg_meta.itertuples(index=False)), desc="Building spectrograms"):
        y, sr = librosa.load(row.segment_path, sr=None, mono=True)
        arr = mel_spectrogram_float32(y, sr=sr, image_size=config.image_size, n_mels=config.n_mels)

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
    image_size: int = 128,
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
        image_size: int = 128,
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


def compute_train_mean_std(train_df: pd.DataFrame, image_size: int = 128) -> Tuple[float, float]:
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
    image_size: int = 128,
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

    use_pin_memory = str(device).startswith("cuda")
    loader_kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": use_pin_memory,
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
# Model and metrics
# =============================================================================


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


class SmallCNN(nn.Module):
    def __init__(self, dropout: float = 0.35, use_batchnorm: bool = True):
        super().__init__()
        self.stem = ConvBNAct(1, 32, kernel_size=5, stride=1, use_batchnorm=use_batchnorm)
        self.features = nn.Sequential(
            ResidualBlock(32, 32, stride=1, dropout=0.0, use_batchnorm=use_batchnorm),
            ResidualBlock(32, 64, stride=2, dropout=dropout * 0.3, use_batchnorm=use_batchnorm),
            ResidualBlock(64, 64, stride=1, dropout=dropout * 0.3, use_batchnorm=use_batchnorm),
            ResidualBlock(64, 128, stride=2, dropout=dropout * 0.5, use_batchnorm=use_batchnorm),
            ResidualBlock(128, 128, stride=1, dropout=dropout * 0.5, use_batchnorm=use_batchnorm),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x.squeeze(1)


def build_model(model_config: ModelConfig) -> nn.Module:
    return SmallCNN(dropout=model_config.dropout, use_batchnorm=model_config.use_batchnorm)


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


def binary_metrics_from_probs(y_true: np.ndarray, p_allow: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    p_allow = np.asarray(p_allow).astype(float)
    y_pred = (p_allow >= threshold).astype(int)

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


def compute_speaker_level_metrics(recording_df: pd.DataFrame, threshold: float, aggregation_method: str = "mean") -> Dict[str, Any]:
    if len(recording_df) == 0:
        empty_df = pd.DataFrame(columns=[
            "speaker", "y_true", "n_recordings", "speaker_prob", "speaker_pred",
            "recording_accept_rate", "recording_reject_rate",
            "speaker_far_local", "speaker_frr_local",
            "speaker_far_hard", "speaker_frr_hard",
        ])
        empty_metrics = {
            "macro_far_recording": np.nan,
            "macro_frr_recording": np.nan,
            "macro_balanced_error_recording": np.nan,
            "gap_far_frr_recording": np.nan,
            "far_speaker_hard": np.nan,
            "frr_speaker_hard": np.nan,
            "balanced_error_speaker_hard": np.nan,
            "gap_far_frr_speaker_hard": np.nan,
            "n_allow_speakers": 0,
            "n_not_allow_speakers": 0,
        }
        return {"speaker_df": empty_df, "speaker_metrics": empty_metrics}

    rows: List[Dict[str, Any]] = []
    for speaker, g in recording_df.groupby("speaker"):
        y_true = int(g["y_true"].iloc[0])
        probs = g["p_allow"].to_numpy(dtype=float)
        rec_preds = (probs >= threshold).astype(int)

        if aggregation_method == "mean":
            speaker_prob = float(np.mean(probs))
        elif aggregation_method == "median":
            speaker_prob = float(np.median(probs))
        elif aggregation_method == "top3mean":
            k = min(3, len(probs))
            speaker_prob = float(np.mean(np.sort(probs)[-k:]))
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")

        speaker_pred = int(speaker_prob >= threshold)
        rows.append(
            {
                "speaker": speaker,
                "y_true": y_true,
                "n_recordings": int(len(g)),
                "speaker_prob": speaker_prob,
                "speaker_pred": speaker_pred,
                "recording_accept_rate": float(np.mean(rec_preds)),
                "recording_reject_rate": float(np.mean(1 - rec_preds)),
                "speaker_far_local": float(np.mean(rec_preds)) if y_true == 0 else np.nan,
                "speaker_frr_local": float(np.mean(1 - rec_preds)) if y_true == 1 else np.nan,
                "speaker_far_hard": float(speaker_pred == 1) if y_true == 0 else np.nan,
                "speaker_frr_hard": float(speaker_pred == 0) if y_true == 1 else np.nan,
            }
        )

    speaker_df = pd.DataFrame(rows).sort_values(["y_true", "speaker"]).reset_index(drop=True)
    far_local = float(speaker_df["speaker_far_local"].dropna().mean())
    frr_local = float(speaker_df["speaker_frr_local"].dropna().mean())
    far_hard = float(speaker_df["speaker_far_hard"].dropna().mean())
    frr_hard = float(speaker_df["speaker_frr_hard"].dropna().mean())

    metrics = {
        "macro_far_recording": far_local,
        "macro_frr_recording": frr_local,
        "macro_balanced_error_recording": 0.5 * (far_local + frr_local),
        "gap_far_frr_recording": abs(far_local - frr_local),
        "far_speaker_hard": far_hard,
        "frr_speaker_hard": frr_hard,
        "balanced_error_speaker_hard": 0.5 * (far_hard + frr_hard),
        "gap_far_frr_speaker_hard": abs(far_hard - frr_hard),
        "n_allow_speakers": int((speaker_df["y_true"] == 1).sum()),
        "n_not_allow_speakers": int((speaker_df["y_true"] == 0).sum()),
    }
    return {"speaker_df": speaker_df, "speaker_metrics": metrics}


def find_best_threshold(y_true: np.ndarray, p_allow: np.ndarray, penalty_gap: float = 0.25) -> Dict[str, float]:
    rows: List[Dict[str, float]] = []
    for t in np.linspace(0.01, 0.99, 99):
        m = binary_metrics_from_probs(y_true, p_allow, threshold=float(t))
        score = m["balanced_error"] + penalty_gap * m["gap_far_frr"]
        rows.append({"threshold": float(t), "score": float(score), **m})
    th_df = pd.DataFrame(rows).sort_values(["score", "balanced_error", "gap_far_frr", "threshold"]).reset_index(drop=True)
    return th_df.iloc[0].to_dict()


def evaluate_model(model: nn.Module, loader: DataLoader, device: str, threshold: float, aggregation_method: str = "mean") -> Dict[str, Any]:
    seg_df = collect_segment_predictions(model, loader, device=device)
    rec_df = aggregate_recording_predictions(seg_df, method=aggregation_method)

    segment_metrics = binary_metrics_from_probs(
        y_true=seg_df["y_true"].to_numpy(),
        p_allow=seg_df["p_allow_segment"].to_numpy(),
        threshold=threshold,
    )
    recording_metrics = binary_metrics_from_probs(
        y_true=rec_df["y_true"].to_numpy(),
        p_allow=rec_df["p_allow"].to_numpy(),
        threshold=threshold,
    )
    speaker_eval = compute_speaker_level_metrics(recording_df=rec_df, threshold=threshold, aggregation_method=aggregation_method)

    return {
        "segment_df": seg_df,
        "recording_df": rec_df,
        "speaker_df": speaker_eval["speaker_df"],
        "segment_metrics": segment_metrics,
        "recording_metrics": recording_metrics,
        "speaker_metrics": speaker_eval["speaker_metrics"],
    }


# =============================================================================
# Training and experiment management
# =============================================================================

def compute_loader_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> float:
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

def _experiment_checkpoint_path(paths: ProjectPaths, experiment_name: str) -> Path:
    return paths.model_dir / f"{experiment_name}.pt"


def _experiment_config_path(paths: ProjectPaths, experiment_name: str) -> Path:
    return paths.config_dir / f"{experiment_name}.json"


def _save_experiment_config(config: ExperimentConfig, paths: Optional[ProjectPaths] = None) -> Path:
    paths = paths or get_paths(config)
    out_path = _experiment_config_path(paths, config.experiment_name)
    payload = asdict(config)
    payload["data"]["source_dir"] = str(payload["data"]["source_dir"])
    payload["data"]["work_dir"] = str(payload["data"]["work_dir"])
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out_path


def load_experiment_config(work_dir: Path | str, experiment_name: str) -> ExperimentConfig:
    paths = get_paths(work_dir)
    with open(_experiment_config_path(paths, experiment_name), "r", encoding="utf-8") as f:
        payload = json.load(f)
    data = DataPrepConfig(**payload["data"])
    model = ModelConfig(**payload["model"])
    train = TrainConfig(**payload["train"])
    return ExperimentConfig(experiment_name=payload["experiment_name"], data=data, model=model, train=train)


def clone_experiment(
    base_config: ExperimentConfig,
    *,
    experiment_name: str,
    data_updates: Optional[Dict[str, Any]] = None,
    model_updates: Optional[Dict[str, Any]] = None,
    train_updates: Optional[Dict[str, Any]] = None,
) -> ExperimentConfig:
    data = replace(base_config.data, **(data_updates or {}))
    model = replace(base_config.model, **(model_updates or {}))
    train = replace(base_config.train, **(train_updates or {}))
    return ExperimentConfig(experiment_name=experiment_name, data=data, model=model, train=train)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: str,
    epochs: int = 15,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    checkpoint_path: Optional[Path] = None,
    train_mean: Optional[float] = None,
    train_std: Optional[float] = None,
    image_size: Optional[int] = None,
    model_config: Optional[dict] = None,
    early_stopping_patience: int = 5,
    lr_scheduler_patience: int = 2,
    lr_scheduler_factor: float = 0.5,
    max_grad_norm: float = 5.0,
    aggregation_method: str = "mean",
) -> pd.DataFrame:
    model = model.to(device)

    train_targets: List[float] = []
    for _, y, *_ in train_loader:
        train_targets.extend(y.tolist())
    train_targets = np.asarray(train_targets, dtype=np.float32)

    n_pos = float((train_targets == 1).sum())
    n_neg = float((train_targets == 0).sum())
    pos_weight_value = n_neg / max(n_pos, 1.0)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_scheduler_factor,
        patience=lr_scheduler_patience,
    )

    history_rows: List[Dict[str, Any]] = []
    best_valid_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    epoch_bar = tqdm(range(1, epochs + 1), desc="Training epochs", leave=True)

    best_threshold = 0.5
    best_bacc = 0.0

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

            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

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
            valid_rec_df = aggregate_recording_predictions(valid_seg_df, method=aggregation_method)

            best_thr_info = find_best_threshold(
                y_true=valid_rec_df["y_true"].to_numpy(),
                p_allow=valid_rec_df["p_allow"].to_numpy(),
            )

            best_threshold = float(best_thr_info["threshold"])
            best_bacc = float(1.0 - best_thr_info["balanced_error"])

            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "image_size": image_size,
                        "train_mean": train_mean,
                        "train_std": train_std,
                        "threshold": best_threshold,
                        "aggregation_method": aggregation_method,
                        "model_config": model_config or {},
                        "best_valid_loss": best_valid_loss,
                        "best_epoch": best_epoch,
                        "best_valid_balanced_accuracy": best_bacc,
                        "best_valid_threshold_info": best_thr_info,
                    },
                    checkpoint_path,
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
            }
        )

        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            valid_loss=f"{valid_loss:.4f}",
            best=f"{best_valid_loss:.4f}",
        )

        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    return pd.DataFrame(history_rows)

def train_experiment(config: ExperimentConfig, rebuild_data: bool = False, clean_intermediate: bool = False) -> pd.DataFrame:
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
        epochs=config.train.epochs,
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
        checkpoint_path=checkpoint_path,
        train_mean=stats["train_mean"],
        train_std=stats["train_std"],
        image_size=config.data.image_size,
        model_config=asdict(config.model),
        early_stopping_patience=config.train.early_stopping_patience,
        lr_scheduler_patience=config.train.lr_scheduler_patience,
        lr_scheduler_factor=config.train.lr_scheduler_factor,
        max_grad_norm=config.train.max_grad_norm,
        aggregation_method=config.train.aggregation_method,
    )

    history.to_csv(paths.results_dir / f"{config.experiment_name}_history.csv", index=False)
    _save_experiment_config(config, paths=paths)
    return history


def _load_checkpoint_bundle(config: ExperimentConfig) -> Tuple[nn.Module, Dict[str, Any], ProjectPaths]:
    paths = get_paths(config)
    checkpoint_path = _experiment_checkpoint_path(paths, config.experiment_name)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=config.train.device)
    model_cfg = ModelConfig(**ckpt.get("model_config", asdict(config.model)))
    model = build_model(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(config.train.device)
    return model, ckpt, paths

def evaluate_saved_experiment(config: ExperimentConfig) -> pd.DataFrame:
    model, ckpt, paths = _load_checkpoint_bundle(config)

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

    train_eval = evaluate_model(
        model,
        train_eval_loader,
        device=config.train.device,
        threshold=threshold,
        aggregation_method=aggregation_method,
    )
    valid_eval = evaluate_model(
        model,
        valid_loader,
        device=config.train.device,
        threshold=threshold,
        aggregation_method=aggregation_method,
    )
    test_eval = evaluate_model(
        model,
        test_loader,
        device=config.train.device,
        threshold=threshold,
        aggregation_method=aggregation_method,
    )

    results = pd.DataFrame(
        [
            {
                "experiment_name": config.experiment_name,
                "threshold": threshold,
                "aggregation_method": aggregation_method,
                "split": "train",
                **{f"segment_{k}": v for k, v in train_eval["segment_metrics"].items()},
                **{f"recording_{k}": v for k, v in train_eval["recording_metrics"].items()},
                **{f"speaker_{k}": v for k, v in train_eval["speaker_metrics"].items()},
            },
            {
                "experiment_name": config.experiment_name,
                "threshold": threshold,
                "aggregation_method": aggregation_method,
                "split": "valid",
                **{f"segment_{k}": v for k, v in valid_eval["segment_metrics"].items()},
                **{f"recording_{k}": v for k, v in valid_eval["recording_metrics"].items()},
                **{f"speaker_{k}": v for k, v in valid_eval["speaker_metrics"].items()},
            },
            {
                "experiment_name": config.experiment_name,
                "threshold": threshold,
                "aggregation_method": aggregation_method,
                "split": "test",
                **{f"segment_{k}": v for k, v in test_eval["segment_metrics"].items()},
                **{f"recording_{k}": v for k, v in test_eval["recording_metrics"].items()},
                **{f"speaker_{k}": v for k, v in test_eval["speaker_metrics"].items()},
            },
        ]
    )

    results.to_csv(paths.results_dir / f"{config.experiment_name}_evaluation.csv", index=False)
    return results

def run_full_experiment(config: ExperimentConfig, rebuild_data: bool = False, clean_intermediate: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    history = train_experiment(config=config, rebuild_data=rebuild_data, clean_intermediate=clean_intermediate)
    results = evaluate_saved_experiment(config=config)
    return history, results


def compare_experiments(
    experiment_configs: Sequence[ExperimentConfig],
    rebuild_data_for_first: bool = False,
    clean_intermediate_for_first: bool = False,
) -> pd.DataFrame:
    all_results: List[pd.DataFrame] = []
    for i, cfg in enumerate(experiment_configs):
        _, results = run_full_experiment(
            config=cfg,
            rebuild_data=rebuild_data_for_first if i == 0 else False,
            clean_intermediate=clean_intermediate_for_first if i == 0 else False,
        )
        all_results.append(results)

    comparison_df = pd.concat(all_results, ignore_index=True)
    comparison_df = comparison_df.sort_values(
        ["test_balanced_error", "test_gap_far_frr", "test_balanced_error_speaker_hard", "experiment_name"]
    ).reset_index(drop=True)

    work_dir = experiment_configs[0].data.work_dir
    paths = get_paths(work_dir)
    comparison_df.to_csv(paths.results_dir / "comparison_results.csv", index=False)
    return comparison_df


# =============================================================================
# Optional helper for adding a new allowed speaker
# =============================================================================


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
    speaker_to_wavs = load_speaker_to_wavs(config.source_dir)
    if new_speaker_name not in speaker_to_wavs:
        raise ValueError(f"Speaker {new_speaker_name!r} not found in source_dir")

    current_allow = config.allow_speakers
    if current_allow is None:
        current_allow = sorted(speaker_to_wavs.keys())[:5]
    current_allow = sorted(set(current_allow) | {new_speaker_name})
    return replace(config, allow_speakers=current_allow)


__all__ = [
    "ProjectPaths",
    "DataPrepConfig",
    "ModelConfig",
    "TrainConfig",
    "ExperimentConfig",
    "get_paths",
    "set_seed",
    "prepare_data_artifacts",
    "train_experiment",
    "evaluate_saved_experiment",
    "run_full_experiment",
    "compare_experiments",
    "clone_experiment",
    "load_experiment_config",
    "add_new_allowed_speaker_recordings",
    "extend_allow_speakers",
]
