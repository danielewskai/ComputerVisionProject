from __future__ import annotations

import json
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Default config
# =========================
DEFAULT_WORK_DIR = Path("voice_access_files")
DEFAULT_SEGMENT_SECONDS = 3.0
DEFAULT_TRIM_SILENCE = True
DEFAULT_TRIM_TOP_DB = 25.0
DEFAULT_N_MELS = 128
DEFAULT_IMAGE_SIZE = 128
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Project paths
# =========================
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
    app_added_dir: Path

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
            app_added_dir=work_dir / "app_added_allow_speakers",
        )


def prepare_dirs(paths: ProjectPaths) -> None:
    for p in [
        paths.work_dir,
        paths.model_dir,
        paths.results_dir,
        paths.segment_dir,
        paths.spectro_dir,
        paths.app_added_dir,
    ]:
        p.mkdir(parents=True, exist_ok=True)


# =========================
# Model
# =========================
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
    def __init__(self, dropout: float = 0.3, use_batchnorm: bool = True):
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


# =========================
# Audio helpers
# =========================
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


def mel_spectrogram_float32(
    y: np.ndarray,
    sr: int,
    image_size: int = DEFAULT_IMAGE_SIZE,
    n_mels: int = DEFAULT_N_MELS,
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


# =========================
# Prediction
# =========================
def load_checkpoint_and_model(checkpoint_path: Path | str, device: str = DEFAULT_DEVICE) -> Tuple[nn.Module, dict]:
    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device)

    model_cfg = ckpt.get("model_config", {})
    model = SmallCNN(
        dropout=float(model_cfg.get("dropout", 0.3)),
        use_batchnorm=bool(model_cfg.get("use_batchnorm", True)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, ckpt


def predict_wav_file(
    wav_path: Path | str,
    checkpoint_path: Path | str,
    device: str = DEFAULT_DEVICE,
    segment_seconds: float = DEFAULT_SEGMENT_SECONDS,
    trim_silence: bool = DEFAULT_TRIM_SILENCE,
    trim_top_db: float = DEFAULT_TRIM_TOP_DB,
    n_mels: int = DEFAULT_N_MELS,
) -> dict:
    wav_path = Path(wav_path)
    model, ckpt = load_checkpoint_and_model(checkpoint_path, device=device)

    image_size = int(ckpt.get("image_size", DEFAULT_IMAGE_SIZE))
    train_mean = float(ckpt.get("train_mean", 0.0))
    train_std = float(ckpt.get("train_std", 1.0))
    threshold = float(ckpt.get("threshold", 0.5))
    aggregation_method = ckpt.get("aggregation_method", "mean")

    y, sr = librosa.load(wav_path, sr=None, mono=True)
    if trim_silence:
        y = trim_silence_waveform(y, top_db=trim_top_db)

    segment_len = int(round(segment_seconds * sr))

    if len(y) <= segment_len:
        segments = [_pad_or_trim_segment(y, segment_len)]
    else:
        segments = []
        for start in range(0, len(y), segment_len):
            end = start + segment_len
            seg = y[start:end]
            if len(seg) < segment_len:
                break
            segments.append(seg.astype(np.float32))

    probs = []
    arrays = []
    with torch.no_grad():
        for seg in segments:
            arr = mel_spectrogram_float32(seg, sr=sr, image_size=image_size, n_mels=n_mels)
            arrays.append(arr)
            x = torch.from_numpy(arr).unsqueeze(0)
            x = (x - train_mean) / max(train_std, 1e-8)
            x = x.unsqueeze(0).to(device)
            prob = torch.sigmoid(model(x)).item()
            probs.append(float(prob))

    if len(probs) == 0:
        raise ValueError("No valid segments were produced from this recording.")

    if aggregation_method == "mean":
        recording_prob = float(np.mean(probs))
    elif aggregation_method == "median":
        recording_prob = float(np.median(probs))
    elif aggregation_method == "top3mean":
        k = min(3, len(probs))
        recording_prob = float(np.mean(np.sort(np.array(probs))[-k:]))
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")

    pred = int(recording_prob >= threshold)
    return {
        "wav_path": str(wav_path),
        "sample_rate": int(sr),
        "n_segments": len(probs),
        "segment_probs_allow": probs,
        "recording_prob_allow": recording_prob,
        "threshold": threshold,
        "predicted_label": pred,
        "predicted_class_name": "allow" if pred == 1 else "not_allow",
        "aggregation_method": aggregation_method,
        "spectrogram_arrays": arrays,
    }


# =========================
# Metadata / adding new allowed speaker
# =========================
def _split_files_within_speaker(
    files: List[str],
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


def validate_recording_metadata(meta: pd.DataFrame) -> Dict[str, pd.DataFrame | int]:
    dup = int(meta.duplicated(subset=["audio_path"]).sum())
    split_counts = meta.groupby("audio_path")["split"].nunique()
    same_audio_in_many_splits = int((split_counts > 1).sum())

    summary = meta.groupby(["split", "class_name"]).size().unstack(fill_value=0)
    speakers = meta.groupby(["split", "class_name"])["speaker"].nunique().unstack(fill_value=0)
    allow_valid_speakers = int(meta.loc[(meta["label"] == 1) & (meta["split"] == "valid"), "speaker"].nunique())

    return {
        "duplicate_audio_rows": dup,
        "same_audio_in_many_splits": same_audio_in_many_splits,
        "summary": summary,
        "speakers": speakers,
        "allow_valid_speakers": allow_valid_speakers,
    }


def append_new_allow_speaker(
    paths: ProjectPaths,
    new_person_dir: Path | str,
    seed: int = 123,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.1,
) -> pd.DataFrame:
    rng = random.Random(seed)
    new_person_dir = Path(new_person_dir)

    wavs = sorted(str(p) for p in new_person_dir.rglob("*.wav"))
    if not wavs:
        raise ValueError(f"No wav files found in {new_person_dir}")

    if paths.recording_meta_path.exists():
        meta = pd.read_csv(paths.recording_meta_path)
    else:
        raise ValueError("No recording metadata found. Build the project first in the notebook.")

    speaker = new_person_dir.name
    meta = meta[meta["speaker"] != speaker].copy()

    train_wavs, valid_wavs, test_wavs = _split_files_within_speaker(
        wavs,
        rng=rng,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        min_valid=1 if len(wavs) >= 2 else 0,
        min_test=0,
    )

    new_rows = []
    for split, split_wavs in [("train", train_wavs), ("valid", valid_wavs), ("test", test_wavs)]:
        for audio_path in split_wavs:
            recording_id = f"{speaker}__{Path(audio_path).stem}"
            new_rows.append(
                {
                    "recording_id": recording_id,
                    "speaker": speaker,
                    "label": 1,
                    "class_name": "allow",
                    "split": split,
                    "audio_path": str(audio_path),
                }
            )

    updated = (
        pd.concat([meta, pd.DataFrame(new_rows)], ignore_index=True)
        .sort_values(["split", "label", "speaker", "audio_path"])
        .reset_index(drop=True)
    )
    updated.to_csv(paths.recording_meta_path, index=False)

    if paths.allow_speakers_path.exists():
        with open(paths.allow_speakers_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        current_allow = sorted(set(payload.get("allow_speakers", [])) | {speaker})
    else:
        current_allow = [speaker]

    with open(paths.allow_speakers_path, "w", encoding="utf-8") as f:
        json.dump({"allow_speakers": current_allow}, f, ensure_ascii=False, indent=2)

    return updated


# =========================
# Optional rebuild after adding person
# =========================
def segment_recordings(
    paths: ProjectPaths,
    segment_seconds: float = DEFAULT_SEGMENT_SECONDS,
    keep_remainder: bool = False,
    trim_silence: bool = DEFAULT_TRIM_SILENCE,
    trim_top_db: float = DEFAULT_TRIM_TOP_DB,
) -> pd.DataFrame:
    rec_meta = pd.read_csv(paths.recording_meta_path)
    rows = []

    for row in rec_meta.itertuples(index=False):
        y, sr = librosa.load(row.audio_path, sr=None, mono=True)
        y = y.astype(np.float32)

        original_seconds = len(y) / max(sr, 1)
        if trim_silence:
            y = trim_silence_waveform(y, top_db=trim_top_db)
        trimmed_seconds = len(y) / max(sr, 1)

        segment_len = int(round(segment_seconds * sr))

        if len(y) <= segment_len:
            seg = _pad_or_trim_segment(y, segment_len)
            seg_idx = 0
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
                    "segment_seconds": segment_seconds,
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
                if not keep_remainder:
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
                    "segment_seconds": segment_seconds,
                    "original_seconds": original_seconds,
                    "trimmed_seconds": trimmed_seconds,
                }
            )
            seg_idx += 1

    seg_meta = (
        pd.DataFrame(rows)
        .sort_values(["split", "label", "speaker", "recording_id", "segment_index"])
        .reset_index(drop=True)
    )
    seg_meta.to_csv(paths.segment_meta_path, index=False)
    return seg_meta


def build_spectrogram_arrays(
    paths: ProjectPaths,
    image_size: int = DEFAULT_IMAGE_SIZE,
    n_mels: int = DEFAULT_N_MELS,
) -> pd.DataFrame:
    seg_meta = pd.read_csv(paths.segment_meta_path)
    rows = []

    for row in seg_meta.itertuples(index=False):
        y, sr = librosa.load(row.segment_path, sr=None, mono=True)
        arr = mel_spectrogram_float32(y, sr=sr, image_size=image_size, n_mels=n_mels)

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

    spec_meta = (
        pd.DataFrame(rows)
        .sort_values(["split", "label", "speaker", "recording_id", "segment_index"])
        .reset_index(drop=True)
    )
    spec_meta.to_csv(paths.spectro_meta_path, index=False)
    return spec_meta


def rebuild_after_new_person(
    paths: ProjectPaths,
    segment_seconds: float = DEFAULT_SEGMENT_SECONDS,
    keep_remainder: bool = False,
    trim_silence: bool = DEFAULT_TRIM_SILENCE,
    trim_top_db: float = DEFAULT_TRIM_TOP_DB,
    image_size: int = DEFAULT_IMAGE_SIZE,
    n_mels: int = DEFAULT_N_MELS,
):
    segment_meta = segment_recordings(
        paths,
        segment_seconds=segment_seconds,
        keep_remainder=keep_remainder,
        trim_silence=trim_silence,
        trim_top_db=trim_top_db,
    )
    spectro_meta = build_spectrogram_arrays(paths, image_size=image_size, n_mels=n_mels)
    return segment_meta, spectro_meta


# =========================
# Streamlit helpers
# =========================
def list_checkpoints(paths: ProjectPaths) -> List[Path]:
    if not paths.model_dir.exists():
        return []
    return sorted(paths.model_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)


def load_allow_speakers(paths: ProjectPaths) -> List[str]:
    if not paths.allow_speakers_path.exists():
        return []
    with open(paths.allow_speakers_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return sorted(payload.get("allow_speakers", []))


def save_uploaded_files_to_speaker_dir(paths: ProjectPaths, speaker_name: str, uploaded_files) -> Path:
    speaker_name = speaker_name.strip()
    if not speaker_name:
        raise ValueError("Speaker name cannot be empty.")

    target_dir = paths.app_added_dir / speaker_name
    target_dir.mkdir(parents=True, exist_ok=True)

    kept = 0
    for uploaded in uploaded_files:
        suffix = Path(uploaded.name).suffix.lower()
        if suffix != ".wav":
            continue
        out_path = target_dir / Path(uploaded.name).name
        with open(out_path, "wb") as f:
            f.write(uploaded.getbuffer())
        kept += 1

    if kept == 0:
        raise ValueError("No .wav files were uploaded.")
    return target_dir


def render_metadata_summary(paths: ProjectPaths) -> None:
    if not paths.recording_meta_path.exists():
        st.warning("Brak metadata_recording_level.csv. Najpierw uruchom notebook bazowy.")
        return

    meta = pd.read_csv(paths.recording_meta_path)
    info = validate_recording_metadata(meta)

    c1, c2, c3 = st.columns(3)
    c1.metric("Nagrania", len(meta))
    c2.metric("Allow speakerzy", meta.loc[meta["label"] == 1, "speaker"].nunique())
    c3.metric("Allow speakerzy w valid", info["allow_valid_speakers"])

    st.write("Liczba nagrań per split i klasa")
    st.dataframe(info["summary"], use_container_width=True)

    st.write("Liczba unikalnych speakerów per split i klasa")
    st.dataframe(info["speakers"], use_container_width=True)

    if info["duplicate_audio_rows"] > 0 or info["same_audio_in_many_splits"] > 0:
        st.error(
            f"Problemy w metadanych: duplicate_audio_rows={info['duplicate_audio_rows']}, "
            f"same_audio_in_many_splits={info['same_audio_in_many_splits']}"
        )


# =========================
# App
# =========================
def main():
    st.set_page_config(page_title="Voice Allow App", layout="wide")
    st.title("Voice Allow App")
    st.caption("Testowa apka")

    with st.sidebar:
        work_dir_str = st.text_input("WORK_DIR", str(DEFAULT_WORK_DIR))
        device = st.selectbox("Device", [DEFAULT_DEVICE, "cpu", "cuda"] if DEFAULT_DEVICE == "cuda" else ["cpu", DEFAULT_DEVICE], index=0)
        paths = ProjectPaths.from_work_dir(work_dir_str)
        prepare_dirs(paths)
        checkpoints = list_checkpoints(paths)
        checkpoint_options = [str(p) for p in checkpoints]
        selected_checkpoint = st.selectbox("Checkpoint", checkpoint_options, index=0 if checkpoint_options else None)
        st.write(f"Allow speakerzy: {len(load_allow_speakers(paths))}")
        if selected_checkpoint:
            st.success(f"Aktywny checkpoint: {Path(selected_checkpoint).name}")
        else:
            st.warning("Nie znaleziono checkpointu .pt w work_dir/models")

    tab1, tab2, tab3 = st.tabs(["Test nagrania", "Dodaj czlowieka", "Stan projektu"])

    with tab1:
        st.subheader("Test pojedynczego nagrania")
        recorded_wav = st.audio_input("Nagraj glos mikrofonem", sample_rate=16000, key="predict_mic")

        if recorded_wav is not None:
            st.audio(recorded_wav)

        if st.button("Uruchom predykcje", disabled=(recorded_wav is None or not selected_checkpoint)):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(recorded_wav.getbuffer())
                    tmp_path = Path(tmp.name)

                pred = predict_wav_file(
                    wav_path=tmp_path,
                    checkpoint_path=selected_checkpoint,
                    device=device,
                )

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Klasa", pred["predicted_class_name"])
                c2.metric("P(allow)", f"{pred['recording_prob_allow']:.4f}")
                c3.metric("Threshold", f"{pred['threshold']:.4f}")
                c4.metric("Segmenty", pred["n_segments"])

                # st.write("Prawdopodobienstwa na segmentach")
                # seg_df = pd.DataFrame({
                #     "segment_index": np.arange(len(pred["segment_probs_allow"])),
                #     "p_allow": pred["segment_probs_allow"],
                # })
                # st.dataframe(seg_df, use_container_width=True)
                # st.line_chart(seg_df.set_index("segment_index"))

                decision_text = "ACCESS GRANTED" if pred["predicted_class_name"] == "allow" else "ACCESS DENIED"
                if pred["predicted_class_name"] == "allow":
                    st.success(decision_text)
                else:
                    st.error(decision_text)

                st.info(
                    f"Agregacja: {pred['aggregation_method']}. "
                    f"Model podejmuje decyzje na poziomie calego nagrania, nie pojedynczego segmentu."
                )
            except Exception as e:
                st.exception(e)

    with tab2:
        st.subheader("Dodanie nowego allowed speakera")
        st.warning(
            "Ta operacja doda nowa osobe do datasetu "
            )

        speaker_name = st.text_input("Nazwa speakera", key="speaker_name")
        uploaded_speaker_wavs = st.file_uploader(
            "Wgraj kilka nagran .wav tej osoby",
            type=["wav"],
            accept_multiple_files=True,
            key="speaker_wavs",
        )
        rebuild_now = st.checkbox("Przetrenuj", value=True)

        if st.button("Dodaj osobe do datasetu", disabled=(not speaker_name or not uploaded_speaker_wavs)):
            try:
                speaker_dir = save_uploaded_files_to_speaker_dir(paths, speaker_name, uploaded_speaker_wavs)
                updated_meta = append_new_allow_speaker(paths=paths, new_person_dir=speaker_dir)

                st.success(f"Dodano/odswiezono speakera: {speaker_name}")
                st.write(f"Zapisano pliki do: {speaker_dir}")

                if rebuild_now:
                    with st.spinner("Przebudowuje segmenty i spektrogramy..."):
                        seg_meta, spec_meta = rebuild_after_new_person(paths=paths)
                    st.success(
                        f"Przebudowa zakonczona. Segmenty: {len(seg_meta)}, spektrogramy: {len(spec_meta)}"
                    )

                meta_info = validate_recording_metadata(updated_meta)
                st.write("Nowy podzial zbioru")
                st.dataframe(meta_info["summary"], use_container_width=True)
                st.write("Unikalni speakerzy")
                st.dataframe(meta_info["speakers"], use_container_width=True)

                st.info(
                    "Kolejny krok: uruchom w notebooku komorki od loaderow i treningu, zeby zapisac nowy checkpoint. "
                    "Bez retrainingu aktualny model nie nauczy sie nowej osoby."
                )
            except Exception as e:
                st.exception(e)

    with tab3:
        st.subheader("Stan projektu")
        render_metadata_summary(paths)

        allow_speakers = load_allow_speakers(paths)
        if allow_speakers:
            st.write("Aktualni allow speakerzy")
            st.dataframe(pd.DataFrame({"speaker": allow_speakers}), use_container_width=True)

        if checkpoints:
            st.write("Dostepne checkpointy")
            ckpt_df = pd.DataFrame(
                {
                    "checkpoint": [p.name for p in checkpoints],
                    "path": [str(p) for p in checkpoints],
                    "modified": [pd.Timestamp(p.stat().st_mtime, unit="s") for p in checkpoints],
                }
            )
            st.dataframe(ckpt_df, use_container_width=True)

        st.code("streamlit run streamlit_voice_allow_app.py")


if __name__ == "__main__":
    main()
