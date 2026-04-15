from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import json
import random
import shutil

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset


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
    app_dir: Path
    app_allowed_dir: Path
    app_tmp_dir: Path

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
            spectro_dir=work_dir / "spectrograms_png",
            model_dir=work_dir / "models",
            results_dir=work_dir / "results",
            app_dir=work_dir / "app_data",
            app_allowed_dir=work_dir / "app_data" / "new_allowed",
            app_tmp_dir=work_dir / "app_data" / "tmp",
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def prepare_dirs(paths: ProjectPaths, clean_intermediate: bool = False) -> None:
    for d in [
        paths.work_dir,
        paths.model_dir,
        paths.results_dir,
        paths.app_dir,
        paths.app_allowed_dir,
        paths.app_tmp_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    if clean_intermediate:
        for d in [paths.segment_dir, paths.spectro_dir]:
            if d.exists():
                shutil.rmtree(d)

    paths.segment_dir.mkdir(parents=True, exist_ok=True)
    paths.spectro_dir.mkdir(parents=True, exist_ok=True)



def _split_files_within_speaker(
    wavs: Sequence[str],
    rng: random.Random,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.1,
) -> Tuple[List[str], List[str], List[str]]:
    wavs = list(wavs)
    rng.shuffle(wavs)
    n = len(wavs)

    if n == 0:
        return [], [], []
    if n == 1:
        return wavs, [], []
    if n == 2:
        return [wavs[0]], [], [wavs[1]]

    n_train = max(1, int(round(train_ratio * n)))
    n_valid = max(1, int(round(valid_ratio * n)))

    if n_train + n_valid >= n:
        n_train = max(1, n - 2)
        n_valid = 1

    n_test = n - n_train - n_valid
    if n_test <= 0:
        n_test = 1
        if n_train > 1:
            n_train -= 1
        else:
            n_valid = max(0, n_valid - 1)

    train = wavs[:n_train]
    valid = wavs[n_train:n_train + n_valid]
    test = wavs[n_train + n_valid:]
    return train, valid, test



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
        meta = pd.DataFrame(columns=["speaker", "label", "split", "audio_path"])

    speaker = new_person_dir.name
    meta = meta[meta["speaker"] != speaker].copy()

    train_wavs, valid_wavs, test_wavs = _split_files_within_speaker(
        wavs,
        rng=rng,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
    )

    new_rows = []
    for split, split_wavs in [("train", train_wavs), ("valid", valid_wavs), ("test", test_wavs)]:
        for wav in split_wavs:
            new_rows.append(
                {
                    "speaker": speaker,
                    "label": 1,
                    "split": split,
                    "audio_path": wav,
                }
            )

    meta = pd.concat([meta, pd.DataFrame(new_rows)], ignore_index=True)
    meta = meta.sort_values(["split", "label", "speaker", "audio_path"]).reset_index(drop=True)
    meta.to_csv(paths.recording_meta_path, index=False)

    allow_speakers = []
    if paths.allow_speakers_path.exists():
        with open(paths.allow_speakers_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
            allow_speakers = list(payload.get("allow_speakers", []))

    if speaker not in allow_speakers:
        allow_speakers.append(speaker)

    with open(paths.allow_speakers_path, "w", encoding="utf-8") as f:
        json.dump({"allow_speakers": sorted(allow_speakers)}, f, indent=2)

    return meta



def _trim_if_needed(y: np.ndarray, top_db: float) -> np.ndarray:
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    if len(y_trimmed) == 0:
        return y
    return y_trimmed



def segment_recordings(
    paths: ProjectPaths,
    segment_seconds: float = 3.0,
    keep_remainder: bool = False,
    trim_silence: bool = False,
    top_db: float = 30.0,
    clean_output: bool = True,
) -> pd.DataFrame:
    if clean_output and paths.segment_dir.exists():
        shutil.rmtree(paths.segment_dir)
    paths.segment_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(paths.recording_meta_path)
    rows: List[dict] = []

    for row in meta.itertuples(index=False):
        audio_path = Path(row.audio_path)
        y, sr = librosa.load(audio_path, sr=None, mono=True)

        if trim_silence:
            y = _trim_if_needed(y, top_db=top_db)

        segment_len = int(round(segment_seconds * sr))
        if segment_len <= 0:
            raise ValueError("segment_seconds must be positive.")

        n_samples = len(y)
        n_full_segments = n_samples // segment_len

        out_dir = paths.segment_dir / row.split / f"label_{int(row.label)}" / row.speaker / audio_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        for seg_idx in range(n_full_segments):
            start = seg_idx * segment_len
            end = start + segment_len
            seg = y[start:end]
            out_path = out_dir / f"{audio_path.stem}_seg{seg_idx:04d}.wav"
            sf.write(out_path, seg, sr)
            rows.append(
                {
                    "speaker": row.speaker,
                    "label": int(row.label),
                    "split": row.split,
                    "source_audio_path": str(audio_path),
                    "segment_path": str(out_path),
                }
            )

        remainder = n_samples - n_full_segments * segment_len
        if keep_remainder and remainder > 0:
            start = n_full_segments * segment_len
            end = n_samples
            seg = y[start:end]
            out_path = out_dir / f"{audio_path.stem}_seg{n_full_segments:04d}.wav"
            sf.write(out_path, seg, sr)
            rows.append(
                {
                    "speaker": row.speaker,
                    "label": int(row.label),
                    "split": row.split,
                    "source_audio_path": str(audio_path),
                    "segment_path": str(out_path),
                }
            )

    segments_meta = pd.DataFrame(rows)
    segments_meta.to_csv(paths.segment_meta_path, index=False)
    return segments_meta



def _mel_to_png_array(
    y: np.ndarray,
    sr: int,
    image_size: int = 128,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 128,
    fmin: int = 20,
    fmax: int = 8000,
) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    S_db = np.flipud(S_db)

    denom = float(S_db.max() - S_db.min())
    if denom < 1e-8:
        img = np.zeros_like(S_db, dtype=np.uint8)
    else:
        img = ((S_db - S_db.min()) / denom * 255.0).clip(0, 255).astype(np.uint8)

    pil_img = Image.fromarray(img, mode="L").resize((image_size, image_size))
    return np.asarray(pil_img, dtype=np.uint8)



def build_spectrogram_pngs(
    paths: ProjectPaths,
    image_size: int = 128,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 128,
    fmin: int = 20,
    fmax: int = 8000,
    clean_output: bool = True,
) -> pd.DataFrame:
    if clean_output and paths.spectro_dir.exists():
        shutil.rmtree(paths.spectro_dir)
    paths.spectro_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(paths.segment_meta_path)
    rows: List[dict] = []

    for row in meta.itertuples(index=False):
        segment_path = Path(row.segment_path)
        class_name = "allow" if int(row.label) == 1 else "not_allow"
        out_dir = paths.spectro_dir / row.split / class_name / row.speaker
        out_dir.mkdir(parents=True, exist_ok=True)

        y, sr = librosa.load(segment_path, sr=None, mono=True)
        img = _mel_to_png_array(
            y=y,
            sr=sr,
            image_size=image_size,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
        )

        out_path = out_dir / f"{segment_path.stem}.png"
        Image.fromarray(img, mode="L").save(out_path)

        rows.append(
            {
                "speaker": row.speaker,
                "label": int(row.label),
                "split": row.split,
                "source_audio_path": row.source_audio_path,
                "segment_path": row.segment_path,
                "image_path": str(out_path),
            }
        )

    spectro_meta = pd.DataFrame(rows)
    spectro_meta.to_csv(paths.spectro_meta_path, index=False)
    return spectro_meta


class SpectrogramDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, image_size: int = 128, mean: Optional[float] = None, std: Optional[float] = None):
        self.df = dataframe.reset_index(drop=True).copy()
        self.image_size = image_size
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("L").resize((self.image_size, self.image_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).unsqueeze(0)
        if self.mean is not None and self.std is not None:
            x = (x - float(self.mean)) / max(float(self.std), 1e-8)
        y = int(row["label"])
        return x, y



def compute_train_mean_std(train_df: pd.DataFrame, image_size: int = 128, batch_size: int = 64) -> Tuple[float, float]:
    ds = SpectrogramDataset(train_df, image_size=image_size, mean=None, std=None)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    mean_sum = 0.0
    std_sum = 0.0
    n_batches = 0
    for x, _ in loader:
        mean_sum += float(x.mean().item())
        std_sum += float(x.std().item())
        n_batches += 1
    return mean_sum / max(1, n_batches), std_sum / max(1, n_batches)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.extend([nn.ReLU(inplace=True), nn.MaxPool2d(2)])
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SmallCNN(nn.Module):
    def __init__(self, dropout: float = 0.3, use_batchnorm: bool = True, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 16, use_batchnorm=use_batchnorm),
            ConvBlock(16, 32, use_batchnorm=use_batchnorm),
            ConvBlock(32, 64, use_batchnorm=use_batchnorm),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x



def make_loaders(paths: ProjectPaths, image_size: int = 128, batch_size: int = 32, num_workers: int = 0):
    meta = pd.read_csv(paths.spectro_meta_path)
    train_df = meta[meta["split"] == "train"].copy()
    valid_df = meta[meta["split"] == "valid"].copy()
    test_df = meta[meta["split"] == "test"].copy()

    train_mean, train_std = compute_train_mean_std(train_df, image_size=image_size, batch_size=batch_size)

    train_ds = SpectrogramDataset(train_df, image_size=image_size, mean=train_mean, std=train_std)
    valid_ds = SpectrogramDataset(valid_df, image_size=image_size, mean=train_mean, std=train_std)
    test_ds = SpectrogramDataset(test_df, image_size=image_size, mean=train_mean, std=train_std)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    stats = {
        "train_mean": train_mean,
        "train_std": train_std,
        "n_train": int(len(train_df)),
        "n_valid": int(len(valid_df)),
        "n_test": int(len(test_df)),
    }
    return train_loader, valid_loader, test_loader, stats



def evaluate_binary_classifier(model: nn.Module, loader: DataLoader, device: str) -> dict:
    model.eval()
    total = 0
    correct = 0
    false_accept = 0
    false_reject = 0
    n_not_allow = 0
    n_allow = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            total += int(y.size(0))
            correct += int((pred == y).sum().item())
            n_not_allow += int((y == 0).sum().item())
            n_allow += int((y == 1).sum().item())
            false_accept += int(((y == 0) & (pred == 1)).sum().item())
            false_reject += int(((y == 1) & (pred == 0)).sum().item())
    acc = correct / total if total > 0 else 0.0
    far = false_accept / n_not_allow if n_not_allow > 0 else 0.0
    frr = false_reject / n_allow if n_allow > 0 else 0.0
    return {"acc": acc, "far": far, "frr": frr, "n_total": total, "n_not_allow": n_not_allow, "n_allow": n_allow}



def train_smallcnn(
    paths: ProjectPaths,
    experiment_name: str = "app_smallcnn",
    image_size: int = 128,
    batch_size: int = 32,
    epochs: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.3,
    use_batchnorm: bool = True,
    device: str = "cpu",
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train_loader, valid_loader, test_loader, stats = make_loaders(paths, image_size=image_size, batch_size=batch_size)
    model = SmallCNN(dropout=dropout, use_batchnorm=use_batchnorm, num_classes=2).to(device)

    train_targets = []
    for _, y in train_loader:
        train_targets.extend(y.tolist())
    class_counts = torch.bincount(torch.tensor(train_targets, dtype=torch.long), minlength=2).float()
    class_weights = class_counts.sum() / (2.0 * torch.clamp(class_counts, min=1.0))
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history_rows = []
    best_score = float("inf")
    checkpoint_path = paths.model_dir / f"{experiment_name}.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        n_seen = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * int(x.size(0))
            n_seen += int(x.size(0))

        train_loss = loss_sum / max(1, n_seen)
        train_metrics = evaluate_binary_classifier(model, train_loader, device=device)
        valid_metrics = evaluate_binary_classifier(model, valid_loader, device=device)
        score = valid_metrics["far"] + valid_metrics["frr"]
        if score < best_score:
            best_score = score
            torch.save(model.state_dict(), checkpoint_path)

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_metrics["acc"],
                "train_far": train_metrics["far"],
                "train_frr": train_metrics["frr"],
                "valid_acc": valid_metrics["acc"],
                "valid_far": valid_metrics["far"],
                "valid_frr": valid_metrics["frr"],
                "valid_score_far_plus_frr": score,
            }
        )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    train_metrics = evaluate_binary_classifier(model, train_loader, device=device)
    valid_metrics = evaluate_binary_classifier(model, valid_loader, device=device)
    test_metrics = evaluate_binary_classifier(model, test_loader, device=device)

    history = pd.DataFrame(history_rows)
    results = pd.DataFrame([
        {
            "experiment_name": experiment_name,
            "model_name": "smallcnn",
            "train_acc": train_metrics["acc"],
            "train_far": train_metrics["far"],
            "train_frr": train_metrics["frr"],
            "valid_acc": valid_metrics["acc"],
            "valid_far": valid_metrics["far"],
            "valid_frr": valid_metrics["frr"],
            "test_acc": test_metrics["acc"],
            "test_far": test_metrics["far"],
            "test_frr": test_metrics["frr"],
            "train_mean": stats["train_mean"],
            "train_std": stats["train_std"],
            "image_size": image_size,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "dropout": dropout,
            "use_batchnorm": int(use_batchnorm),
        }
    ])

    history.to_csv(paths.results_dir / f"{experiment_name}_history.csv", index=False)
    results.to_csv(paths.results_dir / f"{experiment_name}_results.csv", index=False)

    inference_config = {
        "checkpoint_path": str(checkpoint_path),
        "model_name": "smallcnn",
        "train_mean": float(stats["train_mean"]),
        "train_std": float(stats["train_std"]),
        "image_size": int(image_size),
        "segment_seconds": 3.0,
        "dropout": float(dropout),
        "use_batchnorm": bool(use_batchnorm),
    }
    with open(paths.model_dir / f"{experiment_name}_inference_config.json", "w", encoding="utf-8") as f:
        json.dump(inference_config, f, indent=2)

    return history, results, inference_config



def load_smallcnn_for_inference(checkpoint_path: Path | str, dropout: float = 0.3, use_batchnorm: bool = True, device: str = "cpu") -> nn.Module:
    model = SmallCNN(dropout=dropout, use_batchnorm=use_batchnorm, num_classes=2)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model



def predict_wav_file(
    wav_path: Path | str,
    checkpoint_path: Path | str,
    train_mean: float,
    train_std: float,
    threshold: float = 0.5,
    segment_seconds: float = 3.0,
    image_size: int = 128,
    device: str = "cpu",
    dropout: float = 0.3,
    use_batchnorm: bool = True,
) -> dict:
    wav_path = Path(wav_path)
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    segment_len = int(round(segment_seconds * sr))

    if len(y) < segment_len:
        y = np.pad(y, (0, segment_len - len(y)))

    segments = []
    for start in range(0, len(y), segment_len):
        end = start + segment_len
        seg = y[start:end]
        if len(seg) < segment_len:
            break
        segments.append(seg)

    model = load_smallcnn_for_inference(checkpoint_path=checkpoint_path, dropout=dropout, use_batchnorm=use_batchnorm, device=device)
    probs = []
    with torch.no_grad():
        for seg in segments:
            img = _mel_to_png_array(seg, sr=sr, image_size=image_size)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
            x = (x - float(train_mean)) / max(float(train_std), 1e-8)
            x = x.to(device)
            logits = model(x)
            prob_allow = float(torch.softmax(logits, dim=1)[0, 1].item())
            probs.append(prob_allow)

    mean_prob_allow = float(np.mean(probs)) if probs else 0.0
    predicted_label = int(mean_prob_allow >= threshold)
    return {
        "wav_path": str(wav_path),
        "n_segments_used": len(probs),
        "segment_probs_allow": probs,
        "mean_prob_allow": mean_prob_allow,
        "threshold": float(threshold),
        "predicted_label": predicted_label,
        "predicted_class_name": "allow" if predicted_label == 1 else "not_allow",
    }



def save_uploaded_audio(uploaded_file, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    return out_path
