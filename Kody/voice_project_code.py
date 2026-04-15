
from __future__ import annotations

from pathlib import Path
import json
import math
import random
import shutil
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


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
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_dirs(paths: ProjectPaths, clean_intermediate: bool = False) -> None:
    paths.work_dir.mkdir(parents=True, exist_ok=True)
    paths.model_dir.mkdir(parents=True, exist_ok=True)
    paths.results_dir.mkdir(parents=True, exist_ok=True)

    if clean_intermediate:
        for path in [paths.segment_dir, paths.spectro_dir]:
            if path.exists():
                shutil.rmtree(path)

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
    if n == 3:
        return [speakers[0]], [speakers[1]], [speakers[2]]

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

    train = speakers[:n_train]
    valid = speakers[n_train:n_train + n_valid]
    test = speakers[n_train + n_valid:]
    return train, valid, test


def load_speaker_to_wavs(source_dir: Path | str) -> Dict[str, List[str]]:
    source_dir = Path(source_dir)
    speaker_to_wavs: Dict[str, List[str]] = {}

    for speaker_dir in sorted([p for p in source_dir.iterdir() if p.is_dir()]):
        wavs = sorted(str(p) for p in speaker_dir.rglob("*.wav"))
        if wavs:
            speaker_to_wavs[speaker_dir.name] = wavs

    return speaker_to_wavs


def build_recording_metadata(
    source_dir: Path | str,
    paths: ProjectPaths,
    allow_speakers: Optional[Sequence[str]] = None,
    n_allow_speakers: int = 5,
    seed: int = 123,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.2,
) -> pd.DataFrame:
    if not math.isclose(train_ratio + valid_ratio + test_ratio, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("train_ratio + valid_ratio + test_ratio must sum to 1.")

    rng = random.Random(seed)
    speaker_to_wavs = load_speaker_to_wavs(source_dir)
    all_speakers = sorted(speaker_to_wavs.keys())

    if allow_speakers is None:
        allow_speakers = all_speakers[:n_allow_speakers]
    else:
        allow_speakers = list(allow_speakers)

    missing = sorted(set(allow_speakers) - set(all_speakers))
    if missing:
        raise ValueError(f"Missing allow speakers in dataset: {missing}")

    rows: List[dict] = []

    # class 1: keep each allow speaker in every split whenever possible
    for speaker in allow_speakers:
        train_wavs, valid_wavs, test_wavs = _split_files_within_speaker(
            speaker_to_wavs[speaker],
            rng=rng,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
        )

        for split, wavs in [("train", train_wavs), ("valid", valid_wavs), ("test", test_wavs)]:
            for wav in wavs:
                rows.append(
                    {
                        "speaker": speaker,
                        "label": 1,
                        "split": split,
                        "audio_path": wav,
                    }
                )

    # class 0: speaker-disjoint split to avoid speaker leakage
    not_allow_speakers = [speaker for speaker in all_speakers if speaker not in allow_speakers]
    train_neg_speakers, valid_neg_speakers, test_neg_speakers = _split_speakers(
        not_allow_speakers,
        rng=rng,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
    )

    for split, speakers in [
        ("train", train_neg_speakers),
        ("valid", valid_neg_speakers),
        ("test", test_neg_speakers),
    ]:
        for speaker in speakers:
            for wav in speaker_to_wavs[speaker]:
                rows.append(
                    {
                        "speaker": speaker,
                        "label": 0,
                        "split": split,
                        "audio_path": wav,
                    }
                )

    meta = pd.DataFrame(rows).sort_values(["split", "label", "speaker", "audio_path"]).reset_index(drop=True)
    meta.to_csv(paths.recording_meta_path, index=False)

    with open(paths.allow_speakers_path, "w", encoding="utf-8") as f:
        json.dump({"allow_speakers": list(allow_speakers)}, f, indent=2)

    return meta


def validate_recording_metadata(meta: pd.DataFrame) -> dict:
    if meta.empty:
        raise ValueError("Recording metadata is empty.")

    audio_split_counts = meta.groupby("audio_path")["split"].nunique()
    if (audio_split_counts > 1).any():
        raise AssertionError("The same source recording appears in more than one split.")

    neg = meta[meta["label"] == 0]
    neg_sets = {
        split: set(df["speaker"].unique())
        for split, df in neg.groupby("split")
    }
    neg_overlap = {
        "train_valid": sorted(neg_sets.get("train", set()) & neg_sets.get("valid", set())),
        "train_test": sorted(neg_sets.get("train", set()) & neg_sets.get("test", set())),
        "valid_test": sorted(neg_sets.get("valid", set()) & neg_sets.get("test", set())),
    }
    if any(neg_overlap.values()):
        raise AssertionError(f"Negative speakers overlap across splits: {neg_overlap}")

    allow = meta[meta["label"] == 1]
    summary = {
        "n_recordings": int(len(meta)),
        "split_label_counts": meta.groupby(["split", "label"]).size().to_dict(),
        "allow_speakers_per_split": allow.groupby("split")["speaker"].nunique().to_dict(),
        "not_allow_speakers_per_split": neg.groupby("split")["speaker"].nunique().to_dict(),
    }
    return summary


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
                    "segment_index": seg_idx,
                    "start_sample": start,
                    "end_sample": end,
                    "start_sec": start / sr,
                    "end_sec": end / sr,
                    "duration_sec": len(seg) / sr,
                    "sample_rate": sr,
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
                    "segment_index": n_full_segments,
                    "start_sample": start,
                    "end_sample": end,
                    "start_sec": start / sr,
                    "end_sec": end / sr,
                    "duration_sec": len(seg) / sr,
                    "sample_rate": sr,
                }
            )

    segments_meta = pd.DataFrame(rows)
    segments_meta.to_csv(paths.segment_meta_path, index=False)
    return segments_meta


def validate_segment_metadata(segments_meta: pd.DataFrame) -> dict:
    if segments_meta.empty:
        raise ValueError("Segment metadata is empty.")

    source_split_counts = segments_meta.groupby("source_audio_path")["split"].nunique()
    if (source_split_counts > 1).any():
        raise AssertionError("Segments from the same recording appear in more than one split.")

    return {
        "n_segments": int(len(segments_meta)),
        "split_label_counts": segments_meta.groupby(["split", "label"]).size().to_dict(),
        "segments_per_source_recording_head": (
            segments_meta.groupby("source_audio_path").size().sort_values(ascending=False).head(10).to_dict()
        ),
    }


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

    denom = S_db.max() - S_db.min()
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


class SpectrogramPngDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.df = dataframe.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("L")
        label = int(row["label"])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def compute_train_mean_std(train_df: pd.DataFrame, image_size: int = 128, batch_size: int = 64) -> Tuple[float, float]:
    dataset = SpectrogramPngDataset(
        train_df,
        transform=transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        ),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    mean_sum = 0.0
    std_sum = 0.0
    n_batches = 0

    for x, _ in loader:
        mean_sum += x.mean().item()
        std_sum += x.std().item()
        n_batches += 1

    train_mean = mean_sum / max(1, n_batches)
    train_std = std_sum / max(1, n_batches)
    return train_mean, train_std


def build_transforms(
    train_mean: float,
    train_std: float,
    image_size: int = 128,
    normalize: bool = True,
    augment: bool = False,
):
    train_ops = [transforms.Resize((image_size, image_size))]
    eval_ops = [transforms.Resize((image_size, image_size))]

    if augment:
        train_ops.extend(
            [
                transforms.RandomAffine(degrees=0, translate=(0.03, 0.03)),
            ]
        )

    train_ops.append(transforms.ToTensor())
    eval_ops.append(transforms.ToTensor())

    if augment:
        train_ops.append(transforms.RandomErasing(p=0.15, scale=(0.02, 0.08), ratio=(0.3, 3.3)))

    if normalize:
        norm = transforms.Normalize(mean=[train_mean], std=[max(train_std, 1e-8)])
        train_ops.append(norm)
        eval_ops.append(norm)

    return transforms.Compose(train_ops), transforms.Compose(eval_ops)


def make_loaders(
    paths: ProjectPaths,
    image_size: int = 128,
    batch_size: int = 32,
    normalize: bool = True,
    augment: bool = False,
    num_workers: int = 0,
):
    meta = pd.read_csv(paths.spectro_meta_path)
    train_df = meta[meta["split"] == "train"].copy()
    valid_df = meta[meta["split"] == "valid"].copy()
    test_df = meta[meta["split"] == "test"].copy()

    train_mean, train_std = compute_train_mean_std(train_df, image_size=image_size, batch_size=batch_size)

    train_tf, eval_tf = build_transforms(
        train_mean=train_mean,
        train_std=train_std,
        image_size=image_size,
        normalize=normalize,
        augment=augment,
    )

    train_ds = SpectrogramPngDataset(train_df, transform=train_tf)
    valid_ds = SpectrogramPngDataset(valid_df, transform=eval_tf)
    test_ds = SpectrogramPngDataset(test_df, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    stats = {
        "train_mean": train_mean,
        "train_std": train_std,
        "class_counts_train": train_df["label"].value_counts().sort_index().to_dict(),
        "n_train": len(train_df),
        "n_valid": len(valid_df),
        "n_test": len(test_df),
    }

    return train_loader, valid_loader, test_loader, stats


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


def build_resnet18_from_scratch(num_classes: int = 2) -> nn.Module:
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_resnet18_transfer(num_classes: int = 2) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    pretrained_weight = model.conv1.weight.data.mean(dim=1, keepdim=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.conv1.weight.data.copy_(pretrained_weight)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_model(
    model_name: str,
    dropout: float = 0.3,
    use_batchnorm: bool = True,
    num_classes: int = 2,
) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "smallcnn":
        return SmallCNN(dropout=dropout, use_batchnorm=use_batchnorm, num_classes=num_classes)
    if model_name == "resnet18_scratch":
        return build_resnet18_from_scratch(num_classes=num_classes)
    if model_name == "resnet18_transfer":
        return build_resnet18_transfer(num_classes=num_classes)

    raise ValueError(f"Unknown model_name={model_name}")


def evaluate_binary_classifier(model: nn.Module, loader: DataLoader, device: str) -> dict:
    model.eval()

    total = 0
    correct = 0

    false_accept = 0
    false_reject = 0
    n_not_allow = 0
    n_allow = 0

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            probs_allow = torch.softmax(logits, dim=1)[:, 1]
            pred = logits.argmax(dim=1)

            total += y.size(0)
            correct += (pred == y).sum().item()

            n_not_allow += (y == 0).sum().item()
            n_allow += (y == 1).sum().item()

            false_accept += ((y == 0) & (pred == 1)).sum().item()
            false_reject += ((y == 1) & (pred == 0)).sum().item()

            all_probs.extend(probs_allow.detach().cpu().tolist())
            all_targets.extend(y.detach().cpu().tolist())

    acc = correct / total if total > 0 else 0.0
    far = false_accept / n_not_allow if n_not_allow > 0 else 0.0
    frr = false_reject / n_allow if n_allow > 0 else 0.0

    return {
        "acc": acc,
        "far": far,
        "frr": frr,
        "n_total": total,
        "n_not_allow": n_not_allow,
        "n_allow": n_allow,
        "probs_allow": all_probs,
        "targets": all_targets,
    }


def _make_optimizer(model: nn.Module, optimizer_name: str, lr: float, weight_decay: float):
    optimizer_name = optimizer_name.lower()

    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    raise ValueError(f"Unknown optimizer_name={optimizer_name}")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: str,
    epochs: int = 10,
    optimizer_name: str = "adam",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    checkpoint_path: Optional[Path] = None,
) -> pd.DataFrame:
    model = model.to(device)

    train_targets = []
    for _, y in train_loader:
        train_targets.extend(y.tolist())

    class_counts = torch.bincount(torch.tensor(train_targets, dtype=torch.long), minlength=2).float()
    class_weights = class_counts.sum() / (2.0 * torch.clamp(class_counts, min=1.0))
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = _make_optimizer(model, optimizer_name=optimizer_name, lr=lr, weight_decay=weight_decay)

    history_rows: List[dict] = []
    best_score = float("inf")

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

            loss_sum += loss.item() * x.size(0)
            n_seen += x.size(0)

        train_loss = loss_sum / max(1, n_seen)
        train_metrics = evaluate_binary_classifier(model, train_loader, device=device)
        valid_metrics = evaluate_binary_classifier(model, valid_loader, device=device)

        score = valid_metrics["far"] + valid_metrics["frr"]
        if score < best_score:
            best_score = score
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
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

    return pd.DataFrame(history_rows)


def run_experiment(
    paths: ProjectPaths,
    experiment_name: str,
    model_name: str = "smallcnn",
    dropout: float = 0.3,
    use_batchnorm: bool = True,
    optimizer_name: str = "adam",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 10,
    batch_size: int = 32,
    image_size: int = 128,
    normalize: bool = True,
    augment: bool = False,
    device: str = "cpu",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_loader, valid_loader, test_loader, stats = make_loaders(
        paths=paths,
        image_size=image_size,
        batch_size=batch_size,
        normalize=normalize,
        augment=augment,
    )

    model = build_model(
        model_name=model_name,
        dropout=dropout,
        use_batchnorm=use_batchnorm,
        num_classes=2,
    )

    checkpoint_path = paths.model_dir / f"{experiment_name}.pt"
    history = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        epochs=epochs,
        optimizer_name=optimizer_name,
        lr=lr,
        weight_decay=weight_decay,
        checkpoint_path=checkpoint_path,
    )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)

    train_metrics = evaluate_binary_classifier(model, train_loader, device=device)
    valid_metrics = evaluate_binary_classifier(model, valid_loader, device=device)
    test_metrics = evaluate_binary_classifier(model, test_loader, device=device)

    results = pd.DataFrame(
        [
            {
                "experiment_name": experiment_name,
                "model_name": model_name,
                "optimizer_name": optimizer_name,
                "dropout": dropout,
                "use_batchnorm": use_batchnorm,
                "normalize": normalize,
                "augment": augment,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
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
                "n_train": stats["n_train"],
                "n_valid": stats["n_valid"],
                "n_test": stats["n_test"],
            }
        ]
    )

    history_path = paths.results_dir / f"{experiment_name}_history.csv"
    results_path = paths.results_dir / f"{experiment_name}_results.csv"
    history.to_csv(history_path, index=False)
    results.to_csv(results_path, index=False)

    return history, results


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


def rebuild_after_new_person(
    paths: ProjectPaths,
    segment_seconds: float = 3.0,
    keep_remainder: bool = False,
    trim_silence: bool = False,
    top_db: float = 30.0,
    image_size: int = 128,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 128,
    fmin: int = 20,
    fmax: int = 8000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    segments_meta = segment_recordings(
        paths=paths,
        segment_seconds=segment_seconds,
        keep_remainder=keep_remainder,
        trim_silence=trim_silence,
        top_db=top_db,
        clean_output=True,
    )
    spectro_meta = build_spectrogram_pngs(
        paths=paths,
        image_size=image_size,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        clean_output=True,
    )
    return segments_meta, spectro_meta


def load_model_for_inference(
    checkpoint_path: Path | str,
    model_name: str,
    device: str = "cpu",
    dropout: float = 0.3,
    use_batchnorm: bool = True,
) -> nn.Module:
    model = build_model(model_name=model_name, dropout=dropout, use_batchnorm=use_batchnorm, num_classes=2)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def predict_wav_file(
    wav_path: Path | str,
    checkpoint_path: Path | str,
    model_name: str,
    train_mean: float,
    train_std: float,
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
        pad = segment_len - len(y)
        y = np.pad(y, (0, pad))

    segments = []
    for start in range(0, len(y), segment_len):
        end = start + segment_len
        seg = y[start:end]
        if len(seg) < segment_len:
            break
        segments.append(seg)

    model = load_model_for_inference(
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        device=device,
        dropout=dropout,
        use_batchnorm=use_batchnorm,
    )

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[train_mean], std=[max(train_std, 1e-8)]),
        ]
    )

    probs = []
    with torch.no_grad():
        for seg in segments:
            img = _mel_to_png_array(seg, sr=sr, image_size=image_size)
            pil_img = Image.fromarray(img, mode="L")
            x = transform(pil_img).unsqueeze(0).to(device)
            logits = model(x)
            prob_allow = torch.softmax(logits, dim=1)[0, 1].item()
            probs.append(prob_allow)

    mean_prob_allow = float(np.mean(probs)) if probs else 0.0
    predicted_label = int(mean_prob_allow >= 0.5)

    return {
        "wav_path": str(wav_path),
        "n_segments_used": len(probs),
        "segment_probs_allow": probs,
        "mean_prob_allow": mean_prob_allow,
        "predicted_label": predicted_label,
        "predicted_class_name": "allow" if predicted_label == 1 else "not_allow",
    }
