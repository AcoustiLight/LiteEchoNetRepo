"""ESC-50 metadata validation, path resolution, and waveform loading."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

EXPECTED_SAMPLE_COUNT = 2000
EXPECTED_SAMPLES_PER_FOLD = 400
NUM_CLASSES = 50
FOLDS = tuple(range(1, 6))
TARGET_SAMPLE_RATE = 32000
CLIP_DURATION_SECONDS = 5
TARGET_NUM_SAMPLES = TARGET_SAMPLE_RATE * CLIP_DURATION_SECONDS


def find_esc50_csv(root):
    root = Path(root)
    candidates = [
        root / "esc50.csv",
        root / "meta" / "esc50.csv",
        root / "ESC-50-master" / "meta" / "esc50.csv",
        root / "ESC-50-master" / "esc50.csv",
    ]
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _resolve_audio_path(data_root, csv_path, filename, fold):
    data_root = Path(data_root)
    csv_path = Path(csv_path)
    candidates = [
        data_root / f"audio-fold-{fold}" / "audio" / filename,
        data_root / f"audio-fold-{fold}" / filename,
        data_root / "audio" / filename,
        data_root / "ESC-50-master" / "audio" / filename,
        csv_path.parent / "audio" / filename,
        csv_path.parent.parent / "audio" / filename,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return str(candidates[0])


def load_esc50_metadata(data_root, validate_files=True):
    """Load and strictly validate the ESC-50 metadata used by the notebook."""
    data_root = Path(data_root)
    csv_path = find_esc50_csv(data_root)
    if csv_path is None:
        raise FileNotFoundError(f"Could not find esc50.csv under {data_root}")

    metadata = pd.read_csv(csv_path)
    metadata.columns = [str(column).strip().lower() for column in metadata.columns]

    required_columns = {"filename", "fold", "target", "category"}
    missing_columns = required_columns - set(metadata.columns)
    if missing_columns:
        raise ValueError(f"ESC-50 CSV is missing columns: {sorted(missing_columns)}")

    audio_df = pd.DataFrame(
        {
            "filename": metadata["filename"].astype(str).map(lambda x: Path(x).name),
            "fold": pd.to_numeric(metadata["fold"], errors="raise").astype(np.int64),
            "label_id": pd.to_numeric(metadata["target"], errors="raise").astype(np.int64),
            "label": metadata["category"].astype(str).str.strip(),
        }
    )

    if len(audio_df) != EXPECTED_SAMPLE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_SAMPLE_COUNT} ESC-50 rows, found {len(audio_df)}."
        )
    if set(audio_df["fold"].unique()) != set(FOLDS):
        raise RuntimeError("ESC-50 metadata must contain exactly folds 1 through 5.")
    if set(audio_df["label_id"].unique()) != set(range(NUM_CLASSES)):
        raise RuntimeError("ESC-50 target values must be exactly 0 through 49.")

    fold_counts = audio_df["fold"].value_counts().sort_index()
    if not (fold_counts == EXPECTED_SAMPLES_PER_FOLD).all():
        raise RuntimeError(
            f"Every fold must contain {EXPECTED_SAMPLES_PER_FOLD} samples; "
            f"observed {fold_counts.to_dict()}."
        )
    if audio_df["filename"].duplicated().any():
        raise RuntimeError("Duplicate filenames were found in ESC-50 metadata.")

    class_mapping = (
        audio_df[["label_id", "label"]].drop_duplicates().sort_values("label_id")
    )
    if (
        len(class_mapping) != NUM_CLASSES
        or class_mapping["label_id"].duplicated().any()
        or class_mapping["label"].duplicated().any()
    ):
        raise RuntimeError("ESC-50 target/category mapping is not one-to-one.")

    audio_df["audio_path"] = [
        _resolve_audio_path(data_root, csv_path, filename, int(fold))
        for filename, fold in zip(audio_df["filename"], audio_df["fold"])
    ]

    if validate_files:
        missing = ~audio_df["audio_path"].map(lambda value: Path(value).is_file())
        if missing.any():
            examples = audio_df.loc[missing, "audio_path"].head(20).tolist()
            raise FileNotFoundError(
                f"{int(missing.sum())} WAV files listed in esc50.csv are missing. "
                f"Examples: {examples}"
            )

    return audio_df


class ESC50Dataset(Dataset):
    """Waveform dataset matching the supplied ESC-50 evaluation notebook."""

    def __init__(self, dataframe, target_sample_rate=TARGET_SAMPLE_RATE, clip_seconds=5):
        self.paths = dataframe["audio_path"].tolist()
        self.labels = dataframe["label_id"].to_numpy(dtype=np.int64)
        self.target_sample_rate = int(target_sample_rate)
        self.target_num_samples = int(target_sample_rate * clip_seconds)
        self.resamplers = {}

    def __len__(self):
        return len(self.paths)

    def _resample(self, waveform, sample_rate):
        if sample_rate == self.target_sample_rate:
            return waveform
        if sample_rate not in self.resamplers:
            self.resamplers[sample_rate] = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.target_sample_rate
            )
        return self.resamplers[sample_rate](waveform)

    def __getitem__(self, index):
        audio_path = self.paths[index]
        waveform, sample_rate = torchaudio.load(audio_path)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = self._resample(waveform, sample_rate)

        if waveform.shape[-1] < self.target_num_samples:
            waveform = F.pad(
                waveform, (0, self.target_num_samples - waveform.shape[-1])
            )
        else:
            waveform = waveform[..., : self.target_num_samples]

        expected_shape = (1, self.target_num_samples)
        if tuple(waveform.shape) != expected_shape:
            raise RuntimeError(
                f"Unexpected waveform shape {tuple(waveform.shape)} for {audio_path}; "
                f"expected {expected_shape}."
            )

        label = torch.tensor(self.labels[index], dtype=torch.long)
        return waveform.to(torch.float32), label
