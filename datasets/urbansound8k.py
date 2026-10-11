"""UrbanSound8K metadata validation, path resolution, and label mapping.

Dataset conventions here (UrbanSound8K.csv columns, audio/fold{N}/ layout, 10
uneven folds, 4 s clips at 32 kHz) follow the official UrbanSound8K release and
the preprocessing described in the paper. classID (0-9) is taken directly from
the CSV, so the label mapping is canonical. Verify one fold against its known
accuracy before trusting the 10-fold aggregate.
"""

from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_SAMPLE_COUNT = 8732
NUM_CLASSES = 10
FOLDS = tuple(range(1, 11))
TARGET_SAMPLE_RATE = 32000
CLIP_DURATION_SECONDS = 4


def find_urbansound8k_csv(root):
    root = Path(root)
    candidates = [
        root / "UrbanSound8K.csv",
        root / "metadata" / "UrbanSound8K.csv",
        root / "UrbanSound8K" / "metadata" / "UrbanSound8K.csv",
    ]
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _resolve_audio_path(data_root, csv_path, slice_file_name, fold):
    data_root = Path(data_root)
    csv_path = Path(csv_path)
    candidates = [
        data_root / "audio" / f"fold{fold}" / slice_file_name,
        data_root / "UrbanSound8K" / "audio" / f"fold{fold}" / slice_file_name,
        csv_path.parent.parent / "audio" / f"fold{fold}" / slice_file_name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return str(candidates[0])


def load_urbansound8k_metadata(data_root, validate_files=True):
    """Load and validate the UrbanSound8K metadata.

    Unlike ESC-50, UrbanSound8K folds are NOT equal in size, so only the total
    count, the fold set, and the class set are checked.
    """
    data_root = Path(data_root)
    csv_path = find_urbansound8k_csv(data_root)
    if csv_path is None:
        raise FileNotFoundError(f"Could not find UrbanSound8K.csv under {data_root}")

    metadata = pd.read_csv(csv_path)
    metadata.columns = [str(column).strip().lower() for column in metadata.columns]

    required_columns = {"slice_file_name", "fold", "classid", "class"}
    missing_columns = required_columns - set(metadata.columns)
    if missing_columns:
        raise ValueError(
            f"UrbanSound8K CSV is missing columns: {sorted(missing_columns)}"
        )

    audio_df = pd.DataFrame(
        {
            "filename": metadata["slice_file_name"].astype(str).map(lambda x: Path(x).name),
            "fold": pd.to_numeric(metadata["fold"], errors="raise").astype(np.int64),
            "label_id": pd.to_numeric(metadata["classid"], errors="raise").astype(np.int64),
            "label": metadata["class"].astype(str).str.strip(),
        }
    )

    if len(audio_df) != EXPECTED_SAMPLE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_SAMPLE_COUNT} UrbanSound8K rows, found {len(audio_df)}."
        )
    if set(audio_df["fold"].unique()) != set(FOLDS):
        raise RuntimeError("UrbanSound8K metadata must contain exactly folds 1 through 10.")
    if set(audio_df["label_id"].unique()) != set(range(NUM_CLASSES)):
        raise RuntimeError("UrbanSound8K classID values must be exactly 0 through 9.")

    class_mapping = (
        audio_df[["label_id", "label"]].drop_duplicates().sort_values("label_id")
    )
    if (
        len(class_mapping) != NUM_CLASSES
        or class_mapping["label_id"].duplicated().any()
        or class_mapping["label"].duplicated().any()
    ):
        raise RuntimeError("UrbanSound8K classID/class mapping is not one-to-one.")

    audio_df["audio_path"] = [
        _resolve_audio_path(data_root, csv_path, filename, int(fold))
        for filename, fold in zip(audio_df["filename"], audio_df["fold"])
    ]

    if validate_files:
        missing = ~audio_df["audio_path"].map(lambda value: Path(value).is_file())
        if missing.any():
            examples = audio_df.loc[missing, "audio_path"].head(20).tolist()
            raise FileNotFoundError(
                f"{int(missing.sum())} WAV files listed in UrbanSound8K.csv are missing. "
                f"Examples: {examples}"
            )

    return audio_df
