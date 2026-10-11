"""TAU Urban Acoustic Scenes 2022 Mobile — test-split metadata loading.

TAU metadata carries only the string `scene_label` (no numeric id), so the
label -> index mapping comes from `dataset.class_order` in the config and MUST
match the order used during training. Source clips (1 s, 44.1 kHz) are
resampled to 32 kHz by the shared waveform loader. `dataset.test_list` must
point at the split the paper reports on; it needs a 'filename' column and a
'scene_label' column (a filename-only test list cannot be scored).
"""

from pathlib import Path

import numpy as np
import pandas as pd

NUM_CLASSES = 10
TARGET_SAMPLE_RATE = 32000
CLIP_DURATION_SECONDS = 1


def _read_split_csv(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"TAU split list not found: {path}")
    # DCASE setup files are tab-separated; sniff to also accept comma/space.
    frame = pd.read_csv(path, sep=None, engine="python")
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    return frame


def _resolve_audio_path(data_root, filename):
    data_root = Path(data_root)
    name = str(filename).replace("\\", "/")
    base = Path(name).name
    candidates = [
        data_root / name,
        data_root / "audio" / base,
        data_root / "TAU-urban-acoustic-scenes-2022-mobile-development" / name,
        data_root / "TAU-urban-acoustic-scenes-2022-mobile-development" / "audio" / base,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return str(candidates[0])


def load_tau_metadata(data_root, test_list, class_order, validate_files=True):
    data_root = Path(data_root)

    if class_order is None or len(class_order) != NUM_CLASSES:
        raise ValueError(
            "config dataset.class_order must list exactly 10 TAU scene labels "
            "in the order used during training."
        )
    label_to_id = {str(label).strip(): index for index, label in enumerate(class_order)}
    if len(label_to_id) != NUM_CLASSES:
        raise ValueError("dataset.class_order contains duplicate labels.")

    # Resolve the split list relative to the data root if it is not absolute.
    test_path = Path(test_list)
    if not test_path.is_absolute() and not test_path.is_file():
        test_path = data_root / test_list

    split = _read_split_csv(test_path)
    if "filename" not in split.columns:
        raise ValueError(f"TAU split list {test_path} has no 'filename' column.")
    if "scene_label" not in split.columns:
        raise ValueError(
            f"TAU split list {test_path} has no 'scene_label' column, so it "
            "cannot be scored. Use the evaluate/meta file that includes labels."
        )

    labels = split["scene_label"].astype(str).str.strip()
    unknown = sorted(set(labels) - set(label_to_id))
    if unknown:
        raise RuntimeError(
            f"Scene labels not present in dataset.class_order: {unknown}"
        )

    audio_df = pd.DataFrame(
        {
            "filename": split["filename"].astype(str),
            "label": labels,
            "label_id": labels.map(label_to_id).astype(np.int64),
        }
    )
    audio_df["audio_path"] = [
        _resolve_audio_path(data_root, filename) for filename in audio_df["filename"]
    ]

    if validate_files:
        missing = ~audio_df["audio_path"].map(lambda value: Path(value).is_file())
        if missing.any():
            examples = audio_df.loc[missing, "audio_path"].head(20).tolist()
            raise FileNotFoundError(
                f"{int(missing.sum())} TAU WAV files are missing. Examples: {examples}"
            )

    return audio_df
