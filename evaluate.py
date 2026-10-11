"""Repository-level evaluation entry point.

ESC-50 is implemented from the supplied notebook. UrbanSound8K and TAU remain
placeholders until their original evaluation code is added.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.esc50 import ESC50Dataset, load_esc50_metadata
from datasets.transforms import LogMelTransform
from models.student import StudentLightEchoNet
from utils import load_weights_only, seed_everything

ORIGINAL_NAME_PATTERN = re.compile(r"^fold([1-5])\.ESC50_([0-9]+(?:\.[0-9]+)?)\.pth$")
SIMPLE_NAME_PATTERN = re.compile(r"^fold([1-5])\.pth$")


def resolve_esc50_checkpoints(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {directory}")

    mapping = {}
    for path in directory.glob("*.pth"):
        match = ORIGINAL_NAME_PATTERN.fullmatch(path.name) or SIMPLE_NAME_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        fold = int(match.group(1))
        if fold in mapping:
            raise RuntimeError(f"More than one checkpoint found for fold {fold} in {directory}")
        filename_accuracy = None
        original_match = ORIGINAL_NAME_PATTERN.fullmatch(path.name)
        if original_match:
            filename_accuracy = float(original_match.group(2))
        mapping[fold] = {"path": path, "filename_accuracy_percent": filename_accuracy}

    missing = sorted(set(range(1, 6)) - set(mapping))
    if missing:
        raise FileNotFoundError(
            f"Missing ESC-50 checkpoints for folds {missing} in {directory}."
        )
    return mapping


@torch.inference_mode()
def evaluate_esc50(config, data_root, checkpoints_dir, device, batch_size=None, workers=0):
    dataset_cfg = config["dataset"]
    audio_cfg = config["audio"]
    spec_cfg = config["spectrogram"]

    metadata = load_esc50_metadata(data_root, validate_files=True)
    checkpoints = resolve_esc50_checkpoints(checkpoints_dir)

    batch_size = batch_size or (64 if device.type == "cuda" else 4)
    use_amp = device.type == "cuda"

    transform = LogMelTransform(
        sample_rate=audio_cfg["sample_rate"],
        clip_duration_seconds=audio_cfg["clip_duration_seconds"],
        n_fft=spec_cfg["n_fft"],
        win_length=spec_cfg["win_length"],
        hop_length=spec_cfg["hop_length"],
        n_mels=spec_cfg["n_mels"],
        power=spec_cfg["power"],
        top_db=spec_cfg["top_db"],
        center=spec_cfg["center"],
        pad_mode=spec_cfg["pad_mode"],
        mel_scale=spec_cfg["mel_scale"],
        emulate_float16_cache=spec_cfg.get("emulate_float16_cache", True),
        device=device,
    )

    results = []
    for fold in dataset_cfg["folds"]:
        fold_df = metadata.loc[metadata["fold"] == fold].reset_index(drop=True)
        ds = ESC50Dataset(
            fold_df,
            target_sample_rate=audio_cfg["sample_rate"],
            clip_seconds=audio_cfg["clip_duration_seconds"],
        )
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

        ckpt = checkpoints[fold]
        model = StudentLightEchoNet(num_classes=dataset_cfg["num_classes"])
        model.load_state_dict(load_weights_only(ckpt["path"]), strict=True)
        model = model.to(device, dtype=torch.float32).eval()

        correct = 0
        total = 0
        for waveforms, labels in tqdm(loader, desc=f"ESC-50 fold {fold}", leave=False):
            waveforms = waveforms.to(device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            mels = transform(waveforms)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(mels)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        acc = 100.0 * correct / total
        result = {
            "fold": fold,
            "model_file": ckpt["path"].name,
            "correct": correct,
            "samples": total,
            "accuracy_percent": acc,
        }
        if ckpt["filename_accuracy_percent"] is not None:
            result["filename_accuracy_percent"] = ckpt["filename_accuracy_percent"]
            result["difference_from_filename_pp"] = acc - ckpt["filename_accuracy_percent"]
        results.append(result)
        print(f"Fold {fold}: {acc:.6f}% ({correct}/{total})")

    df = pd.DataFrame(results).sort_values("fold").reset_index(drop=True)
    accs = df["accuracy_percent"].to_numpy(dtype=np.float64)
    total_correct = int(df["correct"].sum())
    total_samples = int(df["samples"].sum())

    print("\nPer-fold results")
    print(df.to_string(index=False))
    print("\nSummary across 5 folds")
    print(f"Arithmetic mean accuracy:       {np.mean(accs):.6f}%")
    print(f"Population variance (ddof=0):   {np.var(accs, ddof=0):.8f} percentage-points^2")
    print(f"Sample variance (ddof=1):       {np.var(accs, ddof=1):.8f} percentage-points^2")
    print(f"Population standard deviation:  {np.std(accs, ddof=0):.6f} percentage points")
    print(f"Sample standard deviation:      {np.std(accs, ddof=1):.6f} percentage points")
    print(f"Pooled accuracy:                 {100.0 * total_correct / total_samples:.6f}% ({total_correct}/{total_samples})")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/esc50.yaml"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints/esc50"))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset_name = str(config["dataset"]["name"]).lower()
    if dataset_name != "esc50":
        raise NotImplementedError(
            f"Evaluation for '{dataset_name}' has not yet been extracted."
        )

    seed_everything(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    evaluate_esc50(
        config,
        data_root=args.data_root,
        checkpoints_dir=args.checkpoints_dir,
        device=device,
        batch_size=args.batch_size,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
