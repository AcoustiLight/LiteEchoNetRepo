"""Repository-level evaluation entry point.

Evaluates released LightEchoNet student checkpoints on ESC-50 and UrbanSound8K
(k-fold) and on TAU Urban Acoustic Scenes 2022 Mobile (single dev-test split).
The student architecture and the log-Mel pipeline are identical across
datasets; only the clip length, metadata, and checkpoint layout differ.
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
from datasets.urbansound8k import load_urbansound8k_metadata
from datasets.tau import load_tau_metadata
from datasets.transforms import LogMelTransform
from models.student import StudentLightEchoNet
from utils import load_weights_only, seed_everything

KFOLD_TAGS = {"esc50": "ESC50", "urbansound8k": "US8K"}


def build_transform(config, device):
    audio_cfg = config["audio"]
    spec_cfg = config["spectrogram"]
    return LogMelTransform(
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


def _run_loader(model, loader, transform, device, use_amp, desc):
    correct = 0
    total = 0
    for waveforms, labels in tqdm(loader, desc=desc, leave=False):
        waveforms = waveforms.to(device, dtype=torch.float32, non_blocking=True)
        labels = labels.to(device, dtype=torch.long, non_blocking=True)
        mels = transform(waveforms)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(mels)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return correct, total


def resolve_kfold_checkpoints(directory, folds, tag):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {directory}")

    all_pths = list(directory.glob("*.pth"))
    mapping = {}
    for fold in folds:
        fold_re = re.compile(rf"^fold{fold}([._].*)?\.pth$")
        matches = [p for p in all_pths if fold_re.fullmatch(p.name)]
        if not matches:
            continue
        if len(matches) > 1:
            raise RuntimeError(
                f"More than one checkpoint matches fold {fold} in {directory}: "
                f"{[m.name for m in matches]}"
            )
        path = matches[0]
        filename_accuracy = None
        acc_match = re.search(rf"{tag}_([0-9]+(?:\.[0-9]+)?)", path.name)
        if acc_match:
            filename_accuracy = float(acc_match.group(1))
        mapping[fold] = {"path": path, "filename_accuracy_percent": filename_accuracy}

    missing = sorted(set(folds) - set(mapping))
    if missing:
        raise FileNotFoundError(
            f"Missing checkpoints for folds {missing} in {directory}. "
            f"Expected names like 'fold<N>.pth' or 'fold<N>.{tag}_<acc>.pth'."
        )
    return mapping


@torch.inference_mode()
def evaluate_kfold(config, dataset_name, data_root, checkpoints_dir, device,
                   batch_size=None, workers=0):
    dataset_cfg = config["dataset"]
    audio_cfg = config["audio"]
    tag = KFOLD_TAGS[dataset_name]

    if dataset_name == "esc50":
        metadata = load_esc50_metadata(data_root, validate_files=True)
    else:
        metadata = load_urbansound8k_metadata(data_root, validate_files=True)

    checkpoints = resolve_kfold_checkpoints(checkpoints_dir, dataset_cfg["folds"], tag)

    batch_size = batch_size or (64 if device.type == "cuda" else 4)
    use_amp = device.type == "cuda"
    transform = build_transform(config, device)

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

        correct, total = _run_loader(
            model, loader, transform, device, use_amp, f"{dataset_name} fold {fold}"
        )
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
    print(f"\nSummary across {len(df)} folds")
    print(f"Arithmetic mean accuracy:       {np.mean(accs):.6f}%")
    print(f"Population variance (ddof=0):   {np.var(accs, ddof=0):.8f} percentage-points^2")
    print(f"Sample variance (ddof=1):       {np.var(accs, ddof=1):.8f} percentage-points^2")
    print(f"Population standard deviation:  {np.std(accs, ddof=0):.6f} percentage points")
    print(f"Sample standard deviation:      {np.std(accs, ddof=1):.6f} percentage points")
    print(f"Pooled accuracy:                 {100.0 * total_correct / total_samples:.6f}% "
          f"({total_correct}/{total_samples})")


def resolve_single_checkpoint(checkpoints_dir, explicit_checkpoint=None):
    if explicit_checkpoint is not None:
        path = Path(explicit_checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    directory = Path(checkpoints_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {directory}")
    pths = sorted(directory.glob("*.pth"))
    if not pths:
        raise FileNotFoundError(f"No .pth checkpoint found in {directory}.")
    if len(pths) == 1:
        return pths[0]
    tau_pths = [p for p in pths if "tau" in p.name.lower()]
    if len(tau_pths) == 1:
        return tau_pths[0]
    raise RuntimeError(
        f"Multiple checkpoints in {directory}; pass --checkpoint to choose one: "
        f"{[p.name for p in pths]}"
    )


@torch.inference_mode()
def evaluate_tau(config, data_root, checkpoints_dir, device, batch_size=None,
                 workers=0, explicit_checkpoint=None, test_list_override=None):
    dataset_cfg = config["dataset"]
    audio_cfg = config["audio"]

    test_list = test_list_override or dataset_cfg.get("test_list")
    if not test_list:
        raise ValueError("TAU evaluation needs dataset.test_list (or --test-list).")

    metadata = load_tau_metadata(
        data_root,
        test_list=test_list,
        class_order=dataset_cfg.get("class_order"),
        validate_files=True,
    )

    checkpoint_path = resolve_single_checkpoint(checkpoints_dir, explicit_checkpoint)
    batch_size = batch_size or (64 if device.type == "cuda" else 4)
    use_amp = device.type == "cuda"
    transform = build_transform(config, device)

    ds = ESC50Dataset(
        metadata,
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

    model = StudentLightEchoNet(num_classes=dataset_cfg["num_classes"])
    model.load_state_dict(load_weights_only(checkpoint_path), strict=True)
    model = model.to(device, dtype=torch.float32).eval()

    correct, total = _run_loader(model, loader, transform, device, use_amp, "TAU dev-test")
    acc = 100.0 * correct / total
    print(f"Checkpoint: {checkpoint_path.name}")
    print(f"TAU dev-test accuracy: {acc:.6f}% ({correct}/{total})")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoints-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="TAU only: explicit checkpoint file.")
    parser.add_argument("--test-list", type=str, default=None,
                        help="TAU only: override dataset.test_list.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset_name = str(config["dataset"]["name"]).lower()
    seed_everything(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    checkpoints_dir = args.checkpoints_dir or Path("checkpoints") / dataset_name

    if dataset_name in KFOLD_TAGS:
        evaluate_kfold(
            config, dataset_name,
            data_root=args.data_root,
            checkpoints_dir=checkpoints_dir,
            device=device,
            batch_size=args.batch_size,
            workers=args.workers,
        )
    elif dataset_name == "tau":
        evaluate_tau(
            config,
            data_root=args.data_root,
            checkpoints_dir=checkpoints_dir,
            device=device,
            batch_size=args.batch_size,
            workers=args.workers,
            explicit_checkpoint=args.checkpoint,
            test_list_override=args.test_list,
        )
    else:
        raise NotImplementedError(f"Unknown dataset '{dataset_name}'.")


if __name__ == "__main__":
    main()
