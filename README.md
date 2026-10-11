# A Structurally Aligned Inner–Outer Knowledge Distillation Pipeline for Lightweight Audio Classification

Repository preparation in progress.

This repository is intentionally kept small and paper-oriented. At the current stage, the ESC-50 student model, dataset preparation, preprocessing, and five-fold evaluation path have been extracted from the supplied ESC-50 notebook. UrbanSound8K, TAU Urban Acoustic Scenes, the inner/outer teachers, KD losses, augmentations, and training scripts are reserved as placeholders until their original experiment code is added.

## Repository status

| Component | ESC-50 | UrbanSound8K | TAU |
|---|---|---|---|
| Dataset loader | Implemented | Placeholder | Placeholder |
| Audio preparation | Implemented | Placeholder | Placeholder |
| Log-Mel preprocessing | Implemented | Placeholder | Placeholder |
| Student architecture | Implemented from supplied ESC notebook | Not yet extracted | Not yet extracted |
| Evaluation | Implemented | Placeholder | Placeholder |
| Checkpoints | Directory reserved; copy the five supplied fold checkpoints here | Placeholder | Placeholder |
| Inner teacher / KD / outer KD | Placeholder | Placeholder | Placeholder |

## ESC-50 dataset layout

The loader supports the common official layout:

```text
ESC-50-master/
├── audio/
│   └── *.wav
└── meta/
    └── esc50.csv
```

It also retains the alternate path patterns used by the supplied evaluation notebook.

## ESC-50 preprocessing

- Mono audio (multi-channel inputs are averaged)
- Sample rate: 32 kHz
- Clip duration: 5 seconds / 160,000 samples
- FFT size: 4096
- Window length: 4096
- Hop length: 502
- Mel bins: 256
- Power: 2.0
- dB range: 80 dB
- `center=True`
- `pad_mode="reflect"`
- `mel_scale="htk"`
- The evaluation path reproduces the training cache quantization via float16 -> float32 conversion.

## ESC-50 checkpoints

Place exactly one trained student checkpoint for each fold in:

```text
checkpoints/esc50/
```

The evaluator accepts either simplified names such as `fold1.pth` or the original notebook-style names such as `fold1.ESC50_94.50.pth`.

The actual checkpoint binaries were not part of the files available when this repository ZIP was generated, so no fake `.pth` files are included.

## Evaluation

```bash
python evaluate.py \
  --config configs/esc50.yaml \
  --data-root /path/to/ESC-50-master \
  --checkpoints-dir checkpoints/esc50
```

The ESC-50 path reports per-fold accuracy and aggregate mean, variance, standard deviation, and pooled accuracy.

## Method placeholders

The following files are intentionally reserved and not reconstructed from assumptions:

```text
models/inner_teacher.py
models/outer_teachers.py
augmentations.py
losses.py
train_inner_teacher.py
train_inner_kd.py
train_outer_kd.py
```

They should be populated from the corresponding original training code.
