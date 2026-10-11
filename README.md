# LightEchoNet: Structurally Aligned Inner–Outer Knowledge Distillation for Lightweight Audio Classification

Official PyTorch implementation and released checkpoints for **"LightEchoNet: Structurally Aligned Inner–Outer Knowledge Distillation for Lightweight Audio Classification."**

> **Authors:** Mohamad Mahdee Ramezanee\*, Hossein Sharifi\*, Behnam Raoufi\*, Reza Tavakoli, Siavash Ahmadi†, Babak Khalaj, Saeed Bagheri Shouraki, Mohammad Sabokrou, Ali Diba†
> \* Equal contribution. † Corresponding authors.
>
> Sharif University of Technology · Virginia Tech · New Uzbekistan University · Okinawa Institute of Science and Technology · QCRI, Hamad Bin Khalifa University

**Project page:** https://acoustilight.github.io/LightEchoNet

<p align="center">
  <img src="assets/overview.png" width="100%" alt="Structurally aligned inner–outer knowledge distillation pipeline">
</p>


## 📖 Abstract

> We present LightEchoNet, a lightweight audio classification framework designed for resource-constrained devices. Its compact CNN student provides strong standalone performance with low parameter and computational costs. The Inner Block is structurally aligned with the student and uses FusionConv blocks to produce compatible intermediate representations without additional projection modules. The proposed staged distillation strategy combines normalized Huber-based feature matching and temperature-scaled logit supervision from the Inner Block with complementary output-level guidance from the Outer Block. The final student achieves 60.24% accuracy on TAU using the official 25% development-training subset, 93.4% on ESC-50, and 94.43% on UrbanSound8K. For a one-second audio clip, the model requires only 31.2K parameters, 28.6 MMACs, and 62.5 kB of storage in float16 format.


## 🧩 Method

LightEchoNet co-designs a compact CNN **student** with a structurally aligned **Inner Teacher**, then distills from an **Outer Teacher** ensemble. Nine aligned teacher–student (T–S) block pairs are distributed through the network so intermediate features can be matched directly, without extra projection modules. Training uses three phases:

- **Phase I — Inner Teacher:** train the structurally aligned inner teacher, whose FusionConv blocks mirror the student's `3×3` convolutions.
- **Phase II — Student (inner KD):** cross-entropy, temperature-scaled KL on inner-teacher logits, and a normalized Huber feature-matching loss (`α = 0.6`, `β = 0.8`, `T = 5`).
- **Phase III — Student (outer KD):** emphasize the outer-teacher ensemble with `T = 8`, `α = 0.95`.

All teachers are discarded after training, so inference cost is unchanged. **This repository releases the trained students and the evaluation code** — the student is the only network needed to reproduce the reported accuracies.


## ⚙️ Setup

```bash
git clone https://github.com/<your-org>/LightEchoNet.git
cd LightEchoNet

conda create -n lightechonet python=3.9
conda activate lightechonet

# Adjust the CUDA build to your system.
conda install pytorch==2.1.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

The pipeline uses `torchaudio` for waveform loading and resampling; ensure your `torchaudio` backend can decode the dataset audio (e.g. FFmpeg).


## 🤖 Model

The released student (`models/student.py`) is a compact CNN: a two-layer stem, three stages of Light Echo (LE) blocks with Residual Normalization, an attention-based pooling head, and a linear classifier — about **31.2K parameters** for a one-second input. The same architecture is used for every dataset; only the classifier width (`num_classes`) and the input clip length change. Checkpoints are raw tensor-only state dicts loaded with `strict=True`, so the architecture must not be modified.


## 📁 Data Preparation

All audio is converted to mono and resampled to **32 kHz**. Log-Mel features use FFT size **4096**, hop length **502**, **256** mel bands, power **2.0**, and an **80 dB** range. Clip length is the only per-dataset audio difference.

| Dataset | Classes | Clip | Protocol | Config |
|---|---|---|---|---|
| [ESC-50](https://github.com/karolpiczak/ESC-50) | 50 | 5 s | 5-fold CV | `configs/esc50.yaml` |
| [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) | 10 | 4 s | 10-fold CV | `configs/urbansound8k.yaml` |
| [TAU 2022 Mobile](https://zenodo.org/record/6337421) | 10 | 1 s | Dev-test split | `configs/tau.yaml` |

### ESC-50

Point `--data-root` at a directory with `esc50.csv` (auto-detected in the standard locations) and the `audio/` folder. The loader validates 2000 clips / 5 folds / 50 classes before running.

### UrbanSound8K

Expects the official layout:

```text
/path/to/UrbanSound8K/
├── metadata/UrbanSound8K.csv
└── audio/fold1 … fold10/*.wav
```

`classID` (0–9) from the CSV is used directly as the label, so the mapping is canonical. Folds are uneven, as in the official release.

### TAU

TAU's metadata stores only the string `scene_label`, so `configs/tau.yaml` defines the **`class_order`** that maps labels to indices. This must match the order used when the TAU checkpoint was trained (default: alphabetical). Set `test_list` to the split your results are reported on; it must contain `filename` and `scene_label` columns.


## 🚀 Evaluation

Place the per-fold / per-model checkpoints under `checkpoints/<dataset>/`. For k-fold datasets name them `fold<N>.pth` (or `fold<N>.ESC50_<acc>.pth` / `fold<N>.US8K_<acc>.pth` to also report the delta from the filename accuracy):

```text
checkpoints/
├── esc50/fold1.pth … fold5.pth
├── urbansound8k/fold1.pth … fold10.pth
└── tau/<single-model>.pth
```

```bash
# ESC-50 (5-fold)
python evaluate.py --config configs/esc50.yaml --data-root /path/to/ESC-50

# UrbanSound8K (10-fold)
python evaluate.py --config configs/urbansound8k.yaml --data-root /path/to/UrbanSound8K

# TAU (single dev-test split)
python evaluate.py --config configs/tau.yaml --data-root /path/to/TAU \
  --test-list evaluation_setup/fold1_evaluate.csv
```

K-fold runs print per-fold accuracy plus the mean, population/sample variance and standard deviation, and the pooled accuracy. `--checkpoints-dir` defaults to `checkpoints/<dataset>` and can be overridden.


## 📊 Results

Student accuracy on the official splits (from the paper):

| Dataset | Accuracy (%) | Params | MMACs | Size (float16) |
|---|---|---|---|---|
| TAU (25% dev-train) | 60.2 | 31.2K | 28.6 | 62.5 kB |
| ESC-50 | 93.4 ± 0.9 | 36K | — | — |
| UrbanSound8K | 94.4 ± 1.9 | 31.2K | — | — |

Scaling the student to ~350K parameters (`LightEchoNet-350K`) raises ESC-50 to **95.8 ± 0.83%** and UrbanSound8K to **96.25 ± 1.72%**.


## 🛠️ Notes

The dataset loaders follow the official dataset conventions and the paper's preprocessing. ESC-50 and UrbanSound8K label mappings are taken directly from the dataset CSVs; the TAU mapping is set by `class_order` in the config. Verify one fold/model against a known accuracy after setting your paths. Training code (inner/outer teachers, distillation losses, augmentations) is not part of this checkpoint-evaluation release.


## 📝 Citation

```bibtex
@inproceedings{ramezanee2026lightechonet,
  title     = {LightEchoNet: Structurally Aligned Inner--Outer Knowledge Distillation for Lightweight Audio Classification},
  author    = {Ramezanee, Mohamad Mahdee and Sharifi, Hossein and Raoufi, Behnam and Tavakoli, Reza and Ahmadi, Siavash and Khalaj, Babak and Bagheri Shouraki, Saeed and Sabokrou, Mohammad and Diba, Ali},
  booktitle = {TODO: fill in venue},
  year      = {2026}
}
```
