# LightEchoNet: Structurally Aligned Inner–Outer Knowledge Distillation for Lightweight Audio Classification

Official PyTorch implementation of **"LightEchoNet: Structurally Aligned Inner–Outer Knowledge Distillation for Lightweight Audio Classification."**

> **Authors:** Mohamad Mahdee Ramezanee\*, Hossein Sharifi\*, Behnam Raoufi\*, Reza Tavakoli, Siavash Ahmadi†, Babak Khalaj, Saeed Bagheri Shouraki, Mohammad Sabokrou, Ali Diba†  
> \* Equal contribution. † Corresponding authors.
>
> ¹ Sharif University of Technology, Iran · ² Virginia Tech, USA · ³ New Uzbekistan University, Uzbekistan · ⁴ Okinawa Institute of Science and Technology, Japan · ⁵ QCRI, Hamad Bin Khalifa University, Qatar

**Project page:** [acoustilight.github.io/LightEchoNet](https://acoustilight.github.io/LightEchoNet) · **Paper:** [`LightEchoNet.pdf`](LightEchoNet.pdf)

<p align="center">
  <img src="assets/overview.png" width="100%" alt="Structurally aligned inner teacher, compact student, and the inner–outer distillation pipeline">
</p>


## 📖 Abstract

> We present LightEchoNet, a lightweight audio classification framework designed for resource-constrained devices. Its compact CNN student provides strong standalone performance with low parameter and computational costs. The Inner Block is structurally aligned with the student and uses FusionConv blocks to produce compatible intermediate representations without additional projection modules. The proposed staged distillation strategy combines normalized Huber-based feature matching and temperature-scaled logit supervision from the Inner Block with complementary output-level guidance from the Outer Block. The final student achieves 60.24% accuracy on TAU using the official 25% development-training subset, 93.4% on ESC-50, and 94.43% on UrbanSound8K. For a one-second audio clip, the model requires only 31.2K parameters, 28.6 MMACs, and 62.5 kB of storage in float16 format.


## 🧩 Method

LightEchoNet co-designs a compact CNN **student** with a **structurally aligned inner teacher**, formed by replacing nine of the student's 3×3 convolutions with FusionConv blocks. Because both networks share stage layout and feature dimensions, intermediate features can be matched directly — with no projection modules. Knowledge is then transferred through a three-phase pipeline, after which all teachers are discarded, leaving the student's inference cost unchanged.

- **Phase I — Inner-Teacher Training:** Train the structurally aligned inner teacher (Light Echo blocks + FusionConv, Residual Normalization, and an attention-GAP head), pretrained on AudioSet and CochlScene, then fine-tuned on the target dataset. ~140K parameters.
- **Phase II — Inner Knowledge Distillation:** Train the ~77%-smaller student with (i) cross-entropy on true labels, (ii) temperature-scaled KL divergence against the inner-teacher logits, and (iii) normalized Huber feature matching at `|S| = 4` aligned layers. Settings: `α = 0.6`, `β = 0.8`, `T = 5`.
- **Phase III — Outer Knowledge Distillation:** Introduce the outer-teacher ensemble (LEN-v1, LEN-v2, CP-Mobile, CP-ResNet, HTS-AT) and raise `T = 8`, `α = 0.95` to emphasize output-level guidance that refines the already-strong student.

Selected features are ℓ₂-normalized to remove scale differences before matching:

```math
\widehat{f}^{(l)}_s = \frac{f^{(l)}_s}{\lVert f^{(l)}_s \rVert_2 + \varepsilon}, \qquad
\widehat{f}^{(l)}_i = \frac{f^{(l)}_i}{\lVert f^{(l)}_i \rVert_2 + \varepsilon}.
```

The complete training objective combines label, logit, and feature supervision:

```math
\mathcal{L} = (1-\alpha)\,\mathcal{L}_{\mathrm{CE}}(z_s, y) + \alpha T^2\,\mathcal{L}_{\mathrm{KL}}\!\left(p_t^T \,\Vert\, p_s^T\right) + \beta\,\mathcal{L}_{\mathrm{feat}},
```

where `L_feat` is an element-wise Huber loss (`δ = 1`) over the `|S| = 4` aligned feature maps, with stop-gradient on the teacher side.


## 📈 Results

Accuracy with the full inner–outer pipeline, together with the student's deployment complexity (FP16). ESC-50 and UrbanSound8K are reported on the official folds.

| Dataset | Accuracy (%) | Student Params | MMACs | Size (FP16) |
|---|---|---|---|---|
| TAU Urban Acoustic Scenes 2022 Mobile (official 25% split) | **60.24** | 31.2K | 28.6 | 62.5 kB |
| ESC-50 | **93.4** | 36.4K | 143.2 | 72.8 kB |
| UrbanSound8K | **94.43** | 31.2K | 114.5 | 62.5 kB |

> MMACs/size vary with clip length (1 s for TAU, 4 s for UrbanSound8K, 5 s for ESC-50). Scaling the student and inner teacher to a **350K** variant further raises accuracy to **95.8%** on ESC-50 and **96.25%** on UrbanSound8K.


## ⚙️ Setup

Clone the repository:

```bash
git clone https://github.com/<user>/LightEchoNet.git
cd LightEchoNet
```

Create a Python environment:

```bash
conda create -n lightechonet python=3.9
conda activate lightechonet
```

Install PyTorch and the remaining dependencies:

```bash
conda install pytorch torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt   # numpy pandas pyyaml tqdm
```

Adjust the PyTorch/CUDA build to match your system. The repository uses `torchaudio` for waveform loading and resampling, so ensure your `torchaudio` backend can decode the dataset's audio format (e.g. an FFmpeg or SoX backend for `.wav`). Evaluation runs on CPU or a single GPU; the original experiments used an NVIDIA RTX 3090.


## 🤖 Models

The distillation framework involves three model roles. Only the student is needed at inference; the teachers are used during training and then discarded.

| Role | Module | Status |
|---|---|---|
| **Student** — `StudentLightEchoNet` (LE blocks, ResNorm, learnable AttnGAP pooling) | [`models/student.py`](models/student.py), [`models/blocks.py`](models/blocks.py) | ✅ Implemented |
| **Inner teacher** — structurally aligned, FusionConv-based | [`models/inner_teacher.py`](models/inner_teacher.py) | 🚧 Placeholder |
| **Outer teachers** — LEN-v1, LEN-v2, CP-Mobile, CP-ResNet, HTS-AT | [`models/outer_teachers.py`](models/outer_teachers.py) | 🚧 Placeholder |

### Checkpoints

Pretrained student checkpoints are organized per dataset and per fold:

```text
checkpoints/
├── esc50/          # fold1.ESC50_93.xx.pth ... fold5.ESC50_xx.xx.pth
├── urbansound8k/
└── tau/
```

The ESC-50 evaluator accepts either the accuracy-annotated naming `fold<N>.ESC50_<acc>.pth` or the simple `fold<N>.pth`, for `N ∈ {1,…,5}`. When the accuracy is encoded in the filename, the evaluator reports the measured-vs-filename difference as a sanity check. Checkpoints are raw tensor-only `state_dict` files loaded with `weights_only=True`.


## 📁 Data Preparation

LightEchoNet is evaluated on three public audio-classification benchmarks. Set the dataset location with `--data-root` (ESC-50) or the corresponding config field.

| Dataset | Classes | Clip length | Config | Status |
|---|---|---|---|---|
| [ESC-50](https://github.com/karolpiczak/ESC-50) | 50 | 5 s | [`configs/esc50.yaml`](configs/esc50.yaml) | ✅ Implemented |
| [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) | 10 | ≤ 4 s | [`configs/urbansound8k.yaml`](configs/urbansound8k.yaml) | 🚧 Placeholder |
| [TAU Urban Acoustic Scenes 2022 Mobile](https://zenodo.org/record/6337421) | 10 | 1 s | [`configs/tau.yaml`](configs/tau.yaml) | 🚧 Placeholder |

### ESC-50 layout

The ESC-50 loader auto-discovers `esc50.csv` and the audio directory. The standard release layout works out of the box:

```text
/path/to/ESC-50-master/
├── meta/
│   └── esc50.csv
└── audio/
    └── *.wav
```

The loader strictly validates the metadata (2000 clips, folds 1–5 with 400 each, targets 0–49, one-to-one target↔category mapping, no duplicate filenames) and resolves every WAV path before evaluation.

### Audio preprocessing

All signals are converted to mono and resampled to 32 kHz, then clipped or zero-padded to the dataset's clip length. Log-Mel spectrograms follow the paper's settings:

| Parameter | Value |
|---|---|
| Sample rate | 32 kHz |
| FFT / window size | 4096 |
| Hop length | 502 |
| Mel bins | 256 |
| Power | 2.0 |
| Amplitude→dB top_dB | 80.0 |
| Mel scale | HTK |

> The ESC-50 transform reproduces the original training pipeline's **float16 Mel cache** by quantizing each log-Mel to `float16` and back to `float32` before the model (`emulate_float16_cache: true`).


## 🚀 Evaluation

ESC-50 evaluation is fully implemented. It runs all five official folds, applies the matching checkpoint per fold, and reports per-fold and aggregate accuracy:

```bash
python evaluate.py \
  --config configs/esc50.yaml \
  --data-root /path/to/ESC-50-master \
  --checkpoints-dir checkpoints/esc50 \
  --batch-size 64 \
  --workers 4
```

On CUDA the evaluator uses mixed precision (FP16 autocast) and a default batch size of 64; on CPU it defaults to a batch size of 4. The random seed is fixed from the config (`seed: 42`) for reproducibility.


## 🏋️ Training

The three-phase pipeline is scaffolded as separate entry points:

```bash
python train_inner_teacher.py   # Phase I  — inner-teacher training
python train_inner_kd.py        # Phase II — inner knowledge distillation (feature + logit)
python train_outer_kd.py        # Phase III — outer knowledge distillation (ensemble refinement)
```

> **Note:** The training stages, the inner/outer teachers, the distillation losses ([`losses.py`](losses.py)), and the augmentation stack ([`augmentations.py`](augmentations.py)) are currently placeholders and raise `NotImplementedError`. See **Repository Status** below.


## 📊 Outputs

The ESC-50 evaluator prints a per-fold table followed by a summary across the five folds:

```text
Per-fold results
 fold  model_file           correct  samples  accuracy_percent  ...

Summary across 5 folds
Arithmetic mean accuracy:       ...
Population variance (ddof=0):   ...
Sample variance (ddof=1):       ...
Population standard deviation:  ...
Sample standard deviation:      ...
Pooled accuracy:                ... (correct/total)
```

When checkpoints carry accuracy-annotated filenames, each row additionally reports the measured accuracy's deviation from the filename value.


## ⚙️ Main Paper Settings

| Component | Setting |
|---|---|
| Optimizer | Adam, cosine annealing, fixed seed |
| Initial learning rate / batch size | `1e-3` / 64 |
| Phase II (inner KD) | `α = 0.6`, `β = 0.8`, `T = 5` |
| Phase III (outer KD) | `α = 0.95`, `T = 8` |
| Feature-loss layers / loss | `|S| = 4` / Huber (`δ = 1`), ℓ₂-normalized |
| Mel front end | 32 kHz, FFT 4096, hop 502, 256 bins, power 2.0, 80 dB |
| Outer-teacher ensemble | LEN-v1, LEN-v2, CP-Mobile, CP-ResNet, HTS-AT |
| Training hardware | NVIDIA RTX 3090 (24 GB) |
| Deployment precision | FP16 (≈½ model size, <0.1 pp accuracy loss) |

All experiment settings are exposed in the YAML files under [`configs/`](configs/).


## 📦 Repository Status

This repository is released incrementally. The ESC-50 evaluation path is complete and reproduces the reported fold accuracies from released checkpoints. Other components are being ported from the original experiment code.

| Component | Status |
|---|---|
| ESC-50 dataset, transform, student, evaluation | ✅ Implemented |
| UrbanSound8K / TAU datasets + configs | 🚧 Placeholder |
| Inner & outer teachers | 🚧 Placeholder |
| Distillation losses, augmentations | 🚧 Placeholder |
| Phase I / II / III training | 🚧 Placeholder |


## 📝 Citation

If you find our code or method useful in your research, please consider citing our paper:

```bibtex
@inproceedings{ramezanee2026lightechonet,
  title     = {LightEchoNet: Structurally Aligned Inner--Outer Knowledge Distillation for Lightweight Audio Classification},
  author    = {Ramezanee, Mohamad Mahdee and Sharifi, Hossein and Raoufi, Behnam and Tavakoli, Reza and Ahmadi, Siavash and Khalaj, Babak and Bagheri Shouraki, Saeed and Sabokrou, Mohammad and Diba, Ali},
  booktitle = {TODO: venue},
  year      = {TODO},
  note      = {Project page: https://acoustilight.github.io/LightEchoNet}
}
```
