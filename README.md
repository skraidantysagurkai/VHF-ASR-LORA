# VHF-ASR

Automatic Speech Recognition pipeline for VHF AM radio communications (Air Traffic Control). The system simulates realistic VHF channel degradation, classifies signal quality, and fine-tunes Whisper with per-condition LoRA adapters.

## Pipeline overview

```
LibriSpeech / ATC audio
        │
        ▼
simulate_libre_data_to_vhf.py   — apply VHF AM channel simulation (noise, multipath, fading…)
        │
        ▼
label_data.py / label_real_asr.py — transcribe with base Whisper to produce manifests
        │
        ▼
train_classifiers.py             — train ECAPA signal-quality classifier (perfect/good/okay/bad)
        │
        ▼
train_lora.py                    — fine-tune per-condition LoRA adapters on frozen Whisper
        │
        ▼
eval_whole_pipe.py               — end-to-end WER evaluation (classifier → adapter routing)
```

## Requirements

- Python 3.13 (pinned — `requires-python = ">=3.13, < 3.14"`)
- [uv](https://docs.astral.sh/uv/) package manager
- CUDA 12.4 (PyTorch is pulled from the `pytorch-cu124` index)
- `ffmpeg` available on `PATH`

## Setup

```bash
# 1. Clone
git clone <repo-url>
cd VHF-ASR

# 2. Install dependencies (uv resolves the lockfile including CUDA PyTorch)
uv sync --extra-index-url https://download.pytorch.org/whl/cu124

# 3. (Optional) activate the venv for interactive use
source .venv/bin/activate
```

> **CPU-only / non-CUDA machines:** replace the `[tool.uv.sources]` block in `pyproject.toml` with the default PyPI torch entries, then re-run `uv sync`.

## Usage

All scripts live in `scripts/` and are run from the repo root so that `paths.py` resolves correctly.

### 1. Simulate VHF channel conditions

```bash
uv run python scripts/simulate_libre_data_to_vhf.py \
    --input-dir  data/raw \
    --output-dir data/simulated \
    --condition  good          # perfect | good | okay | bad
```

### 2. Label audio with base Whisper

```bash
uv run python scripts/label_data.py \
    --audio-dir data/simulated \
    --output    data/manifests/train.jsonl
```

### 3. Train the signal-quality classifier

```bash
uv run python scripts/train_classifiers.py \
    --manifest  data/manifests/train.jsonl \
    --output    checkpoints/classifier.pt
```

### 4. Fine-tune LoRA adapters

```bash
uv run python scripts/train_lora.py \
    --manifest     data/manifests/train.jsonl \
    --val-manifest data/manifests/dev.jsonl \
    --output-dir   adapters \
    --conditions   perfect good okay bad \
    --epochs 5 --batch 8
```

### 5. Evaluate the full pipeline

```bash
uv run python scripts/eval_whole_pipe.py \
    --audio-dir   data/test \
    --adapter-dir adapters \
    --classifier  checkpoints/classifier.pt
```

## GNU Radio simulations

The `gnu-radio-sim/` directory contains GRC flowgraphs and Python scripts for hardware-in-the-loop VHF AM simulation. Open `.grc` files with GNU Radio Companion (≥ 3.10).

## Notebooks

Exploratory notebooks are in `notebooks/` and follow the naming convention `[YYYY-MM-DD]-<topic>.ipynb`. Launch with:

```bash
uv run jupyter lab
```

## Project layout

```
VHF-ASR/
├── scripts/          # all pipeline scripts
├── notebooks/        # exploratory analysis
├── gnu-radio-sim/    # GRC flowgraphs for hardware simulation
├── checkpoints/      # saved classifier weights
├── data/             # raw, simulated, and manifest data (gitignored)
├── results/          # evaluation outputs (gitignored)
├── paths.py          # ROOT_DIR / DATA_DIR constants
└── pyproject.toml
```