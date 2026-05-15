# Beatrice Trainer (Fork)

A fork of [Beatrice 2](https://prj-beatrice.com) voice conversion trainer with additional tooling for English-focused training and improved cross-gender conversion.

> **Original documentation**: See [README.original.md](README.original.md) for the upstream Beatrice Trainer documentation (in Japanese/English).

---

## What's New in This Fork

### 1. PhoneExtractor Trainer (`phone_extractor_trainer/`)

Retrain the PhoneExtractor on English-only data to remove Japanese phonetic bias.

```bash
make phone-train-en   # One-shot: download LibriSpeech + train + export
```

See [phone_extractor_trainer/README.md](phone_extractor_trainer/README.md) for details.

### 2. PitchEstimator Trainer (`pitch_estimator_trainer/`)

Retrain the PitchEstimator on diverse pitch data for better cross-gender (female→male) conversion.

```bash
make pitch-train-vctk   # One-shot: download VCTK + preprocess + train + export
```

See [pitch_estimator_trainer/README.md](pitch_estimator_trainer/README.md) for details.

### 3. Main Trainer Documentation (`beatrice_trainer/`)

Comprehensive documentation for the main voice conversion trainer.

See [beatrice_trainer/README.md](beatrice_trainer/README.md) for architecture, configuration, and usage.

---

## Quick Start

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (12+ GB VRAM recommended)
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

```bash
git lfs install
git clone https://github.com/leoliu1221/beatrice_td26.git
cd beatrice_td26
uv sync --extra cu128
```

### Train a Voice Model

1. **Prepare data** — one folder per speaker:
   ```
   inputs/my_dataset/
   ├── speaker_a/
   │   └── audio.wav
   └── speaker_b/
       └── audio.wav
   ```

2. **Train**:
   ```bash
   make DATASET=my_dataset
   ```

3. **Use** — load `outputs/my_dataset/paraphernalia_*/` in [Beatrice VST](https://prj-beatrice.com) or [VCClient](https://github.com/w-okada/voice-changer)

---

## Makefile Targets

### Main Trainer

| Target | Description |
|--------|-------------|
| `make` | Preprocess + train (default dataset: `lol_data`) |
| `make DATASET=foo` | Train on custom dataset |
| `make resume` | Resume training without re-preprocessing |
| `make tensorboard` | Monitor training progress |
| `make clean` | Remove preprocessed data and outputs |

### PhoneExtractor Trainer

| Target | Description |
|--------|-------------|
| `make phone-train-en` | Download LibriSpeech + train + export (one-shot) |
| `make phone-data-download` | Download LibriSpeech only |
| `make phone-train` | Train (requires `PHONE_DATA=`) |
| `make phone-export` | Export to Beatrice format |

### PitchEstimator Trainer

| Target | Description |
|--------|-------------|
| `make pitch-train-vctk` | Download VCTK + preprocess + train + export (one-shot) |
| `make pitch-data-download` | Download VCTK only |
| `make pitch-preprocess` | Pre-extract F0 (speeds up training 10x) |
| `make pitch-train` | Train (requires `PITCH_DATA=`) |
| `make pitch-export` | Export to Beatrice format |

---

## Configuration

After training the sub-models, update `assets/default_config.json`:

```json
{
    "phone_extractor_file": "assets/pretrained/phone_extractor_en.pt",
    "pitch_estimator_file": "assets/pretrained/pitch_estimator_v2.pt"
}
```

Then retrain your voice model to use the new extractors.

---

## Project Structure

```
beatrice_td26/
├── beatrice_trainer/          # Main voice conversion trainer
│   ├── __main__.py            # Training script
│   └── README.md              # Documentation
├── phone_extractor_trainer/   # PhoneExtractor retraining
│   ├── train.py, export.py, data.py
│   └── README.md
├── pitch_estimator_trainer/   # PitchEstimator retraining
│   ├── train.py, export.py, data.py, preprocess.py
│   └── README.md
├── assets/
│   ├── default_config.json    # Training configuration
│   └── pretrained/            # Pretrained model checkpoints
├── inputs/                    # Raw training audio (per dataset)
├── preprocessed/              # Segmented clips (auto-generated)
├── outputs/                   # Trained models + checkpoints
├── Makefile                   # Build automation
└── README.original.md         # Upstream documentation
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Credits

- Original [Beatrice Trainer](https://huggingface.co/fierce-cats/beatrice-trainer) by Project Beatrice
- This fork maintained by [@leoliu1221](https://github.com/leoliu1221)
