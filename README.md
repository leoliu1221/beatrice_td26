# Beatrice Trainer (Fork)

A fork of [Beatrice 2](https://prj-beatrice.com) voice conversion trainer with additional tooling for English-focused training and improved cross-gender conversion.

> **Original documentation**: See [README.original.md](README.original.md) for the upstream Beatrice Trainer documentation (in Japanese/English).

---

## What's New in This Fork

### 1. PhoneExtractor Trainer (`phone_extractor_trainer/`)

Tooling to retrain the PhoneExtractor on English-only data (originally to remove
Japanese phonetic bias).

```bash
make phone-train-en   # One-shot: download LibriSpeech + train + export
```

> **Note (2026-06-02):** our English retrains (`en_clean_*`, `en_nr_targetmix02_300k`)
> turned out **worse** than the upstream `jp_122` extractor on a phonetic-contrast
> probe and in listening tests, so `current.pt` points at upstream `jp_122`. Keep
> the trainer for experiments, but a new variant must beat upstream on the probe
> **and** a listening test before promotion. See the Experiments section and
> `project_context.md` lessons 2.6 / 2.7.

See [phone_extractor_trainer/README.md](phone_extractor_trainer/README.md) for details.

### 2. PitchEstimator Trainer (`pitch_estimator_trainer/`)

Tooling to retrain the PitchEstimator on diverse pitch data (e.g. for cross-gender
female→male conversion).

```bash
make pitch-train-vctk   # One-shot: download VCTK + preprocess + train + export
```

> **Note (2026-06-02):** the noise-robust `vctk_nr_300k` retrain showed no benefit
> over upstream `jp_104` on an F0 probe, so `current.pt` points at upstream `jp_104`.

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

The shipped `assets/default_config.json` references stable symlinks under the
pretrained registry:

```json
{
    "phone_extractor_file": "assets/pretrained/phone_extractor/current.pt",
    "pitch_estimator_file": "assets/pretrained/pitch_estimator/current.pt",
    "pretrained_file":      "assets/pretrained/vocoder/current.pt.gz"
}
```

To swap to a different variant after training, either retarget the symlink:

```bash
ln -sfn jp_122_3000k.pt assets/pretrained/phone_extractor/current.pt
```

or pin a specific variant directly in your config (recommended for A/B runs):

```json
"phone_extractor_file": "assets/pretrained/phone_extractor/jp_122_3000k.pt"
```

See [`assets/pretrained/README.md`](assets/pretrained/README.md) and
[`assets/pretrained/manifest.json`](assets/pretrained/manifest.json) for the
full registry and naming convention.

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

## Pretrained Models

Variants live under `assets/pretrained/<module>/` and are **immutable** — every
export gets a unique descriptive filename. Each module has a `current.pt`
symlink to the recommended variant; the default config resolves through it.

| Module | Current variant | Training data | Notes |
|---|---|---|---|
| **PhoneExtractor** | `phone_extractor/jp_122_3000k.pt` (**upstream**) | Upstream Beatrice (Japanese-leaning) | Reverted to upstream 2026-06-02. Our English-distilled variants regressed phonetic richness ("big tongue") — see note below. Richest of all variants on the contrast probe, even on English clips. |
| **PitchEstimator** | `pitch_estimator/jp_104_3_300k.pt` (**upstream**) | Upstream Beatrice | Reverted to upstream 2026-06-02. The noise-robust `vctk_nr_300k` showed no benefit (worst on noisy input) in the F0 probe. |
| **Vocoder** | `vocoder/libritts_r_200_2750k.pt.gz` | Upstream LibriTTS-R | Warm-start for the converter network. |

> **2026-06-02 — upstream stack restored.** A "big tongue" (mumbled / under-articulated) regression in fine-tunes was traced to our **retrained** extractors, not the converter. The English PhoneExtractor distillation (`en_clean_580k`, `en_nr_targetmix02_300k`) collapsed phonetic resolution, and the noise-robust PitchEstimator gave no benefit. Both `current.pt` symlinks were reverted to the upstream `jp_122` / `jp_104` checkpoints. **Cosine similarity to HuBERT is not a sufficient selection metric** — it rewards consistency with a (possibly smeared) target, not phonetic separability. Use the probes in `analysis/` plus a listening test. See `project_context.md` lessons 2.6 / 2.7.

Deprecated/regressed variants (`phone_extractor/en_clean_*.pt`, `phone_extractor/en_nr_targetmix02_300k.pt`, `pitch_estimator/vctk_*.pt`) are kept on disk for ablations but should **not** be used. Each variant has a sibling `.meta.json` with full recipe + eval + probe scores.

---

## Experiments & Findings (`new_lol_data` voice fine-tuning)

Full lab notebook in [`experimental_log.md`](experimental_log.md); invariant
rules and reusable patterns are codified in
[`project_context.md`](project_context.md). Highlights:

### The small-dataset noise trap (Experiment 3, Phase A)

Lowering `phone_noise_ratio` to `0.1` and `floor_noise_level` to `1e-5` made
the model **overfit to silent gaps in the target audio**: it reproduced the
room hiss as part of speaker identity. **Fix**: keep the noise-ratio at `0.5`
and the augmentation probabilities at their defaults (`reverb: 0.5`,
`formant_shift: 0.5`); control output static by lowering `floor_noise_level`
to `1e-4` instead.

### Deep-voice buzz traces to TTS source artifacts (Phases B–D)

Even with balanced regularization, deep voices kept a buzz/static during
speech. The artifacts were baked into the source TTS recordings. We tested two
source-level denoisers:

| Method | Tool | What it removes | Verdict |
|---|---|---|---|
| Spectral subtraction | [`denoise_sources.py`](denoise_sources.py) (`noisereduce`) | Stationary background hum | Helps but misses phoneme-correlated artifacts |
| Deep-learning denoise | [`denoise_sources_df.py`](denoise_sources_df.py) (DeepFilterNet3) | Non-stationary buzz, vocoder ringing, transients | **Preferred for TTS sources** |

DeepFilterNet needs a Rust toolchain (`curl https://sh.rustup.rs ... | sh`)
and long inputs must be chunked (the script handles 30s chunks; otherwise
cuDNN GRU crashes). Always pair denoising with a permissive VAD
(`uv run python preprocess.py --dataset <foo>_df --energy-threshold 35`).

### Feature extractors must train with Beatrice's augmentation (Experiment 4)

Beatrice's main trainer applies `augment_audio()` (noise SNR 20–45 dB, IR
reverb, LPF, formant shift) to inputs before passing them to the
PhoneExtractor and PitchEstimator. The shipped extractors were distilled on
**clean** audio only, so they produced unstable features under augmentation,
and the converter learned to reproduce that instability as audible noise
during conversion.

**Fix**: noise-robust / consistency distillation. The student sees
`augment_audio(clean)`; targets (HuBERT features / pyworld F0) are computed
from the clean wav. Toggle with `AUGMENT=1` on either Makefile target.

Naive resume (fine-tune a clean-overfit 580k checkpoint with `--augment`)
actually **regressed target-domain performance** by ~3.5%. The fix that
worked was training from scratch with augmentation **and** 20% target-domain
mixing (`--target-data-dir preprocessed/<foo>_df --target-mix-ratio 0.2`).
Held-out cos sim on the new LoL target domain:

| Checkpoint | Clean target | Aug target |
|---|---:|---:|
| v1 200k (clean, best target) | 0.7177 | 0.6674 |
| v1 580k (clean, production) | 0.7095 | 0.6549 |
| **v2 300k (noise-robust + target-mix)** | **0.8390** | **0.7755** |

In-domain LibriSpeech cos sim dropped by ~0.015 — a reasonable price for
+0.12 on the actual deployment distribution.

### Selection rule: held-out eval, not training loss

Training loss kept improving long past the target-domain peak in v1. The
200k checkpoint was the actual best for target speakers, but listening tests
rated it "muffled/slurred" — cos sim is a useful proxy but not a substitute
for downstream listening. Two scripts make the eval reproducible:

```bash
# Single-checkpoint eval across four conditions (clean/aug × in-domain/target).
uv run python -m phone_extractor_trainer.eval \
    --ckpt outputs/phone_extractor_en_v2/checkpoint_00300000.pt

# Sweep every Nth checkpoint to find the peak before overfit.
uv run python -m phone_extractor_trainer.eval_sweep \
    --ckpt-glob 'outputs/phone_extractor_en_v2/checkpoint_*.pt' \
    --every 10000 --target-dir preprocessed/<your_dataset>_df --n-samples 64
```

Invariant rule going forward: **any change to the distillation recipe must be
evaluated against the previous baseline on all four conditions before being
adopted as `current.pt`.**

### Follow-up: the "+0.12 cos sim" win was a mirage (Experiment 4, 2026-06-02)

The v2 table above is **retained for the record but its conclusion is retracted.**
On fine-tunes, the v2 noise-robust + target-mix PhoneExtractor produced
mumbled, under-articulated speech ("big tongue") — it *dropped the
distinguishing features of words*. The higher target-domain cos sim came from
matching a **smeared** HuBERT-on-denoised-TTS target, not from better phonetic
fidelity.

A phonetic-contrast probe (`analysis/probe_phone_extractor.py`, measuring
`effective_rank` / `temporal_contrast` / frame self-similarity on clean clips)
revealed the collapse, ranking the variants by representation richness:

| PhoneExtractor | effective_rank ↑ | temporal_contrast ↑ |
|---|---:|---:|
| `jp_122_3000k` (**upstream**) | **18.8** | **0.42** |
| `en_clean_580k` (v1) | 12.9 | 0.25 |
| `en_nr_targetmix02_300k` (v2) | 8.6 | 0.19 |
| `en_clean_200k` (known muffled) | 7.9 | 0.22 |

The upstream `jp_122` is the **richest of all** — even on English clips — so
both our English-distilled variants were net regressions. A parallel F0 probe
(`analysis/probe_pitch_estimator.py`) showed the noise-robust pitch estimator
gave no benefit either. **Both extractors were reverted to upstream.**

Root causes of the phone collapse: (1) using **LPF (down to 2 kHz)** and
**formant-shift** as invariance targets against a clean teacher is ill-posed —
the student cannot recover removed fricative/formant energy, so it averages
phonemes together; (2) target-mixing against HuBERT features of OOD denoised
TTS trains the student to match smeared targets. The distillation trainer now
defaults `--aug-lpf-prob 0` and `--aug-formant-prob 0`; only phonetic-preserving
degradations (additive noise, mild reverb) should be used. **Selection rule
(corrected): cos sim is necessary but not sufficient — a new extractor must
beat upstream on the probe AND in a listening test before promotion.** Full
write-up in `project_context.md` lessons 2.6 / 2.7.

### Standard pipeline (DeepFilterNet path)

```bash
# 1. Denoise source TTS with DeepFilterNet3 (handles non-stationary artifacts)
uv run python denoise_sources_df.py --dataset <name>

# 2. Segment with permissive VAD
uv run python preprocess.py --dataset <name>_df --energy-threshold 35

# 3. Train Beatrice (uses the current.pt extractor symlinks -> upstream jp_122 / jp_104)
uv run python -m beatrice_trainer \
    -d preprocessed/<name>_df \
    -o outputs/<name>_df \
    -c assets/default_config.json
```

---

## Credits

- Original [Beatrice Trainer](https://huggingface.co/fierce-cats/beatrice-trainer) by Project Beatrice
- This fork maintained by [@leoliu1221](https://github.com/leoliu1221)

---

## License

MIT License. See [LICENSE](LICENSE) for details.
