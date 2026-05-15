# pitch_estimator_trainer

A standalone trainer for Beatrice 2's **`PitchEstimator`** network. Produces
checkpoints that are **drop-in replacements** for
`assets/pretrained/104_3_checkpoint_00300000.pt`.

The motivation: the shipped PitchEstimator may not cover the full pitch range
needed for cross-gender voice conversion (especially female→male). Retraining
on diverse pitch data improves coverage across both male and female ranges.

---

## Why retrain the pitch estimator?

The pitch estimator predicts F0 (fundamental frequency) from audio. It outputs
448 bins:
- **Bin 0** = unvoiced
- **Bins 1-447** = voiced pitches from **55 Hz (A1)** to **~1390 Hz (F6)**
- Resolution: **96 bins/octave** (8 bins/semitone)

Problems with the shipped model for cross-gender conversion:

1. **Training data bias**: The original was trained on data that may skew
   toward certain pitch ranges (e.g., more female speech than male).

2. **Octave errors**: When converting female→male, the pitch shift can be
   12+ semitones. If the estimator makes octave errors on edge cases, the
   converted voice sounds wrong.

3. **Low-pitch underrepresentation**: Deep male voices (80-120 Hz) may be
   underrepresented in the original training data.

---

## Training recipe

| Component | Choice | Why |
|---|---|---|
| **Student** | `PitchEstimator()` with default ctor | Architecturally identical to Beatrice's expected file |
| **Teacher** | pyworld DIO + StoneMask | Robust F0 estimation algorithm, same as Beatrice's internal F0 extraction |
| **Loss** | Cross-entropy over 448 bins | Classification over pitch bins with label smoothing |
| **Optimizer** | AdamW with cosine schedule + warmup | Standard for supervised training |
| **Grad clip** | 1.0 | |
| **AMP** | fp16 on CUDA | |
| **Crop length** | 4 s @ 16 kHz | Matches Beatrice's `wav_length` convention |

---

## Data layout

Any folder of diverse-pitch audio works — the dataset scans recursively for
`.wav .flac .mp3 .ogg .m4a .aac .opus`.

```
diverse_pitch_audio/
  vocalset/...           # singing voice, wide pitch range
  vctk/p225/...          # multi-speaker, male+female
  libritts_r/...         # audiobook, diverse speakers
  your_recordings/...    # any additional audio
```

**Recommended public corpora** (for diverse pitch coverage):

- **VocalSet** — singing voice dataset with extreme pitch ranges, essential
  for robust pitch estimation.
- **VCTK** — 110 speakers, balanced male/female, various accents.
- **LibriTTS-R** — high-quality audiobook recordings, diverse speakers.
- **CommonVoice** — crowd-sourced, very diverse demographics.

**Key principle**: Include roughly equal amounts of male and female speech,
plus singing voice data for extreme pitch coverage. Aim for **≥200 hours**
total, with good male/female balance.

---

## Usage

### 1. Train

From the repo root:

```bash
uv run python -m pitch_estimator_trainer.train \
    --data-dir /path/to/diverse_pitch_audio \
    --out-dir  outputs/pitch_estimator_v2 \
    --steps 300000 \
    --batch-size 32 \
    --num-workers 4
```

Optionally warm-start from the shipped checkpoint — converges faster and
retains learned features, just improves coverage:

```bash
uv run python -m pitch_estimator_trainer.train \
    --data-dir /path/to/diverse_pitch_audio \
    --out-dir  outputs/pitch_estimator_v2 \
    --init-from assets/pretrained/104_3_checkpoint_00300000.pt \
    --steps 200000
```

### 2. Watch progress

```bash
uv run python -m tensorboard.main --logdir outputs/pitch_estimator_v2
```

Key signals:
- `train/acc` — overall accuracy, should reach **>90%**
- `train/voiced_acc` — accuracy on voiced frames, should reach **>85%**
- `train/pitch_error_st` — mean pitch error in semitones, should be **<0.5 st**

### 3. Resume

```bash
uv run python -m pitch_estimator_trainer.train ... --resume
```

(Same flags as the original launch; reads `outputs/.../checkpoint_latest.pt`.)

### 4. Export to Beatrice format

```bash
uv run python -m pitch_estimator_trainer.export \
    outputs/pitch_estimator_v2/checkpoint_latest.pt \
    assets/pretrained/pitch_estimator_v2.pt
```

### 5. Use in Beatrice

Edit `assets/default_config.json` (or your dataset's `config.json`):

```json
"pitch_estimator_file": "assets/pretrained/pitch_estimator_v2.pt"
```

Then retrain your voice conversion model with the new pitch estimator.

---

## Makefile shortcuts

```bash
# Train (requires PITCH_DATA)
make pitch-train PITCH_DATA=/path/to/diverse_audio

# Warm-start from shipped checkpoint (default)
make pitch-train PITCH_DATA=/path/to/diverse_audio PITCH_INIT=assets/pretrained/104_3_checkpoint_00300000.pt

# Train from scratch (no warm-start)
make pitch-train PITCH_DATA=/path/to/diverse_audio PITCH_INIT=

# Resume interrupted training
make pitch-train PITCH_DATA=/path/to/diverse_audio RESUME=1

# Export to Beatrice format
make pitch-export

# Watch training progress
make pitch-tensorboard
```

---

## Troubleshooting

### Low accuracy on voiced frames

- **Cause**: Not enough diverse pitch data, or data is too clean (no noise).
- **Fix**: Add more singing voice data (VocalSet), or add noise augmentation.

### High pitch error

- **Cause**: Octave errors, often from ambiguous harmonics.
- **Fix**: Increase training steps, add more data with clear pitch.

### Poor female→male conversion after retraining

- **Cause**: Still not enough low-pitch (male) data.
- **Fix**: Ensure your training data has substantial deep male voice content
  (target: 80-150 Hz range well represented).

---

## Architecture notes

The `PitchEstimator` uses:
- **Instantaneous frequency features** from STFT (192 channels)
- **Autocorrelation-based features** (256 channels, YIN-style)
- **ConvNeXt backbone** (9 blocks, 192 channels)
- **448-way classification head**

The model is designed for low-latency inference (22.5ms total delay including
feature extraction), making it suitable for real-time voice conversion.
