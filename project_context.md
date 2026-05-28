# Project Context & Memories

This file serves as a permanent context anchor for developers and AI coding assistants working on the Beatrice Trainer Fork repository. It details key invariant rules, design decisions, and system constraints.

---

## 1. Quick Reference: Best Pretrained Assets

| Model Module | Best Checkpoint File | Status / Origin | Usage in Config |
|--------------|----------------------|-----------------|-----------------|
| **PhoneExtractor** | `assets/pretrained/phone_extractor_en.pt` | Converged @ 580,000 steps on LibriSpeech-100. | `"phone_extractor_file"` |
| **PitchEstimator** | `assets/pretrained/pitch_estimator_v2.pt` | Converged @ 300,000 steps on VCTK (109 speakers). | `"pitch_estimator_file"` |

---

## 2. Crucial Lessons Learned (Do Not Repeat Past Mistakes)

### 2.1 The Small Dataset Noise Trap
- **Mistake**: Lowering `phone_noise_ratio` (e.g., to `0.1`) and `floor_noise_level` (to `1e-5`) during fine-tuning on small datasets (`new_lol_data`) to "remove static".
- **Finding**: This causes massive overfitting. The model learns to reproduce quiet room noise/static present in the target speech gaps because it lacks the regularization to ignore it.
- **Invariant Rule**: Always use **`phone_noise_ratio: 0.5`** and maintain standard augmentation probabilities (`reverb: 0.5`, `formant_shift: 0.5`) during voice fine-tuning. Control static by moderately adjusting `floor_noise_level` to `1e-4` (-80dB) instead of removing the noise-ratio regularization.

### 2.2 PhoneExtractor Step Counts
- **Mistake**: Using a 200,000 step English PhoneExtractor.
- **Finding**: Results in slurred, muffled pronunciations where voices sound like they are "averaged together".
- **Invariant Rule**: The English PhoneExtractor bottleneck requires at least **500k+ steps** (our best run is 580k steps, yielding `~78%` cosine similarity to the English HuBERT teacher) to cleanly resolve individual voice characters and clear pronunciations.

### 2.3 Audio Preprocessing for Main Trainer
- **Constraint**: The main `beatrice_trainer`'s VQ codebook builder runs fully in GPU memory at startup. Putting very long audio files (e.g., raw 25-minute recordings) directly into `inputs/` causes immediate **CUDA Out of Memory (OOM)** errors when the network attempts to pass the entire waveform sequence.
- **Invariant Rule**: Always run `make DATASET=your_dataset preprocess` first. This segments long audio files into clean 4-15 second clips inside `preprocessed/` using `auditok`.

### 2.5 Feature Extractors Must Be Trained With the Same Augmentation As Beatrice
- **Failure mode**: Beatrice's main trainer calls `augment_audio()` (noise, reverb, LPF, formant shift) on the input wav before passing it to PhoneExtractor and PitchEstimator. If those extractors were distilled on **clean** audio only, they produce unstable features under augmentation, and the main converter learns to reproduce that instability as noise during conversion.
- **Symptom**: "Noise when I talk" — the converted audio has artifacts overlaid on the user's speech even with a clean denoised dataset.
- **Fix**: train the extractors with the same `augment_audio()` pipeline. Student sees noisy input, target (HuBERT features / pyworld F0) is computed from clean input. This is noise-robust / consistency distillation.
- **How to invoke**:
  ```bash
  # Phone extractor: resume from previous checkpoint, target +200k noise-robust steps
  make phone-train RESUME=1 AUGMENT=1 PHONE_STEPS=780000
  make phone-export

  # Pitch estimator: same idea
  make pitch-train RESUME=1 AUGMENT=1 PITCH_STEPS=500000
  make pitch-export
  ```
- **Files involved**: `distill_augment.py`, `phone_extractor_trainer/{data,train}.py`, `pitch_estimator_trainer/{data,train}.py`, Makefile (`AUGMENT=1` toggle).
- **Pitch caveat**: in `PitchDataset` the formant shift probability is force-set to 0 because Beatrice's formant shift uses spectral-envelope warp + resample, which can perturb pyworld F0 labels and silently corrupt training. All other augmentations (noise/reverb/LPF) are kept.

### 2.4 TTS Source Artifacts → Deep-Pitch Buzz
- **Finding**: Even with balanced regularization, the model faithfully reproduces TTS artifacts (buzz/aliasing) that are baked into the source recordings — especially audible on low-pitch / male voices.
- **Two denoiser options** (use either, not both):
  - `denoise_sources.py` (noisereduce, spectral subtraction). Quick, no deps. Good for stationary background hum. **Misses phoneme-correlated artifacts.**
  - `denoise_sources_df.py` (**DeepFilterNet3**, deep learning). Requires Rust toolchain. Handles non-stationary buzz / vocoder ringing / transients. **Preferred for TTS sources.**
- **Invariant Rule**: When denoising, always pair with a permissive VAD: `uv run python preprocess.py --dataset <foo>_df --energy-threshold 35`. Denoised audio has a lower noise floor and the default threshold (45) over-segments speech.
- **DeepFilterNet gotchas**:
  - Needs Rust: `curl https://sh.rustup.rs ... | sh -s -- -y && source $HOME/.cargo/env`
  - Long files crash cuDNN GRU — script chunks audio at 30s.
  - Downgrades numpy to 1.26 as a transitive dep; Beatrice still imports OK.

---

## 3. Directory Layout Conventions

- `inputs/<dataset_name>/<speaker_name>/`: Raw source audio files.
- `inputs/<dataset_name>_denoised/<speaker_name>/`: Source audio denoised with noisereduce.
- `inputs/<dataset_name>_df/<speaker_name>/`: Source audio denoised with DeepFilterNet3.
- `preprocessed/<dataset_name>/<speaker_name>/`: Segmented clips outputted by `preprocess.py` (24kHz, mono, PCM_16).
- `outputs/<dataset_name>/`: Checkpoints and exported paraphernalia voice packages.

---

## 4. Standard Pipeline (DeepFilterNet Path, Preferred)

```bash
# 1. Denoise source TTS with DeepFilterNet3 (handles non-stationary artifacts)
uv run python denoise_sources_df.py --dataset <name>

# 2. Segment with permissive VAD
uv run python preprocess.py --dataset <name>_df --energy-threshold 35

# 3. Train
uv run python -m beatrice_trainer \
  -d preprocessed/<name>_df \
  -o outputs/<name>_df \
  -c assets/default_config.json
```

---

## 5. Current Work & Goals
- **Active Task**: Fine-tuning Beatrice voice models for `new_lol_data` (6 speakers: sion, teemo, demacia_male, noxus_male, yordle_female, yordle_male).
- **Setup**: Training is running on **denoised** source data with the 580k English PhoneExtractor, the V2 VCTK Pitch Estimator, and a balanced regularization config (60,000 steps).
