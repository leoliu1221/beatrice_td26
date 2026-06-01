# Project Context & Memories

This file serves as a permanent context anchor for developers and AI coding assistants working on the Beatrice Trainer Fork repository. It details key invariant rules, design decisions, and system constraints.

---

## 1. Quick Reference: Best Pretrained Assets

Assets live under `assets/pretrained/<module>/` as **immutable** variant files. Each module has a `current.pt` symlink pointing to the recommended variant. See `assets/pretrained/README.md` and `manifest.json` for the registry, naming convention, and how to promote a new variant.

| Module | Current variant (resolved through symlink) | Status |
|---|---|---|
| **PhoneExtractor** | `phone_extractor/en_nr_targetmix02_300k.pt` | Noise-robust + 20% target-mix, 300k from scratch on LibriSpeech-100. Beats `en_clean_580k` by **+0.13 on clean target / +0.12 on aug target** held-out cos_sim. |
| **PitchEstimator** | `pitch_estimator/vctk_nr_300k.pt` | Noise-robust, 300k from scratch on VCTK (109 speakers). `formant_shift_probability` forced to 0 to protect pyworld F0 labels. |
| **Vocoder** | `vocoder/libritts_r_200_2750k.pt.gz` | Upstream LibriTTS-R checkpoint (warm-start for converter). |

In configs, **reference the symlink** for default behavior:
```json
"phone_extractor_file": "assets/pretrained/phone_extractor/current.pt",
"pitch_estimator_file": "assets/pretrained/pitch_estimator/current.pt",
"pretrained_file":      "assets/pretrained/vocoder/current.pt.gz"
```

For A/B experiments where you need to pin a specific variant for reproducibility (so a later `ln -sfn` doesn't retroactively rewrite history), point at the variant file directly — see `assets/configs/ab_ap{0,100}_v2.json` for the pattern.

---

## 2. Crucial Lessons Learned (Do Not Repeat Past Mistakes)

### 2.1 The Small Dataset Noise Trap
- **Mistake**: Lowering `phone_noise_ratio` (e.g., to `0.1`) and `floor_noise_level` (to `1e-5`) during fine-tuning on small datasets (`new_lol_data`) to "remove static".
- **Finding**: This causes massive overfitting. The model learns to reproduce quiet room noise/static present in the target speech gaps because it lacks the regularization to ignore it.
- **Invariant Rule**: Always use **`phone_noise_ratio: 0.5`** and maintain standard augmentation probabilities (`reverb: 0.5`, `formant_shift: 0.5`) during voice fine-tuning. Control static by moderately adjusting `floor_noise_level` to `1e-4` (-80dB) instead of removing the noise-ratio regularization.

### 2.2 PhoneExtractor Step Counts
- **Mistake** (v1, clean-only distillation): Using a 200,000 step English PhoneExtractor.
- **Finding (v1)**: Results in slurred, muffled pronunciations where voices sound like they are "averaged together". v1 required **500k+ steps** to resolve clear pronunciations (production was 580k, ~78% cos_sim vs HuBERT on clean LibriSpeech).
- **v2 update (noise-robust + target-mix)**: With augmentation + 20% target-domain mix from the start, the model converges cleanly by ~270-300k steps with **no overfit signature** and beats v1 580k by ~12% on target-domain cos_sim. Do NOT carry over the "use 580k" rule; v2 should be selected by held-out target-domain eval, not step count.
- **Invariant Rule**: For any phone-extractor recipe change, run `phone_extractor_trainer/eval_sweep.py` across checkpoints and pick the step that maximizes target-domain (c, d) without regressing in-domain (a, b). Never select by training loss alone.

### 2.3 Audio Preprocessing for Main Trainer
- **Constraint**: The main `beatrice_trainer`'s VQ codebook builder runs fully in GPU memory at startup. Putting very long audio files (e.g., raw 25-minute recordings) directly into `inputs/` causes immediate **CUDA Out of Memory (OOM)** errors when the network attempts to pass the entire waveform sequence.
- **Invariant Rule**: Always run `make DATASET=your_dataset preprocess` first. This segments long audio files into clean 4-15 second clips inside `preprocessed/` using `auditok`.

### 2.5 Feature Extractors Must Be Trained With the Same Augmentation As Beatrice
- **Failure mode**: Beatrice's main trainer calls `augment_audio()` (noise, reverb, LPF, formant shift) on the input wav before passing it to PhoneExtractor and PitchEstimator. If those extractors were distilled on **clean** audio only, they produce unstable features under augmentation, and the main converter learns to reproduce that instability as noise during conversion.
- **Symptom**: "Noise when I talk" — the converted audio has artifacts overlaid on the user's speech even with a clean denoised dataset.
- **Fix**: train the extractors with the same `augment_audio()` pipeline. Student sees noisy input, target (HuBERT features / pyworld F0) is computed from clean input. This is noise-robust / consistency distillation.
- **How to invoke (preferred — from scratch with target-mix, the v2 recipe)**:
  ```bash
  # Phone extractor: from scratch with augmentation + 20% target-domain mix
  uv run python -m phone_extractor_trainer.train \
      --data-dir datasets/librispeech/LibriSpeech/train-clean-100 \
      --out-dir outputs/phone_extractor_en_v2 \
      --steps 300000 --batch-size 32 --num-workers 4 \
      --init-from assets/pretrained/122_checkpoint_03000000.pt \
      --augment \
      --target-data-dir preprocessed/<your_dataset>_df --target-mix-ratio 0.2
  uv run python -m phone_extractor_trainer.export \
      outputs/phone_extractor_en_v2/checkpoint_00300000.pt \
      assets/pretrained/phone_extractor_en_v2.pt

  # Pitch estimator: from scratch with augmentation on VCTK
  make pitch-train AUGMENT=1 PITCH_OUT=outputs/pitch_estimator_v2 \
      PITCH_INIT=assets/pretrained/104_3_checkpoint_00300000.pt
  make pitch-export PITCH_OUT=outputs/pitch_estimator_v2 \
      PITCH_EXPORT_OUT=assets/pretrained/pitch_estimator_v2.pt
  ```
- **Legacy (resume) recipe — NOT recommended**: fine-tuning a clean-distilled 580k checkpoint with `--augment` regressed target-domain cos_sim by ~3.5%. The clean-distillation specialization is sticky; train from scratch instead.
- **Held-out eval is mandatory after any recipe change**:
  ```bash
  uv run python -m phone_extractor_trainer.eval_sweep \
      --ckpt-glob 'outputs/phone_extractor_en_v2/checkpoint_*.pt' \
      --every 10000 --target-dir preprocessed/<your_dataset>_df --n-samples 64
  ```
- **Files involved**: `distill_augment.py`, `phone_extractor_trainer/{data,train,eval,eval_sweep}.py`, `pitch_estimator_trainer/{data,train}.py`, Makefile (`AUGMENT=1` toggle).
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
- **Current state (Run 4 series, see `experimental_log.md`)**:
  - Phase 1 ✅ **PhoneExtractor v2** noise-robust from-scratch training done. Converged @ 300k. Exported to `assets/pretrained/phone_extractor_en_v2.pt`. Held-out target-domain cos_sim: (c) 0.8390 / (d) 0.7755 vs v1 580k 0.7095 / 0.6549.
  - Phase 2 ✅ **PitchEstimator v2** noise-robust from-scratch training done @ 300k on VCTK. Exported to `assets/pretrained/pitch_estimator_v2.pt`.
  - Phase 3 🔄 **Beatrice A/B on `grad_weight_ap`** (Run 4.3). Two sequential 60k-step trainings on `preprocessed/new_lol_data_df` with the v2 extractor pair: one with `grad_weight_ap=0` (upstream default after PR #1), one with `grad_weight_ap=100` (legacy). Driven by `run_ab_grad_weight_ap.sh`. Listen to `outputs/ab_ap*_v2/test/` at end of each run to pick winner on voiced-speech buzz vs. fricative clarity tradeoff.
