# Training Log & Experimental Findings

A running log of experiments, hyperparameter tuning, data scaling, and model behavior for the Beatrice Trainer fork.

---

## Experiment 4: Noise-Robust Feature Extractor Re-Distillation
**Date:** May 28, 2026  
**Objective:** Fix the **train/test mismatch** between the feature extractors and Beatrice's main trainer that causes "noise during speech" at inference.

### Root cause
- Beatrice's main trainer (`beatrice_trainer/__main__.py:3499`) calls `augment_audio()` (noise SNR 20-45 dB, IR reverb, LPF, formant shift) on the input wav before passing it to the PhoneExtractor and PitchEstimator.
- Both trainers (`phone_extractor_trainer/data.py`, `pitch_estimator_trainer/data.py`) originally trained on **clean** audio only — no augmentation.
- Result: the extractors saw never-seen-before noisy/reverberant inputs during Beatrice training and at inference. Their features became unstable, and the main converter learned to reproduce the instability as noise.

### Fix: consistency / noise-robust distillation
- **PhoneExtractor**: teacher (HuBERT) sees clean wav, student sees `augment_audio(clean)`. Loss = `1 - cos(proj(student), teacher) + λ·MSE(...)`. Forces student to be noise-invariant.
- **PitchEstimator**: F0 label is computed from clean wav (pyworld DIO+StoneMask), but the model only sees `augment_audio(clean)`. Note: `formant_shift_probability` is forced to 0 for pitch training because Beatrice's formant shift uses spectral-envelope warp + resample, which can subtly perturb F0 labels.

### Implementation
- New shared helper `distill_augment.py` wrapping Beatrice's `augment_audio()` with default hyperparams from `assets/default_config.json`.
- `WavCropDataset` (phone) gained `noise_files`, `ir_files`, `aug_kwargs` params; when set, `__getitem__` returns `(clean, noisy)` instead of just `clean`. Backward-compatible.
- `PitchDataset` gained the same params; when set, `__getitem__` returns `(noisy_wav, clean_pitch_bins)`.
- `phone_extractor_trainer/train.py` handles the tuple: teacher forward on `clean_wav`, student forward on `student_input`.
- Both trainers now expose `--augment`, `--noise-dir`, `--ir-dir` CLI flags.
- Makefile gained `AUGMENT=1` toggle: `make phone-train RESUME=1 AUGMENT=1 PHONE_STEPS=780000` and the equivalent for pitch.

### Run 4.1: PhoneExtractor noise-robust resume (running)
- Resume from `outputs/phone_extractor_en/checkpoint_latest.pt` (step 580,000).
- Target: 780,000 steps (+200k noise-robust fine-tuning steps).
- LR resumes mid-cosine-decay at ~8e-5.
- **Initial step 580k loss jumped from 0.22 → 0.27, cos_sim dropped 0.78 → 0.73** — expected; the network now has to extract phones from noisy/reverbed audio.
- Throughput: ~6 it/s → ~9 h ETA.
- **Result**: **In Progress**.

### Run 4.2: PitchEstimator noise-robust resume (pending)
- Resume from `outputs/pitch_estimator_v2/checkpoint_latest.pt` (step 300,000).
- Target: 500,000 steps (+200k noise-robust steps).
- Scheduled to start after Run 4.1 completes (single GPU box).

---

## Experiment 3: Beatrice Fine-Tuning - Robustness vs. Audio Clarity (`new_lol_data`)
**Date:** May 26, 2026  
**Objective:** Fine-tune the main Beatrice conversion network on `new_lol_data` (6 League speakers) with the newly-retrained English PhoneExtractor and PitchEstimator.

### Phase A: Low Augmentation / Low Noise Floor
* **Configuration**: `phone_noise_ratio: 0.1`, `floor_noise_level: 1e-5`, `augmentation_reverb_probability: 0.1`, `augmentation_formant_shift_probability: 0.2`
* **Observations**:
  * Output audio was extremely dry and free of synthesized background hiss.
  * **Critical Issue**: The model severely overfit to the small dataset's target voices. Because silent gaps in the raw audio still contained minor physical room/line noise, the network learned to reconstruct the static as a core speaker trait. The dry configuration lacked the generalization regularization to overcome this.
  * **Result**: **Unsuccessful**. Clear but carried reconstructed static on any quiet parts.

### Phase B: Balanced Regularized Fine-Tuning
* **Configuration**: `phone_noise_ratio: 0.5`, `floor_noise_level: 1e-4` (-80dB), `augmentation_reverb_probability: 0.5`, `augmentation_formant_shift_probability: 0.5`
* **Theory**: Keep the original robust augmentation probabilities so the model has enough regularization to handle the small dataset size without overfitting. Reduce the static footprint by dropping `floor_noise_level` from `1e-3` (too loud) to `1e-4` (quiet but present for numerical gradient stability).
* **Result**: **Partial success**. Words became intelligible but **deep-pitch voices (Sion, demacia_male, noxus_male) still carried noticeable buzz/static** which traced to TTS artifacts baked into the source audio. Confirmed root cause: the model was faithfully reproducing source-audio TTS buzz as part of speaker identity.

### Phase C: Source TTS Denoising via spectral subtraction (`noisereduce`)
* **Approach**: Address the artifacts at the **source** rather than via training-time regularization. Created `denoise_sources.py` using `noisereduce` (spectral subtraction, stationary mode, `prop_decrease=0.6`). This produces `inputs/new_lol_data_denoised/` mirroring the original layout. Preprocess uses a permissive VAD (`--energy-threshold 35` instead of 45) because denoised audio has lower noise-floor energy that would otherwise cause auditok to over-segment.
* **Key data-flow change**: `preprocess.py` gained `--energy-threshold`, `--min-dur`, `--max-dur`, `--max-silence` CLI flags (backwards compatible).
* **Clip count comparison**:
  | Speaker | Original | Denoised (et=35) |
  |---|---|---|
  | sion | 51 | 51 |
  | teemo | 38 | 35 |
  | demacia_male | 47 | **52** |
  | noxus_male | 25 | **30** |
  | yordle_female | 23 | 24 |
  | yordle_male | 12 | 13 |
  | **Total** | 196 | **205** |
* **Config**: Same balanced regularization as Phase B (`phone_noise_ratio: 0.5`, `floor_noise_level: 1e-4`, all augmentation probs default).
* **Result**: **Partial success**. Buzz reduced but **non-stationary noise during active speech remained** ("noise when I talk, sound still not very clear"). `noisereduce` only removes stationary background hum; phoneme-correlated TTS artifacts (vocoder ringing, transient buzz) survived.

### Lesson: Denoising tradeoff with VAD
* `noisereduce` at `prop_decrease=0.85` (aggressive) dropped clip count from 196 → 97 because the speech RMS energy dropped below auditok's default threshold of 45.
* Sweet spot: `prop_decrease=0.6` (moderate denoising) + `--energy-threshold 35` (permissive VAD) → matches/exceeds original clip count.
* If you push denoising harder, lower the VAD threshold proportionally; otherwise the deepest voices will suffer the most (because their fundamental energy is lower to begin with).

### Phase D: Source TTS Denoising via DeepFilterNet3 (Current Run)
* **Approach**: Replace `noisereduce` with **DeepFilterNet3** (deep-learning denoiser) which handles **non-stationary** artifacts: phoneme-correlated buzz, vocoder ringing, transient clicks that survived spectral subtraction.
* **Setup notes**:
  - DF requires Rust toolchain to build `deepfilterlib`. Install with `curl https://sh.rustup.rs ... | sh -s -- -y && source $HOME/.cargo/env`.
  - DF runs natively at 48 kHz; sources at 24 kHz are upsampled before denoising.
  - **Must chunk long audio (30s chunks)** — feeding a 25-minute file at once crashes cuDNN GRU with `CUDNN_STATUS_NOT_SUPPORTED`.
  - Script: `denoise_sources_df.py` → outputs `inputs/<dataset>_df/`.
  - numpy is downgraded from 2.4 → 1.26 as a deepfilterlib dep; Beatrice still imports cleanly.
* **Clip count comparison** (preprocess with `--energy-threshold 35`):
  | Speaker | Original | noisereduce | DeepFilterNet3 |
  |---|---|---|---|
  | sion | 51 | 51 | 39 |
  | teemo | 38 | 35 | 36 |
  | demacia_male | 47 | 52 | 51 |
  | noxus_male | 25 | 30 | **40** |
  | yordle_female | 23 | 24 | 25 |
  | yordle_male | 12 | 13 | **15** |
  | **Total** | 196 | 205 | **206** |
* **Notable**: DF picked up many more usable segments for the deepest voice (noxus_male: 25→40) — the non-stationary buzz removal recovered transitions that were previously sub-threshold for VAD.
* **Config**: Same balanced regularization as Phase B/C.
* **Result**: **In Progress (Active)**. Training on `preprocessed/new_lol_data_df` for 60k steps at ~2.9 it/s.

---

## Experiment 2: English PhoneExtractor Scale-Up (`phone_extractor_trainer`)
**Date:** May 14 – May 19, 2026  
**Objective:** Resolve Japanese phonetic bias in English voice conversion by distilling a 128-dimensional student model from a frozen 768-dimensional English HuBERT Teacher (`HUBERT_BASE` Layer 9).

### Run 2.1: Preliminary Short Run (200k steps)
* **Configuration**: 200,000 steps on LibriSpeech-100.
* **Observations**:
  * **Issues**: Sub-optimal phonetic feature map. In Beatrice, this resulted in muffled voice conversion outputs where words felt like they were "averaged together" or slurred. Voice identities were largely unrecognizable.
  * **Cos-Similarity**: ~0.78 (but only visited ~200k steps).
  * **Result**: **Unusable**. Proved that the 128-dim bottleneck requires a very long training schedule to converge on a highly distinct representation space.

### Run 2.2: Deep Distillation Scale-Up (580k steps)
* **Configuration**: Resumed from 200k step checkpoint up to **581,950 steps** on LibriSpeech-100 clean dataset.
* **Observations**:
  * Cosine similarity to the English HuBERT teacher stabilized cleanly at **~0.78**.
  * Distillation loss dropped and settled around **0.22**.
  * **Result**: **Successful**. This model provides significantly more distinct, robust phonetic features, solving the slurring/averaging artifacts observed at 200k steps.

---

## Experiment 1: PitchEstimator V2 Training (`pitch_estimator_trainer`)
**Date:** May 14 – May 16, 2026  
**Objective:** Overcome cross-gender conversion errors (especially high-to-low pitch tracking) by training on the full VCTK dataset.

### Summary
* **Configuration**: Trained supervised pitch estimator using PyWorld ground-truth bins over 109 diverse speakers. Completed full 300,000 steps with an optimized batch size of 256.
* **Result**: **Successful**. Exported as `pitch_estimator_v2.pt`. Resolved the tracking failures where pitch would drop out or fail during intense gender-swapping conversions.
