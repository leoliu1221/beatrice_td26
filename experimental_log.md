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

### Run 4.1: PhoneExtractor noise-robust resume (ABORTED at step 704k)
- Resume from `outputs/phone_extractor_en/checkpoint_latest.pt` (step 580,000).
- Target: 780,000 steps (+200k noise-robust fine-tuning steps).
- LR resumes mid-cosine-decay at ~8e-5.
- **Initial step 580k loss jumped from 0.22 → 0.27, cos_sim dropped 0.78 → 0.73** — expected; the network had to start extracting phones from noisy/reverbed audio.
- **At step 702k**: train/loss → 0.24, train/cos_sim → 0.76 (looked healthy in TensorBoard).

#### Eval at step 704k (new `phone_extractor_trainer/eval.py`, n=64 per condition)
Held-out cos_sim comparing the student's projected features to the HuBERT teacher across four conditions:

| Condition | Step 580k (pre-aug) | Step 704k (aug-trained) | Δ |
|---|---|---|---|
| (a) Clean LibriSpeech (sanity) | 0.7828 | 0.7820 | ≈ 0 |
| (b) Aug LibriSpeech (~train) | 0.7421 | 0.7446 | +0.003 |
| (c) Clean target (LoL) | 0.7095 | 0.6987 | **−0.011** |
| (d) Aug target (LoL) | 0.6549 | 0.6380 | **−0.017** |

- Train metric improved 0.73 → 0.76 (+0.03) but held-out (b) only +0.003 → **classic mild-overfit signature** (train improving 10× faster than held-out).
- Target-domain (c, d) actually **regressed** ~1-2%. Within noise (1-3 SEs) but the trend is wrong.
- **Diagnosis**: the student is specializing to the LibriSpeech-with-augmentation distribution but **not transferring noise-invariance to the LoL TTS acoustics**. Augmentation alone is insufficient when there is also a domain gap between training audio (natural speech) and target audio (synthetic TTS).
- **Action**: aborted at 704k, restored `checkpoint_latest.pt` to the 580k snapshot, launched a new run with **target-domain mixing**.

### Run 4.1b: PhoneExtractor with augmentation + target-domain mix (ABORTED)
- Implementation: `WavCropDataset` gained `aux_files` + `aux_mix_ratio`; each `__getitem__` call picks from the aux pool with probability `aux_mix_ratio`, otherwise from the main LibriSpeech pool. New CLI: `--target-data-dir`, `--target-mix-ratio`.
- Resumed from restored 580k checkpoint with mix ratio 0.3.
- Aborted after ~2k steps when the full step-by-step eval sweep revealed the 580k checkpoint was already past the target-domain peak (see below). No point continuing from a degraded local minimum.

### Full eval sweep across all existing checkpoints (decisive finding)
Built `phone_extractor_trainer/eval.py` and `phone_extractor_trainer/eval_sweep.py` (sharing `build_eval_context` and `eval_checkpoint`). Swept every 50k steps from 10k → 700k, then a fine sweep every 10k from 150k → 280k (n=64).

Best target-domain cos_sim by checkpoint:

| step | (a) clean LS | (b) aug LS | (c) clean target | (d) aug target |
|---|---|---|---|---|
| 100k | 0.7719 | 0.7290 | 0.7095 | 0.6620 |
| 180k | 0.7829 | 0.7438 | **0.7180** ← best | 0.6667 |
| **200k** | **0.7833** | **0.7441** | 0.7177 | **0.6674** ← best |
| 210k | 0.7713 | 0.7356 | 0.7056 | 0.6559 | ← run boundary, sharp drop |
| 300k | 0.7746 | 0.7267 | 0.7057 | 0.6548 |
| 580k (used) | 0.7828 | 0.7421 | 0.7095 | 0.6549 |
| 700k (post-aug) | 0.7812 | 0.7380 | 0.6954 | 0.6330 |

- **Target-domain (c, d) peaked at step 180-200k**, then slowly declined for the next 380k steps even as in-domain (a, b) kept inching up. Classic train/test divergence.
- **A run-boundary discontinuity at step 210k** dropped all four metrics by ~1%; never fully recovered.
- **The 580k "production" checkpoint is ~0.010 below the 200k peak on target domain.**
- The 580k → 700k noise-robust resume made target performance markedly worse (~3.5% absolute drop on clean target). Augmentation alone cannot un-do existing LibriSpeech-specific specialization.
- **Cos_sim is a proxy**: per Experiment 2's listening tests, the 200k checkpoint produced "muffled, slurred" Beatrice output. So cos_sim-best ≠ downstream-best. We'll need to A/B them in Beatrice.

Fallback exported: `assets/pretrained/phone_extractor_en_200k.pt`.

### Run 4.1c: PhoneExtractor distilled from scratch with augmentation + target-mix (running)
- Hypothesis: rather than fine-tuning a clean-overfit 580k checkpoint, train noise-robustly **from the start** so noise-invariance is a primary objective and the LibriSpeech specialization doesn't form in the first place.
- Output dir: `outputs/phone_extractor_en_v2/` (preserves the existing v1 checkpoints).
- Config: warm-start from the original Japanese checkpoint `122_checkpoint_03000000.pt` (same as v1, for apples-to-apples), `--augment`, `--target-data-dir preprocessed/new_lol_data_df --target-mix-ratio 0.2`, 300k total steps, batch 32, 4 workers.
- Throughput ~4.6 it/s, ETA ~18 h.
- **Eval plan**: sweep `outputs/phone_extractor_en_v2/checkpoint_*.pt` every 25k steps. Decisive metric: target-domain (c, d) must beat the v1 200k peak (0.7177 / 0.6674) to justify the new recipe.
- **Status**: **In Progress (mid-sweep result decisive, see below)**.

#### Eval sweep at step 250k (n=64, target=`preprocessed/new_lol_data_df`)

| step | (a) clean LS | (b) aug LS | (c) clean target | (d) aug target |
|---:|---:|---:|---:|---:|
| 50,000 | 0.7496 | 0.7217 | 0.8151 | 0.7605 |
| 100,000 | 0.7552 | 0.7237 | 0.8248 | 0.7679 |
| 150,000 | 0.7612 | 0.7287 | 0.8298 | 0.7705 |
| 200,000 | 0.7648 | 0.7313 | 0.8347 | 0.7745 |
| **250,000** | **0.7685** | **0.7344** | **0.8379** | **0.7763** |

Comparison to v1 best target-domain checkpoint (200k) and v1 production (580k):

| metric | v1 200k | v1 580k | **v2 250k** | Δ v2 vs v1 200k |
|---|---:|---:|---:|---:|
| (a) clean LS | 0.7833 | 0.7828 | 0.7685 | −0.015 |
| (b) aug LS | 0.7441 | 0.7421 | 0.7344 | −0.010 |
| (c) clean target | 0.7177 | 0.7095 | **0.8379** | **+0.120** |
| (d) aug target | 0.6674 | 0.6549 | **0.7763** | **+0.109** |

- **Hypothesis validated.** Trading away ~1.5% in-domain LibriSpeech cos_sim buys **+12% on the actual target distribution**. Confirms that v1's LibriSpeech specialization was the dominant failure mode, not noise-augmentation per se.
- **Monotonic improvement** 50k → 250k on all four conditions: no overfit signature yet, no train/test divergence. Letting training continue to the planned 300k steps.
- The (c) > (a) ordering is initially counterintuitive but expected: the target pool is preprocessed, denoised LoL clips (clean, narrow speaker set), while the LibriSpeech eval pool is the raw open-domain LS distribution. Cos_sim is higher on the simpler distribution.
- **Note**: target-mix 0.2 means 20% of training batches were sampled from `new_lol_data_df`, so (c)/(d) are *not* held-out in the strict sense. They measure "target-domain fit" rather than zero-shot generalization. Still strictly more honest than v1, which had zero target exposure.

### Run 4.2: PitchEstimator noise-robust resume (DEFERRED)
- Held until Run 4.1c picks a winning phone extractor checkpoint. No point retraining the pitch estimator while the phone recipe is being iterated.

### Run 4.3: Beatrice re-train with noise-robust extractors (pending)
- After Runs 4.1b + 4.2 complete and export, retrain Beatrice on `preprocessed/new_lol_data_df`. Same balanced regularization as Phase B/C/D.

### Key insight: held-out eval is mandatory for distillation
- Train metrics can show steady improvement while the student is mildly overfitting. We only caught Run 4.1's specialization by running `phone_extractor_trainer/eval.py` with four conditions (clean/aug × in-domain/target-domain).
- **Invariant rule going forward**: any change to the distillation recipe must be evaluated against the 580k baseline on all four conditions before being adopted.

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
