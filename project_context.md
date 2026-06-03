# Project Context & Memories

This file serves as a permanent context anchor for developers and AI coding assistants working on the Beatrice Trainer Fork repository. It details key invariant rules, design decisions, and system constraints.

---

## 1. Quick Reference: Best Pretrained Assets

Assets live under `assets/pretrained/<module>/` as **immutable** variant files. Each module has a `current.pt` symlink pointing to the recommended variant. See `assets/pretrained/README.md` and `manifest.json` for the registry, naming convention, and how to promote a new variant.

| Module | Current variant (resolved through symlink) | Status |
|---|---|---|
| **PhoneExtractor** | `phone_extractor/jp_122_3000k.pt` (**upstream**) | Reverted to upstream 2026-06-02. Our English-distilled variants (`en_clean_580k`, `en_nr_targetmix02_300k`) **regressed** phonetic richness — see lesson 2.6. The probe (`analysis/probe_phone_extractor.py`) ranks jp_122 the richest of all variants (effective_rank 18.8) even on English clips. |
| **PitchEstimator** | `pitch_estimator/jp_104_3_300k.pt` (**upstream**) | Reverted to upstream 2026-06-02. The noise-robust `vctk_nr_300k` showed **no benefit** (worst on noisy input) in the probe (`analysis/probe_pitch_estimator.py`) — see lesson 2.7. |
| **Vocoder** | `vocoder/libritts_r_200_2750k.pt.gz` | Upstream LibriTTS-R checkpoint (warm-start for converter). |

> **2026-06-02 reversion summary:** after a "big tongue" (mumbled/under-articulated) regression was traced to our retrained extractors, the **full upstream stack** (`jp_122` phone + `jp_104` pitch) was restored as `current.pt`. This is the empirically "worked great" baseline. Do not re-promote a retrained extractor without passing the probe **and** a listening test. Lessons 2.6 / 2.7 below.

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
- **v2 update (noise-robust + target-mix)** — ⚠️ **RETRACTED, see lesson 2.6.** v2 converged to a higher target-domain *cos_sim* (~+0.12) but that metric was misleading: the higher cos_sim came from matching a smeared target, and v2 actually **collapsed phonetic resolution** ("big tongue"). The cos_sim win did not translate to audible quality. Both v1 and v2 were ultimately net regressions from the upstream `jp_122`.
- **Invariant Rule (corrected)**: Cos_sim / `eval_sweep.py` is **not sufficient** to select a PhoneExtractor — it does not measure phonetic separability. Select by `analysis/probe_phone_extractor.py` (effective_rank / temporal_contrast) **and** a listening test, and the candidate must beat the current `jp_122` baseline on both. Never select by training loss or cos_sim alone.

### 2.3 Audio Preprocessing for Main Trainer
- **Constraint**: The main `beatrice_trainer`'s VQ codebook builder runs fully in GPU memory at startup. Putting very long audio files (e.g., raw 25-minute recordings) directly into `inputs/` causes immediate **CUDA Out of Memory (OOM)** errors when the network attempts to pass the entire waveform sequence.
- **Invariant Rule**: Always run `make DATASET=your_dataset preprocess` first. This segments long audio files into clean 4-15 second clips inside `preprocessed/` using `auditok`.

### 2.5 Feature Extractors Must Be Trained With the Same Augmentation As Beatrice
- **⚠️ SUPERSEDED IN PART BY LESSON 2.6 (2026-06-02).** The "from scratch + augment + target-mix" v2 recipe below **collapsed the PhoneExtractor's phonetic resolution** and produced "big tongue" mumbled conversions. Do NOT use LPF or formant-shift augmentation, and do NOT use target-mix, for PhoneExtractor distillation. Read lesson 2.6 first. The augment-with-noise idea is sound *only* for degradations that preserve phonetic information (additive noise, mild reverb).
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

### 2.6 "Big Tongue": PhoneExtractor Phonetic Collapse, and Why Cos-Sim Lies (2026-06-02)
- **Symptom**: converted speech is mumbled/under-articulated — "big tongue" — *dropping the distinguishing features of each word*. Persists at every checkpoint (5k → 50k → 60k) and is **identical across unrelated A/B arms** (grad_weight_ap=0 vs 100), which proves the cause is upstream of the converter: a shared frozen input, i.e. the **PhoneExtractor**.
- **Root cause**: the v2 noise-robust + target-mix distillation recipe (`en_nr_targetmix02_300k.pt`) **collapsed the phone representation**. Two compounding causes:
  1. **Destructive augmentation as an invariance target.** `augment_audio()` includes LPF down to **2 kHz** and ±3-semitone formant shift. Asking the student to map a 2 kHz-bandlimited (or formant-warped) signal to the *clean* full-band HuBERT target is ill-posed — fricative energy (/s/, /ʃ/, /f/ at 4-8 kHz) cannot be recovered — so SGD settles on a **phoneme-agnostic average**. Because weights are shared, this smoothing applies even to clean inference inputs.
  2. **Target-mix against an unreliable teacher.** 20% of crops were denoised LoL TTS, but HuBERT (trained on clean natural speech) produces smeared features on OOD denoised TTS. The student learned to match smeared targets.
- **Why we didn't catch it**: we selected v2 on **cosine similarity to HuBERT** (target-domain 0.71→0.84). Cos-sim measures *consistency* with the (smeared) target, **not phonetic separability**. This is the SAME trap as lesson 2.2 (the cos-sim-best 200k checkpoint sounded muffled). **Cos-sim to HuBERT must never again be the sole/primary selector for a PhoneExtractor.**
- **The diagnostic that actually works** — `analysis/probe_phone_extractor.py`. Feeds clean clips through `units()` and measures phonetic contrast:
  - `effective_rank` (participation ratio of feature covariance) — richness of representation
  - `temporal_contrast` (frame-to-frame delta) — articulation sharpness
  - `mean_pairwise_cos` (frame self-similarity) — collapse indicator (higher = more collapsed)
  - Measured ranking (effective rank): `jp_122` 18.8 > **`en_clean_580k` 12.9** > `en_nr_targetmix02_300k` 8.6 ≈ `en_clean_200k` 7.9 (known-muffled). The promoted v2 was nearly as collapsed as the known-bad 200k.
- **Fix applied (2026-06-02)**: set `assets/pretrained/phone_extractor/current.pt` → **`jp_122_3000k.pt`** (the upstream Beatrice original). The probe shows jp_122 is the richest extractor of all (effective_rank 18.8 vs en_clean_580k 12.9 vs en_nr 8.6) **even on English clips**, and it is the empirical "worked great" baseline. **Conclusion: the entire English-distillation effort (v1 en_clean_580k, v2 en_nr_targetmix) was a net regression from upstream — do not assume an English-distilled extractor beats jp_122 without a probe + listening test that proves it.** This supersedes lesson 2.2's claim that v2 "beats v1". The grad_weight_ap A/B (`outputs/ab_ap*_v2`) is **invalid** (both arms used the regressed v2 extractor) and must be re-run.
- **If a future English distillation is attempted**: warm-start from jp_122, train long (≥580k), use only noise+reverb augmentation, and it must **beat jp_122's probe effective_rank AND win a listening test** before promotion. Until then, jp_122 stays current.
- **Invariant Rules going forward**:
  1. **Select PhoneExtractors by `analysis/probe_phone_extractor.py` (effective_rank / temporal_contrast), not by cos-sim.** A new variant must not regress effective_rank vs the current one.
  2. For noise-robust phone distillation, **only use degradations that preserve phonetic information** (additive noise at SNR ≥ 20 dB, mild reverb). **Never LPF or formant-shift** with a clean teacher.
  3. **Do not target-mix the PhoneExtractor** against HuBERT features of OOD/denoised TTS — the teacher is unreliable there.
  4. Pitch estimator is less affected (formant shift already forced to 0, and F0 ≠ phonetics) but should still be sanity-checked before trusting — see lesson 2.7.

### 2.7 PitchEstimator: Noise-Robust Retrain Gave No Benefit; Upstream jp_104 Restored (2026-06-02)
- **Context**: alongside the phone extractor, the pitch estimator had been retrained noise-robust on VCTK (`vctk_nr_300k.pt`). After reverting the phone extractor to upstream, we verified whether the pitch retrain was worth keeping **before** committing it to the upstream-stack retrain.
- **Diagnostic** — `analysis/probe_pitch_estimator.py`. Scores each estimator against the pyworld DIO+StoneMask F0 (the training ground truth) on the same clips, both clean and augmented. Metrics on co-voiced frames: `voiced_err_st` (semitone error ↓), `gross_err_rate` (octave/halving errors ↓), `corr` (↑). Note: `voicing_acc` is **not informative** here — these estimators predict a pitch on ~100% of frames by design (Beatrice gates unvoiced via the separate energy channel), so voicing_acc just equals the reference voiced fraction.
- **Result (4 clips, vs pyworld GT)**:
  | input | estimator | voiced_err_st ↓ | gross_err ↓ | corr ↑ |
  |---|---|---|---|---|
  | clean | `jp_104` (upstream) | 0.497 | 6.3% | 0.925 |
  | clean | `vctk_clean_300k` | **0.442** | **5.1%** | 0.890 |
  | clean | `vctk_nr_300k` | 0.551 | 7.9% | **0.947** |
  | noisy | `jp_104` (upstream) | **0.540** | **8.0%** | 0.947 |
  | noisy | `vctk_clean_300k` | 0.677 | 9.3% | 0.908 |
  | noisy | `vctk_nr_300k` | 0.766 | 9.4% | 0.925 |
- **Findings**:
  - All three are close and all are good (sub-semitone error, corr 0.89–0.95). **Pitch is not where the quality problem lives** — consistent with "big tongue" being phonetic, not pitch.
  - The noise-robust `vctk_nr` showed **no benefit and was actually the *worst* on noisy input** (0.77 st vs jp_104's 0.54). The noise-robust pitch recipe did not pay off (mirrors the phone-extractor story).
  - `jp_104` upstream is the **most noise-robust** and competitive on clean. Since sources are DeepFilterNet-denoised (clean) anyway, jp_104 is the safe, arguably better choice.
- **Fix applied**: set `assets/pretrained/pitch_estimator/current.pt` → **`jp_104_3_300k.pt`** (upstream), restoring the full upstream stack with jp_122.
- **Invariant Rule**: Validate a retrained PitchEstimator against pyworld GT with `analysis/probe_pitch_estimator.py` before promotion; it must beat upstream `jp_104` on `voiced_err_st`/`gross_err_rate` on the relevant (clean, since sources are denoised) condition. Ignore `voicing_acc`.

### 2.8 English Phonetic Faithfulness: why jp_122 "sounds Japanese", and where our English distillation falls short (2026-06-02)
- **Why upstream sounds Japanese**: the PhoneExtractor maps audio → a 128-dim phone space the converter consumes. If that space is carved around Japanese phoneme categories, English-specific contrasts absent in Japanese (`/r/–/l/`, `/θ/ /ð/`, `/v/–/b/`, `/æ/`) get merged onto the nearest Japanese category, and the converter renders a Japanese-accented version. **The accent is baked into the feature geometry — no converter training removes it.**
- **Our method is conceptually right**: the distillation teacher is `torchaudio HUBERT_BASE` (trained on English LibriSpeech-960), which encodes English phonetics well. Distilling from it *should* yield an English space.
- **Diagnostic** — `analysis/probe_english_faithfulness.py` (label-free). Uses HuBERT-L9 as the English ground-truth geometry and measures, on energy-filtered LibriSpeech frames: `knn_overlap` (do candidates preserve HuBERT's local phonetic neighbourhoods?) and `lin_cka` (global structural similarity). Includes a **PCA-128-of-HuBERT** row = the best a 128-dim representation can do.
- **Results (k=10, ~4000 frames)**:
  | extractor | knn_overlap (local) | lin_cka (global) |
  |---|---:|---:|
  | HuBERT-L9 (self) | 1.000 | 1.000 |
  | **HuBERT PCA-128 (128-dim ceiling)** | **0.850** | 0.990 |
  | `jp_122` (upstream) | **0.199** | 0.540 |
  | `en_clean_580k` | **0.412** | 0.473 |
  | `en_clean_200k` | 0.393 | 0.239 |
  | `en_nr_targetmix_300k` | 0.385 | 0.214 |
- **Conclusions**:
  1. **`jp_122` is genuinely Japanese-leaning**: worst local neighbourhood preservation (0.20). Its high global CKA (0.54) + low local kNN = coarse acoustics preserved, fine English contrasts merged. Concrete signature of the accent.
  2. **Our English approach is directionally correct**: `en_clean_580k` is ~2× more English-aligned (0.41 vs 0.20).
  3. **The 128-dim architecture is NOT the bottleneck**: PCA-128 retains 0.85 of the neighbourhood structure. The student can reach ~0.85.
  4. **Our distillation is under-realized**: best is 0.41 vs an achievable 0.85 — leaving >half the structure untransferred, despite training directly on this teacher. This is a **training/method gap, not an architecture limit**.
- **Likely causes of the gap (priority order)**: (a) undertraining + narrow data (100h / 580k vs upstream ~3M; `train-clean-100` is clean read-only — use LibriSpeech-960 / more diverse English); (b) per-frame `cos + 0.1·MSE` loss does not preserve *local neighbourhoods* — add a correlation / neighbourhood-preserving term; (c) teacher/layer choice (HuBERT-BASE L9 unvalidated — try ContentVec / WavLM-Large, which are more speaker-disentangled content encoders).
- **Label-based confirmation (non-circular)** — `analysis/probe_phone_abx.py` + `analysis/fetch_librispeech_aligned.py`. Uses REAL MFA phoneme labels (HF `gilkeyio/librispeech-alignments`, train_clean_100) to measure phoneme separability. These labels are never used in training, so unlike `knn_overlap` this is not circular.
  | extractor | overall ABX err ↓ | linear phone-acc ↑ | R/L | TH/S | V/B | DH/D |
  |---|---:|---:|---:|---:|---:|---:|
  | HuBERT-L9 (ceiling) | 0.152 | 0.815 | 0.19 | 0.16 | 0.25 | 0.22 |
  | `jp_122` (upstream) | 0.223 | **0.516** | **0.39** | **0.43** | **0.44** | **0.47** |
  | `en_clean_580k` | 0.234 | **0.699** | 0.26 | 0.33 | 0.29 | 0.32 |
  | `en_nr_targetmix_300k` | 0.276 | 0.656 | 0.30 | 0.37 | 0.34 | 0.34 |
  - **The "sounds Japanese" accent is now measured directly**: `jp_122` fails exactly the English contrasts absent in Japanese — R/L, TH, V, DH (0.39–0.47, several near chance). `en_clean_580k` handles them far better and has +18pt phone-acc (0.70 vs 0.52).
  - **Overall ABX is too coarse** (dominated by easy vowel/fricative pairs; jp_122 even edges en_clean there). Use **phone-acc and hard-pair ABX**, not overall ABX, to judge English fidelity.
- **⚠️ Tension with the 2.6 reversion**: we reverted to `jp_122` on `effective_rank` (18.8 ≫ en_clean 12.9) + a "big tongue" listening test — but that A/B used the *collapsed* `en_nr` (v2), NOT `en_clean_580k`. The label probe shows `en_clean_580k` is clearly the most English-correct extractor we have. Reconciliation: **`jp_122` is richer but English-wrong** (Japanese phone categories); **`en_clean_580k` is English-correct but less rich** (limited data/training → may under-articulate). `phone_acc` (linear readability) ≠ articulation; richness matters too. The real target is an extractor that is BOTH English-correct AND rich/articulate.
- **Invariant Rule (updated)**: judge English PhoneExtractors on the full panel — `knn_overlap` vs HuBERT-L9 (English-correctness, label-free), `phone_acc` + hard-pair ABX (English-correctness, label-based), AND `effective_rank` (richness/articulation). A promotable English extractor must beat `jp_122` on phone-acc/hard-pairs **without** regressing `effective_rank` far below jp_122, and win a listening test. No single metric suffices — jp_122 is rich (18.8) yet English-wrong (phone-acc 0.52); en_clean is English-correct (0.70) yet less rich (12.9).

### 2.9 Promotion gate = converted-audio quality, NOT intrinsic probes (2026-06-02)
- **Why**: intrinsic probes (phone-acc, knn_overlap, effective_rank) are *diagnostics*. Optimizing/promoting on them invites Goodhart — e.g., `phone_acc` measures linear readability of frozen features, not whether the converter renders crisp, un-accented English. The north star is the **converted English audio**.
- **⚠️ Eval-set bug**: the trainer's built-in test set `assets/test/` is **Japanese** Common Voice (`common_voice_ja_*`). The PhoneExtractor reads content from the *source*, so converting Japanese clips **cannot** reveal English-pronunciation quality. All prior listening/UTMOS evals on the default test set were measuring the wrong language. **English source clips are required** to evaluate the English accent.
- **Tier-1 harness (the gate)** — converts ENGLISH sources through a checkpoint and scores them:
  - `analysis/fetch_librispeech_aligned.py` → `analysis/data/ls_aligned.pkl` (English wavs + MFA ref phonemes).
  - `analysis/convert_eval.py --ckpt <ck.pt.gz> --targets <spk,...> --n N --tag <tag>`: reconstructs `ConverterNetwork` (load `net_g`/`phone_extractor`/`pitch_estimator`, then **`net_g.enable_hook()`** to activate the VQ codebook; note `frozen_modules` is a plain dict so `.to(device)` does NOT move phone/pitch — move them explicitly). Writes `analysis/converted/<tag>/{source,<target>}/*.wav` + `manifest.json`.
  - `analysis/score_converted.py --tags <a,b> --target <spk>`: **PER** (wav2vec2-espeak IPA, self-referenced source→ref vs converted→hyp; catches R/L, TH, V accent errors), **WER** (Whisper `base.en`, intelligibility), **UTMOS** (naturalness), **SPK** (WavLM-SV cosine to real target-speaker centroid; report `SPK_src` too — should be lower). PER/WER lower=better; UTMOS/SPK higher=better.
  - Deps: `transformers` (Whisper + wav2vec2-CTC + WavLM-SV); PER decoding is a manual CTC greedy decode reading raw `vocab.json` to **avoid the espeak/phonemizer system dependency**.
- **Validation plan**: run a `jp_122` baseline converter with the *identical* Path A config (only phone extractor differs), score both with Tier-1, then check which cheap intrinsic probe predicts the Tier-1 ranking → that becomes the trusted screen for v3.
- **Invariant Rule**: promote a phone extractor only if its converter beats the baseline on Tier-1 (esp. PER) on ENGLISH sources + a listening test. Intrinsic probes are screens, used only after being validated against Tier-1.
- **✅ Gate run (2026-06-03) — `en_clean` FAILED, `jp_122` stays current.** Ran the full Tier-1 gate on the Path A re-run: `path_a_en_clean` vs `path_a_jp122_baseline` (60k each, identical config, only the frozen PhoneExtractor differs), converting n=40 English clips to `sion` + `noxus_male`. Full numbers + v3 plan in `experimental_log.md` Experiment 5. Outcome:
  - **PER: tied** (within noise, both targets) — did not separate the extractors.
  - **WER: contradictory across speakers** — `en_clean` better on `sion` (0.60 vs 0.82), `jp122` better on `noxus_male` (0.28 vs 0.47). No consistent intelligibility winner.
  - **UTMOS: `jp122` wins both** (sion 2.31 vs 2.02; noxus 2.97 vs 2.49) — tracks its higher effective_rank.
  - **SPK: `en_clean` wins both** (timbre), but noxus is male→male so `SPK_src≈0.75` deflates the signal.
  - **Listening test (`analysis/listening_test.html`, blind A/B): `en_clean` audibly muffled / under-articulated ("big tongue")** — the deciding signal. UTMOS predicted it; PER/WER did not. **Do NOT promote `en_clean`.**
  - **Lesson for v3**: of the cheap Tier-1 metrics, **UTMOS was the only one that tracked the listening verdict** here — PER/WER were inconclusive/contradictory. Use UTMOS as the primary cheap proxy for the "muffle/big-tongue" failure, but the listening test still decides promotion.
- **🎯 v3 strategy REVISED 2026-06-03 (user listening verdict): jp_122 richness is a HARD CONSTRAINT.** The user is certain from listening that `jp_122` is better because it extracts **all** phonetic features with no muffled big-tongue mask. This reframes v3: it is **no longer** "balance richness vs English-correctness" — richness/articulation is non-negotiable and English-contrast correction is only a small bounded nudge on top of jp_122. Full plan in `experimental_log.md` Experiment 5. Key points:
  - **From-scratch English distillation is ABANDONED** (v1 muffled @580k, v2 collapsed — both big-tongue). Root cause is structural: a 128-dim student wholesale-matching a clean 768-dim SSL teacher averages phonemes. **Never re-distill the whole representation from a clean SSL teacher again**, regardless of data scale/teacher. DROP the prior "scale to LibriSpeech-960 / ≥1M from scratch" idea.
  - **New recipe = anchored, targeted correction of jp_122**: (1) mandatory warm-start = jp_122; (2) **dominant loss = preservation anchor to frozen jp_122 features** (pins the rich geometry, prevents the muffle); (3) English correction = a *small* supervised contrastive/triplet loss on hard pairs jp_122 merges (R/L, TH/S, V/B, DH/D) using MFA labels — NOT full HuBERT regression; (4) augment with noise(SNR≥20dB)+mild reverb only.
  - **Abort/gate conditions**: `effective_rank` must stay ≥ jp_122 (18.8); UTMOS must not regress; phone-acc/hard-pair ABX must improve; final blind listening test must show zero added muffle. Any failure = no promotion.
  - **Go/No-Go first**: run a cheap ~50–100k pilot; if it can't beat jp_122 on hard-pairs at **zero** richness/UTMOS cost, **keep jp_122 and accept the residual accent**. v3 is upside-only and must never touch articulation.
- **v3/v3.1 RESULT (2026-06-03): anchored-correction approach DID NOT improve intelligibility → superseded by v4.** v3 (anchor + diffuse SupCon, 80k) judged ≈ jp_122 on hard contrasts (R/L 0.388 vs 0.389, all within ±0.01); phone_acc/PNMI gained nothing; eff_rank fell 18.8→14.0. Root cause: the dominant jp_122 anchor pins exactly the frames that need to change (jp_122 *merges* R/L, TH/S). v3.1 added a surgical hard-pair separation loss (`train_v3.py` `hardpair_loss`, gamma=1.0) which *did* push pairs apart (hp 0.30→0.003 by 7.5k, eff_rank held ~31) but was stopped in favour of the principled fix below. Trainers `train_v3.py` remain for reference.
- **🎯 2.10 — THE REAL METHOD (2026-06-03): jp_122 used a CATEGORICAL (classification) objective, not regression. This is why it is rich/clear.** Evidence: (a) `README.original.md` Reference: **Soft-VC** is the explicit basis for `PhoneExtractor`; (b) Soft-VC trains the content encoder by **predicting a distribution over discrete units via cross-entropy** (classification), NOT continuous-feature regression; (c) Project Beatrice publishes `prj-beatrice/japanese-hubert-base-phoneme-ctc` (HuBERT fine-tuned with **phoneme CTC** on pyopenjtalk labels / ReazonSpeech). **Our v1/v2 `train.py` reimplemented `cos+MSE` regression — the fork's `phone_extractor_trainer/README.md` *guessed* this and was WRONG about the objective.** Why it explains everything: a categorical target forces phoneme-discriminative features (rich, clear, high eff_rank); `cos+MSE` regression = conditional-mean smoothing → low eff_rank → the muffle. jp_122's English accent (R/L,TH,V merged) = its phoneme inventory was Japanese.
  - **v4 = Soft-VC done right, on English** (`phone_extractor_trainer/train_v4.py`, LAUNCHED 2026-06-03, `outputs/phone_extractor_en_v4`): (1) k-means HuBERT-BASE-L9 features into K=500 discrete units over English audio; (2) train PhoneExtractor + a linear head to **predict each frame's unit id by cross-entropy** (the 128-d output is the "soft unit"; head dropped at export). Same teacher info as en_clean, ONLY the objective changes (regression→classification). Because HuBERT-L9 is English, the units separate R/L,TH,V → expect richness AND English correctness together, no anchor/contrastive hacks. Run from scratch, 200k steps, K=500. Judge with the same panel (eff_rank + phone_acc/PNMI/hard-pairs) then Tier-1 WER/UTMOS/listen. **This is the prime path; v3.x are deprecated.** Future option: distill from an English **phoneme-CTC** teacher (mirror Beatrice's JP model) for an even sharper target.

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
