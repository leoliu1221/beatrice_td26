# Training Log & Experimental Findings

A running log of experiments, hyperparameter tuning, data scaling, and model behavior for the Beatrice Trainer fork.

---

## Experiment 5: Tier-1 Converted-Audio Gate — `en_clean` vs `jp122` (Path A re-run)
**Date:** Jun 3, 2026
**Objective:** Re-run the PhoneExtractor A/B on the **correct gate** (converted ENGLISH audio, lesson 2.9) after the prior `grad_weight_ap` A/B (`outputs/ab_ap*_v2`) was invalidated by the collapsed v2 extractor (lesson 2.6). Two identical Path A converters trained 60k steps on `preprocessed/new_lol_data_df`, differing **only** in the frozen PhoneExtractor:
- `outputs/path_a_en_clean` — `en_clean_580k` (English-distilled, LibriSpeech-100).
- `outputs/path_a_jp122_baseline` — `jp_122_3000k` (upstream).

### Harness
- `analysis/convert_eval.py` — converted n=40 English LibriSpeech clips (`analysis/data/ls_aligned.pkl`) to targets `sion` + `noxus_male`. Both loaded at iter=60000, n_speakers=6. Output under `analysis/converted/{en_clean_60k,jp122_60k}/`.
- `analysis/score_converted.py` — PER (wav2vec2-espeak IPA, source→ref vs converted→hyp), WER (Whisper base.en), UTMOS, SPK (WavLM-SV cosine to real target centroid; SPK_src = sim to source, want low).

### Results (n=40 per target)
| target | tag | PER ↓ | WER ↓ | UTMOS ↑ | SPK ↑ | SPK_src ↓ |
|---|---|---|---|---|---|---|
| sion | `en_clean_60k` | **0.282** | **0.600** | 2.023 | **0.941** | 0.586 |
| sion | `jp122_60k` | 0.286 | 0.822 | **2.305** | 0.939 | 0.576 |
| noxus_male | `en_clean_60k` | 0.269 | 0.470 | 2.494 | **0.928** | 0.754 |
| noxus_male | `jp122_60k` | **0.266** | **0.277** | **2.970** | 0.878 | 0.759 |

JSON: `analysis/converted_quality_sion.json`, `analysis/converted_quality_noxus_male.json`.

### Findings
- **PER**: tied across the board (within noise) — the IPA recognizer does not separate the two extractors.
- **WER**: **contradictory across speakers** — `en_clean` far more intelligible on `sion` (0.60 vs 0.82) but `jp122` far more intelligible on `noxus_male` (0.28 vs 0.47). No consistent intelligibility winner. Objective metrics do NOT settle the A/B.
- **UTMOS**: `jp122` wins **both** targets — consistent with it being the richer extractor (effective_rank 18.8 vs 12.9, lesson 2.6).
- **SPK**: `en_clean` wins both (esp. noxus_male, 0.928 vs 0.878), but `SPK_src ≈ 0.75` on noxus (male→male) means the gap is more meaningful than the absolute value.

### Listening test (the gate) — `en_clean` FAILS
- Built `analysis/listening_test.html`: blind, per-item randomized A/B over the 40 clips, target + criterion selectors, served via `python -m http.server` from `analysis/`.
- **Verdict (user listen, 2026-06-03):** several `en_clean` conversions are **clearly muffled / under-articulated** — the "big tongue" signature again (lessons 2.2, 2.6, 2.8). UTMOS correctly predicted this; PER/WER did not.
- **Decision: do NOT promote `en_clean`. `jp_122` stays `current.pt`.** The English-distillation effort remains a net *perceived-quality* regression despite being measurably more English-correct on label probes (lesson 2.8). UTMOS is the cheap metric that tracked the muffle; PER/WER were inconclusive/contradictory here.

### Plan: PhoneExtractor v3 — REVISED 2026-06-03 (jp_122 richness is a HARD constraint)
**Reframe after the listening verdict.** The user is certain, from listening, that `jp_122` is the better extractor because it captures **all** phonetic features with no muffled "big-tongue" mask. So v3 is no longer "balance richness vs English-correctness" — **richness/articulation is a non-negotiable constraint, and English-contrast correction is only a small, bounded nudge on top of jp_122.**

**What is now ruled out (failed twice — stop trying):**
- **From-scratch English distillation is ABANDONED.** v1 `en_clean_580k` (muffled @580k) and v2 `en_nr_targetmix` (collapsed) both produced the big-tongue mask. The root cause is structural (lesson 2.6): asking a 128-dim student to *wholesale-match* a clean 768-dim SSL teacher forces a phoneme-agnostic average. **Do not re-distill the whole representation from any clean SSL teacher again** — that is the step that smears, regardless of data scale or teacher choice.
- Therefore **DROP** the old plan's items "scale to LibriSpeech-960 / ≥1M from scratch" and "wholesale richness-preserving re-distillation" — both still re-learn the entire space and re-incur the muffle risk.

**New approach — anchored, targeted correction of jp_122 (richness preserved by construction):**
1. **Mandatory warm-start = jp_122.** The student *is* jp_122 at step 0; training may only perturb it.
2. **jp_122-preservation anchor (dominant loss term).** Keep student features close to the **frozen jp_122** features (e.g. high-weight cos/MSE to jp_122's own output). This pins the rich geometry the user validated; it is the safeguard against the muffle, not an afterthought.
3. **English correction = a gentle, *targeted* nudge, NOT full HuBERT regression.** Only push on the specific contrasts jp_122 merges (R/L, TH/S, V/B, DH/D — measured low in lesson 2.8). Preferred form: a **supervised contrastive / triplet loss on hard phoneme pairs** using REAL MFA labels (`analysis/fetch_librispeech_aligned.py`), so we directly pull those categories apart without asking the student to imitate HuBERT's entire 768-dim manifold. Weight this term **small** relative to the anchor.
4. **Richness is the gate, checked every eval (abort conditions):**
   - `effective_rank` (`probe_phone_extractor.py`) must stay **≥ jp_122 (18.8)** — abort the run if it drops beyond a small margin.
   - **UTMOS** on converted English must not regress vs jp_122 (cheap muffle proxy — the only Tier-1 metric that tracked the listening verdict).
   - `phone_acc` / hard-pair ABX (`probe_phone_abx.py`) must **improve** over jp_122 (the only thing v3 is allowed to win on).
   - Final promotion: blind listening test must show **zero added muffle** AND audible R/L·TH·V improvement. Either failing = no promotion.
5. **Augmentation:** noise (SNR ≥ 20 dB) + mild reverb ONLY. No LPF, no formant-shift, no target-mix (lessons 2.5/2.6).
6. **Go/No-Go reality check (do this FIRST, cheaply).** jp_122 is the confirmed champion; v3 is upside-only on accent and must not touch articulation. Run a **short pilot (~50–100k steps)** of the anchored-correction recipe. If it cannot beat jp_122 on hard-pair ABX **without any** effective_rank/UTMOS regression, **STOP and keep jp_122** — accept the residual accent rather than risk the mask. Only scale up a pilot that has already proven it can improve contrasts at zero richness cost.

**De-prioritized (only as a *correction signal* source, never as a new full teacher):** WavLM-Large / ContentVec hard-pair embeddings could supply the contrastive targets in step 3; they are NOT a replacement teacher for wholesale distillation.

### Judging method (researched + adopted 2026-06-03)
Literature review (ContentVec ICML'22; DC-Spin Interspeech'25; SUPERB-headless arXiv:2308.14456) → a PhoneExtractor is judged on a **two-axis panel**, never a single metric (lesson 2.9 anti-Goodhart):
- **Richness / articulation (the hard constraint, muffle detector)** — `analysis/probe_phone_extractor.py`: `effective_rank` (participation ratio ↑), `temporal_contrast` ↑, `mean_pairwise_cos` ↓. These caught the "big tongue".
- **English correctness** — `analysis/probe_phone_abx.py`: `phone_acc` (linear probe ↑), **`PNMI` (NEW)**, and hard-pair ABX ↓ (R/L, TH/S, V/B, DH/D). **PNMI** = mutual information between k-means clusters of frames and MFA phone labels, normalised by phone entropy; DC-Spin shows it is a *more reliable* proxy than ABX (which can mislead). It needs no trained probe, so it cross-checks `phone_acc` for Goodhart.
- *(Documented but not run here — needs a speaker-balanced set, which `ls_aligned.pkl` is not: 127 spk / 200 utt)*: **speaker-invariance** (ContentVec) — a VC content encoder's features should NOT linearly predict speaker; lower speaker-probe acc = better.
- **Final gate stays Tier-1** converted-audio UTMOS + blind listening (lesson 2.9). Intrinsic panel is only a screen.

**Judging result — jp_122 vs en_clean (validated method, 2026-06-03):**
| extractor | eff_rank ↑ | temp_contrast ↑ | mean_pair_cos ↓ | phone_acc ↑ | PNMI ↑ | R/L ABX ↓ | TH/S ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| HuBERT-L9 (ceiling) | — | — | — | 0.817 | 0.615 | 0.190 | 0.159 |
| **jp_122 (upstream)** | **18.8** | **0.419** | **0.102** | 0.516 | 0.393 | 0.389 | 0.433 |
| en_clean_580k | 12.9 | 0.247 | 0.134 | **0.697** | **0.474** | **0.264** | **0.326** |
- **PNMI corroborates phone_acc** (en_clean 0.474 > jp_122 0.393), and both agree with hard-pair ABX → the English-correctness gap is real, not a linear-probe artifact. The tension is now measured on 5 axes: jp_122 owns richness, en_clean owns correctness. v3 must close correctness without dropping eff_rank.

### v3 Go/No-Go pilot — LAUNCHED 2026-06-03
- **Trainer**: `phone_extractor_trainer/train_v3.py` (new). Loss = `1.0·L_anchor + 0.15·L_supcon`:
  - `L_anchor` = `1 - cos(student, frozen jp_122)` on diverse unlabelled English (`datasets/librispeech/.../train-clean-100`, 28.5k flac) — pins richness everywhere.
  - `L_supcon` = supervised contrastive (Khosla) over frame features of a **disjoint** labelled set (`analysis/data/ls_aligned_train.pkl`, 175 utts / 10.7k phone segs, fetched with `fetch_librispeech_aligned.py --skip 200` so it does NOT overlap the 200-utt probe/eval set → probe stays non-circular).
  - student warm-started from jp_122; jp_122 frozen as anchor; `effective_rank` logged every 100 steps as the abort gate.
- **Launch**: `outputs/phone_extractor_en_v3`, 80k steps, lr 1e-4 cosine, batch 16 (anchor) + 8 (supcon), ~12 it/s (~1h50m). Log: `outputs/phone_extractor_en_v3_train.log`.
- **Early signal (step ~300)**: `effective_rank` 25–33 (≥ jp_122's 18.8 → richness gate satisfied, NOT collapsing); `anchor_cos` 0.98→0.91 (SupCon reshaping toward English contrasts, as intended). `l_supcon` ~6.4.
- **Next**: at checkpoints, run the judging panel (`probe_phone_extractor` + `probe_phone_abx`) — promote to Tier-1 only if eff_rank stays ≥ jp_122 AND phone_acc/PNMI/hard-pairs beat jp_122. Then convert + blind listen. If it can't beat jp_122 on hard-pairs at zero richness cost → keep jp_122 (Go/No-Go).

### v3 (80k) judged → FAILED Go/No-Go (no intelligibility gain). User goal clarified = "intelligible words".
Judged `checkpoint_00080000.pt` on the full panel (added to both probes as `en_v3_anchored_80k`):
| extractor | eff_rank ↑ | temp_contrast ↑ | phone_acc ↑ | PNMI ↑ | R/L ↓ | TH/S ↓ | V/B ↓ | DH/D ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| jp_122 | 18.8 | 0.42 | 0.516 | 0.393 | 0.389 | 0.433 | 0.441 | 0.472 |
| **en_v3_80k** | 14.0 | 0.40 | 0.529 | 0.398 | 0.388 | 0.435 | 0.444 | 0.465 |
| en_clean_580k | 12.9 | 0.25 | 0.698 | 0.474 | 0.264 | 0.326 | 0.290 | 0.318 |
- **Result: v3 ≈ jp_122.** Articulation preserved (temp_contrast 0.40, near jp_122) but the **hard contrasts did not move** (R/L 0.388, TH/S 0.435, ... all within ±0.01 of jp_122); phone_acc/PNMI gained ~nothing (within noise). eff_rank fell 18.8→14.0. → paid a small richness cost for **zero intelligibility gain**. DO NOT promote.
- **Root cause**: `alpha=1.0` anchor to jp_122 **dominated** the small `beta=0.15` diffuse SupCon. The anchor pins exactly the frames we need to change (jp_122 *merges* R/L, TH/S), and SupCon spread over 39 classes couldn't overcome it on the specific pairs. **Lesson: to fix the accent you must explicitly, surgically move the confusable contrasts; a broad correction under a strong jp_122 anchor cannot.**

### v3.1 — surgical hard-pair correction, LAUNCHED 2026-06-03
- **Change to `train_v3.py`**: added `hardpair_loss` = `mean relu(cos(a_i,b_j) - margin)` over the merged pairs (R/L, TH/S, V/B, DH/D, S/SH, IH/IY, AE/EH), which **directly pushes apart** exactly the contrasts that didn't move. New loss = `1.0·L_anchor + 0.5·L_supcon + 1.0·L_hardpair` (args `--gamma`, `--hardpair-margin`). Calibrated: `hp` term ~0.30 at init, active.
- **Launch**: `outputs/phone_extractor_en_v3_1`, warm-start jp_122, 80k steps, pid 1832903, log `..._v3_1_train.log`. Early: `hp` 0.30→0.24 (pairs separating), `eff_rank` ~34 (richness OK).
- **Validation chain (user goal = intelligible words)**: (1) intrinsic gate at ~20k — hard-pair ABX must drop toward en_clean AND temp_contrast/eff_rank hold; if hard-pairs don't move, abort. (2) If intrinsic passes, train a Path A converter + score **WER (Whisper) + UTMOS** on converted English (intelligibility = the user's actual target) + blind listen. Intrinsic correctness is necessary but NOT sufficient (en_clean was correct-on-probe yet muffled-on-audio); v3.1's edge is it keeps jp_122 articulation while fixing contrasts.
- **STOPPED in favour of v4** (the principled Soft-VC fix): v3.x hard-pair loss is a crude manual stand-in for what a categorical objective gives for free. See Experiment 6.

---

## Experiment 6: v4 Soft-VC categorical objective — chasing richness + English correctness
**Date:** Jun 3–5, 2026  
**Objective:** Replace the (wrong) `cos+MSE` regression with the **real** Soft-VC categorical objective (CE to discrete HuBERT-L9 units), then diagnose and fix the richness loss. Trainer: `phone_extractor_trainer/train_v4.py`. Full design rationale in `project_context.md` lessons 2.10–2.11.

### Recipe (`train_v4.py`)
- (1) k-means HuBERT-BASE-L9 features → K discrete units over English audio (cached `kmeans_kK_l9.pt`); (2) train PhoneExtractor + a training-only head to **predict each frame's unit id by cross-entropy** (128-d output = the "soft unit", head dropped at export). Labels upsampled 2× (HuBERT 50 fps → student 100 fps).

### Monitoring + auto-terminate (`phone_extractor_trainer/watch_richness.py`, NEW)
- Scores every new numbered checkpoint's `effective_rank` over a FIXED 40-clip held-out set (seed 1234, train-clean-100), logs `richness_monitor.csv`, and SIGTERMs the training PID after `--patience` (5) consecutive checkpoints **below the running max**. Fixed the checkpoint-save off-by-one so numbered ckpts land every 10k steps.
- ⚠️ This guards against *erosion* (high→low). It will NOT fire on a run that *recovers from collapse* (low→high) — that bit us below.

### Run A: K=500 from scratch
- `eff_rank` **14.25 @40k → 11.65 @153k** while English-correctness kept improving → **richness erodes with more CE steps**. Banked `outputs/phone_extractor_en_v4/checkpoint_bank_153k.pt`. Raised K→2000 to slow collapse.

### Diagnosis: the limiter is the OBJECTIVE, not the architecture
- Our `PhoneExtractor` is the *identical* 323-tensor net as jp_122 (loads `missing=0 unexpected=0`) and jp_122 reaches `eff_rank ≈ 20.7`; the 128-dim output is not binding (we sit at 8–14). So capacity is fine.
- **Mechanism = neural collapse** (Papyan–Han–Donoho): hard CE shrinks within-unit variance and aligns class means to a simplex → destroys sub-phonemic detail → `eff_rank` falls *with* training. jp_122 avoided it via (a) noise on the PhoneExtractor output during training, (b) Soft-VC's soft/distributional framing.
- **Calibration**: jp_122 features sit at a TINY scale (per-dim std median **0.027**, overall 0.030). Richness = how variance is *distributed* across dims, not magnitude → any VICReg must be **scale-invariant** (normalize by global std; `gamma` = fraction of avg dim scale).

**Apples-to-apples richness (same 40 clips, `watch_richness.score_checkpoint`):**
| model | eff_rank ↑ | temporal_contrast ↑ | pairwise_cos ↓ |
|---|---:|---:|---:|
| **jp_122 (target)** | **20.72** | 0.459 | 0.072 |
| K500_bank @153k | 11.12 | 0.369 | 0.097 |
| **rich_200k (all 3 fixes)** | **7.97** | 0.226 | **0.844** |

### Three richness fixes added to `train_v4.py` (opt-in; defaults reproduce plain Soft-VC)
- `--head-mlp-dim` — MLP CE head so collapse is absorbed by the head, not the exported 128-d features (SimCLR projection-head effect).
- `--output-noise` — Gaussian noise (× feature std) on the **CE path only**; jp_122 recipe. Exported/regularized features stay clean.
- `--vicreg-var/--vicreg-cov/--vicreg-gamma` — scale-invariant VICReg variance+covariance on clean exported features (recruits collapsed dims w/o forcing white-noise uniformity).

### ❌ Run B: jp_122 warm-start + ALL THREE fixes at once (`_k2000_rich`) — FAILED
- lr 1e-4, K=2000, MLP head + noise + VICReg. RESULT = **worst of all** (eff_rank 7.97, pairwise_cos 0.844 = heavily collapsed). CSV: **total collapse early** (cos=1.0000 @10k, eff_rank 0.04) then a slow crawl to 7.97 @200k — never recovered. Even plain K=500 from-scratch (14.25 @40k) beat it.
- **Diagnosis = head-shock**: a fresh random head's CE gradients nuked the warm-started body in the first few k steps. **Methodology error: changed 4 variables at once (warm-start + MLP head + noise + VICReg) → confounded.** The monitor never tripped because the trajectory was *increasing* (recovering from collapse), the opposite of the erosion shape it guards.
- **Lesson: ablate ONE variable at a time.**

### 🎯 Run C: frozen-body head-warmup (`_k2000_jpfreeze`) — IN PROGRESS (2026-06-05)
- New `train_v4.py` flags: `--freeze-body-steps` + `--body-lr-scale` (2-group optimizer: head `lr_scale=1.0`, body `lr_scale=body_lr_scale`). Clean isolation of the head-shock fix — **no VICReg/noise/MLP**.
- Recipe: warm-start jp_122, **freeze body 5000 steps** (linear head learns English units on jp's rich features), then **unfreeze at body LR 2e-5** (`--lr 2e-4 --body-lr-scale 0.1`, K=2000, 200k steps).
- Early signal: body frozen → `eff_rank` ~26–29 (richness **intact**), `acc` climbing (0.026 → 0.075 by step 1.3k), `ce` falling — the opposite of Run B's collapse. **Decisive test = does richness HOLD near ~20 after the body unfreezes at 5k?** Monitor's first held-out score at 10k. If it holds while correctness improves → success; if it erodes → re-introduce regularizers one at a time.
- Reminder (lesson 2.9): `eff_rank` is a proxy VICReg can game; CE stays primary; **blind listening test still decides promotion**.

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

### Run 4.1c final: PhoneExtractor v2 converged
- Training completed all 300k steps (run resumed from step 262k after a terminal-closure interruption at 260k).
- Final eval sweep (every 10k from 200k → 300k, n=64):

| step | (a) clean LS | (b) aug LS | (c) clean target | (d) aug target |
|---:|---:|---:|---:|---:|
| 250,000 | 0.7685 | 0.7387 | 0.8379 | 0.7746 |
| 270,000 | 0.7690 | 0.7391 | 0.8388 | 0.7756 |
| 290,000 | 0.7690 | 0.7388 | 0.8390 | 0.7756 |
| **300,000** | **0.7692** | 0.7388 | **0.8390** | 0.7755 |

- All four metrics flatlined from ~270k onward — clean convergence, **no overfit signature** (no metric regression in last 30k).
- **Exported** `outputs/phone_extractor_en_v2/checkpoint_00300000.pt` → `assets/pretrained/phone_extractor_en_v2.pt` (14 MB, load_state_dict clean).
- v1 production was 580k @ (c)=0.7095 / (d)=0.6549. v2 wins by **+0.130 / +0.121** on target domain. The −0.014 in-domain drop is a reasonable price.
- **Status**: ✅ **DONE**.

### Run 4.2: PitchEstimator v2 from-scratch noise-robust training (running)
- **Recipe**: VCTK, warm-start from `assets/pretrained/104_3_checkpoint_00300000.pt`, 300k steps, batch 256, 8 workers, `--augment` (no target-mix — F0 supervision is universal across natural speech; VCTK already covers 109 speakers across both sexes).
- **Why no target-mix**: pyworld F0 is a ground-truth label, not a learned target distribution. The student only needs noise-invariance, not target-domain phonetic specialization. Phone extractor needed target-mix because HuBERT features are continuous representations that *do* drift across domains.
- **Why from scratch (not resume)**: same logic as Run 4.1c — fine-tuning a clean-distilled checkpoint with augmentation likely leaves residual clean-specialization. Better to make noise-invariance a primary objective from step 0.
- **Setup**: archived previous clean-only `outputs/pitch_estimator_v2` → `outputs/pitch_estimator_v2_old_<ts>`. Output dir reused so `pitch-export` Makefile target still works without flag changes.
- **Caveat (already in code)**: `formant_shift_probability` is force-set to 0 in `PitchDataset` because Beatrice's formant shift uses spectral-envelope warp + resample which can perturb pyworld F0 labels.
- Throughput stabilizing at ~3.4 it/s after dataloader warmup. ETA ~25h.
- **Eval plan**: no dedicated eval script (yet); will monitor `train/acc` (target ≥0.70, started at 0.70 due to warm-start) and `train/err_semitones` (target ≤2 st). If those plateau cleanly, export `checkpoint_00300000.pt` → `assets/pretrained/pitch_estimator_v2.pt`.
- **Status**: 🔄 **In Progress**.

### Run 4.2 final: PitchEstimator v2 done
- VCTK noise-robust training reached step 300k. Exported `outputs/pitch_estimator_v2/checkpoint_00300000.pt` → `assets/pretrained/pitch_estimator_v2.pt` (6.7 MB, clean load).
- (No dedicated held-out eval script for the pitch estimator yet; relied on `train/acc` and `train/err_semitones` plateauing cleanly.)
- **Status**: ✅ **DONE**.

### Run 4.3: Beatrice A/B on `grad_weight_ap` (running)
- **Context**: upstream merged PR #1 ("trainer progress reporting and throughput tuning") which, among other things, **flipped the default of `grad_weight_ap` from 100.0 → 0.0**. `loss_ap` is the D4C-based aperiodicity reconstruction loss that supervises the vocoder's aperiodicity branch (noise excitation for unvoiced consonants and breathy components). It conditions on the model's predicted F0; if F0 is wrong, the supervision is corrupted.
- **Hypothesis**: with a freshly retrained pitch estimator (Run 4.2) whose F0 outputs may differ from the original Japanese-leaning baseline, the D4C-based supervision could be either a net positive (more accurate F0 → better aperiodicity targets) or a net negative (slight F0 errors propagated into a 100×-weighted loss → buzz on deep voices). The upstream PR's flip to 0 suggests the latter at least sometimes. We A/B both to find out.
- **Setup**: two configs identical except for `grad_weight_ap`:
  - `assets/configs/ab_ap0_v2.json` — upstream default.
  - `assets/configs/ab_ap100_v2.json` — legacy weight.
  - Both use `phone_extractor_en_v2.pt` + `pitch_estimator_v2.pt`, balanced regularization (Phase B/C/D config), 60k steps, `preprocessed/new_lol_data_df`.
- **Execution**: sequential (12 GB GPU cannot fit two trainers). Driven by `run_ab_grad_weight_ap.sh`. Wall clock ≈ 11 h per run, ≈ 22 h total.
- **Outputs**: `outputs/ab_ap0_v2/`, `outputs/ab_ap100_v2/` (each has `test/` with rendered comparison samples at evaluation intervals).
- **Decision criterion**: A/B listening on `outputs/ab_ap*_v2/test/` at the end of each run. Specifically rate:
  1. Buzz/noise on voiced speech (especially deep voices: sion, demacia_male, noxus_male).
  2. Fricative/sibilant clarity (/s/, /ʃ/, /f/, /h/) — the failure mode for `grad_weight_ap=0` if it exists.
  3. Overall naturalness.
- **Status**: 🔄 **In Progress** (`ab_ap0_v2` started, `ab_ap100_v2` queued).

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
