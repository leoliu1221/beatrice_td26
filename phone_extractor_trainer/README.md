# phone_extractor_trainer

A standalone trainer for Beatrice 2's **`PhoneExtractor`** network. Produces
checkpoints that are **drop-in replacements** for
`assets/pretrained/122_checkpoint_03000000.pt`.

The motivation: the shipped PhoneExtractor was trained on a mixed corpus that
includes ReazonSpeech (Japanese), which baked a Japanese phonetic prior into
its 128-dim output space. For English-only voice conversion, retraining the
PhoneExtractor on an English-only teacher signal eliminates the bias at its
source. See the parent README's *English voice conversion — known caveat*
section.

---

## ⚠️ CORRECTION (2026-06-03): the original objective was CLASSIFICATION, not regression

**The reconstruction below (cos+MSE feature *regression*) was an early guess and
is WRONG about the training objective.** Evidence gathered later:

1. **`README.original.md` Reference** states **Soft-VC** is the basis for
   `PhoneExtractor`. Soft-VC ([van Niekerk et al.](https://arxiv.org/abs/2111.02392),
   `bshall/hubert`) trains its content encoder by **predicting a distribution
   over discrete units (k-means of HuBERT) via cross-entropy** — a
   **classification** objective. It does *not* regress continuous HuBERT vectors.
2. Project Beatrice publishes **`prj-beatrice/japanese-hubert-base-phoneme-ctc`**
   — `rinna/japanese-hubert-base` fine-tuned with a **phoneme-CTC** head on
   pyopenjtalk labels (ReazonSpeech). A supervised phoneme target.

**Why this matters (root cause of the muffle):** regression to continuous targets
(`cos+MSE`, this trainer's `train.py`) collapses features toward the conditional
mean of all sounds at a frame → low `effective_rank` → the muffled "big tongue".
A **categorical** target forces phoneme-discriminative features → high rank,
sharp articulation, clear words. This is exactly why the shipped jp_122
(categorical) is rich while our `en_clean` (regression) is muffled. jp_122's
English accent (R/L, TH, V merged) is because its phoneme inventory was Japanese.

**Use `train_v4.py` (Soft-VC unit classification), not `train.py`, for new runs.**
`train.py`/`train_v3.py` are kept for reference and reproducibility of the v1–v3
experiments only. See `project_context.md` lesson 2.10 and `experimental_log.md`.

---

## (Historical, superseded) How we *first* reconstructed the recipe

> The following inference was made before the evidence above; it correctly
> identified the **teacher** (HuBERT) and architecture but **mis-identified the
> objective** as regression. Retained for history.

The shipped `122_checkpoint_03000000.pt` was packaged for inference only:

```
top-level keys : ['phone_extractor']
tensors        : 323 (matches default PhoneExtractor() ctor exactly,
                       verified by `load_state_dict(strict=True)`)
training steps : 3,000,000  (from filename)
no optimizer state, no extra heads, no quantizer codebooks
```

The training-side supervision head was deliberately stripped before publish.
The architecture + published references established:

1. The parent README cites **Soft-VC** as the inspiration for PhoneExtractor.
   *(Correction: Soft-VC's objective is unit **classification**, not feature
   regression — see the correction note above.)*
2. The output is **128-dim continuous features** with `weight_norm` on the
   final head — the *soft units*. Beatrice's `ConverterNetwork` adds VQ *on top*
   (per-speaker codebooks at `@/beatrice_trainer/__main__.py:2178-2184`), so the
   PhoneExtractor itself is unquantized.
3. The `FeatureExtractor` (6 conv layers, total stride 160) is the **wav2vec
   2.0 feature extractor minus one stride layer** — explicitly cited by the
   parent README.
4. Input is fixed at 16 kHz (`"in_sample_rate": 16000, # 変更不可` in
   `default_config.json`), giving the student a 100 fps output rate. HuBERT
   BASE is 50 fps from 16 kHz — exactly 2× ratio.

---

## v4 recipe — Soft-VC unit classification (CURRENT, `train_v4.py`)

| Component | Choice | Why |
|---|---|---|
| **Student** | `PhoneExtractor()` default ctor | Bit-compatible with Beatrice |
| **Teacher** | `HUBERT_BASE` layer **9**, frozen | English SSL; richest phonetic layer |
| **Units** | **k-means (K=500)** over HuBERT-L9 frames | The Soft-VC discrete units (cached in out-dir) |
| **Head** | `nn.Linear(128, K)`, training-only | Maps soft unit → unit logits; **dropped at export** |
| **Loss** | **cross-entropy** vs each frame's unit id | The categorical objective that preserves richness |
| **Frame rate** | teacher 50 fps → labels `repeat_interleave(2)` → 100 fps | Aligns to student |
| **Optimizer / schedule** | `AdamW(5e-4)`, warmup→cosine | Same as v1 |
| **Init** | from scratch (or `--init-from`) | From scratch isolates objective vs `en_clean` |

```bash
uv run python -m phone_extractor_trainer.train_v4 \
    --data-dir datasets/librispeech/LibriSpeech/train-clean-100 \
    --out-dir outputs/phone_extractor_en_v4 \
    --n-clusters 500 --steps 200000
```

---

## (Superseded) v1/v2 regression recipe — `train.py`

| Component | Choice | Why |
|---|---|---|
| **Student** | `PhoneExtractor()` with default ctor | Architecturally identical to Beatrice's expected file |
| **Teacher** | `torchaudio.pipelines.HUBERT_BASE`, layer **9** of 12 | English-only LibriSpeech-960h; layer 9 maximises phonetic information in probing studies of HuBERT BASE |
| **Frame-rate adapter** | Linear-interpolate teacher 2× → 100 fps | Cheaper than 2:1 pooling student; preserves temporal precision in supervision |
| **Projection head** | `nn.Linear(128, 768)` | Bridges student's 128-dim phone space to teacher's 768-dim feature space. Trainable during distillation, **discarded at export** |
| **Loss** | `(1 − cos_sim) + λ·MSE` (λ = 0.1) | Cosine drives directional alignment; small MSE term anchors magnitude |
| **Optimizer** | `AdamW(lr=5e-4, β=(0.9,0.98), wd=0.01)` | Standard for SSL/distillation |
| **Schedule** | Linear warmup → cosine decay | 5k warmup, decay to `lr_min = 5e-6` |
| **Grad clip** | 1.0 | |
| **AMP** | fp16 on CUDA | Teacher kept in fp32 for stable targets |
| **Crop length** | 4 s @ 16 kHz (must be `% 160 == 0`) | Matches Beatrice's `wav_length` convention |

---

## Data layout

Any folder of English audio works — single flat dir, nested speakers,
audiobook chapters, anything. The dataset scans recursively for
`.wav .flac .mp3 .ogg .m4a .aac .opus`.

```
english_audio/
  libritts_train_clean/...
  vctk/p225/p225_001.wav
  podcasts/episode_001.mp3
  ...
```

**Recommended public corpora** (English, 16 kHz-ready):

- **LibriSpeech-960h** — the gold standard. Already what HuBERT BASE was
  trained on, so the student aligns extremely cleanly.
- **LibriTTS-R** — already used by Beatrice's vocoder pretrain
  (`@/assets/pretrained/151_checkpoint_libritts_r_200_02750000.pt.gz`).
- **CommonVoice (English)** — diverse accents, lots of data; pair with
  LibriSpeech for accent robustness.

For a healthy distillation, aim for **≥500 hours** of audio. The shipped
checkpoint was trained for 3M steps — plan for ~200k–1M steps depending on
your compute and quality target.

---

## Usage

### 1. Train

From the repo root:

```bash
uv run python -m phone_extractor_trainer.train \
    --data-dir /path/to/english_audio \
    --out-dir  outputs/phone_extractor_en \
    --steps 200000 \
    --batch-size 32 \
    --num-workers 4
```

Optionally warm-start from the shipped Japanese checkpoint — converges much
faster than from scratch, retains the architectural priors, just retargets
the output space toward English HuBERT:

```bash
uv run python -m phone_extractor_trainer.train \
    --data-dir /path/to/english_audio \
    --out-dir  outputs/phone_extractor_en \
    --init-from assets/pretrained/122_checkpoint_03000000.pt \
    --steps 100000
```

### 2. Watch progress

```bash
uv run python -m tensorboard.main --logdir outputs/phone_extractor_en
```

Key signal: `train/cos_sim` should climb from ~0 toward **0.7–0.9** as the
student learns to mimic HuBERT layer 9. If it plateaus below ~0.5, either
the teacher layer is wrong, the LR is too low, or the data is too small.

### 3. Resume

```bash
uv run python -m phone_extractor_trainer.train ... --resume
```

(Same flags as the original launch; reads `outputs/.../checkpoint_latest.pt`.)

### 4. Export to Beatrice format

```bash
uv run python -m phone_extractor_trainer.export \
    outputs/phone_extractor_en/checkpoint_latest.pt \
    assets/pretrained/phone_extractor_en.pt
```

This strips the training-only projection head and verifies the result loads
cleanly into a fresh `PhoneExtractor()`.

### 5. Plug into Beatrice

Edit `outputs/<your_dataset>/config.json` (or `assets/default_config.json`
if you want it as the new default) and change:

```json
"phone_extractor_file": "assets/pretrained/phone_extractor_en.pt"
```

Then re-run training as usual:

```bash
make clean DATASET=lol_data    # to drop the old VQ codebooks
make
```

The VQ codebooks need to be rebuilt because they're keyed to the
PhoneExtractor's output space — once you swap the extractor, the old
codebooks are meaningless. `make clean` + `make` does this automatically.

---

## Compute notes

On the RTX 3080 Ti box this repo is developed on:

| Setting | Throughput |
|---|---|
| `batch_size=32`, fp16, num_workers=4, 4s crops @ 16 kHz | ~6 it/s |
| → 200k steps | ~9 h |
| VRAM | ~6 GB (student + HuBERT teacher in fp32) |

The bottleneck is the HuBERT forward pass; reducing batch size or distilling
from a smaller teacher (e.g. WavLM Base+ via `s3prl`) would speed things up
proportionally.

---

## Caveats and known limitations

- **Teacher choice locks the phonetic prior.** This trainer uses
  English-LibriSpeech HuBERT BASE. The student will learn an English-aligned
  phonetic space. For multilingual VC, you'd want a teacher trained on a
  larger mix (e.g. `WAV2VEC2_XLSR_53`, also available via
  `torchaudio.pipelines`).
- **Layer 9 is a default, not a law.** Some downstream tasks prefer layer 6,
  7, or 11. If your end-to-end VC quality is poor, try `--teacher-layer 6`
  and re-train.
- **HuBERT BASE is 95 M params.** You'll download it on first run (~370 MB).
  It's frozen in eval mode throughout.
- **The published 3M-step checkpoint is a *hard* baseline to beat in
  absolute quality.** At 200k steps you're at ~7 % of its compute. For a
  fine-tune from `--init-from <shipped_ckpt>`, you can plausibly close the
  gap on English-specific tasks in 50k–100k steps; from scratch, expect
  500k+ for parity.
