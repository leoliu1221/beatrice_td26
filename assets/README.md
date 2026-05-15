# Assets

This directory contains pretrained models, augmentation data, and test files used by the Beatrice voice conversion system.

---

## Training Pipeline Overview

Beatrice uses a **three-stage training pipeline**. Each stage produces a specialized model that feeds into the final voice conversion system:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         BEATRICE TRAINING PIPELINE                          │
└─────────────────────────────────────────────────────────────────────────────┘

  Stage 1: Phone Extractor          Stage 2: Pitch Estimator
  ─────────────────────────         ────────────────────────
  Input: Raw audio                  Input: Raw audio
  Teacher: HuBERT (SSL model)       Teacher: pyworld DIO+StoneMask
  Output: 128-dim phonetic          Output: 448-bin pitch
          features @ 100 fps                classification @ 100 fps

         │                                   │
         │                                   │
         ▼                                   ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │                     Stage 3: Beatrice Trainer                            │
  │                                                                          │
  │   ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────┐  │
  │   │   Phone     │    │   Pitch     │    │     ConverterNetwork        │  │
  │   │  Extractor  │───▶│  Estimator  │───▶│  (Generator + Discriminator)│  │
  │   │  (frozen)   │    │  (frozen)   │    │       (trainable)           │  │
  │   └─────────────┘    └─────────────┘    └─────────────────────────────┘  │
  │                                                                          │
  │   + Data Augmentation (IR reverb, noise, formant shift, LPF)             │
  │   + Per-speaker VQ codebooks                                             │
  │   + Speaker embeddings                                                   │
  └──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                         Final Voice Conversion Model
                         (paraphernalia directory)
```

---

## Stage 1: Phone Extractor

**Purpose**: Extracts language-independent phonetic content from speech, discarding speaker identity.

**How it works**:
- Uses **knowledge distillation** from HuBERT (a self-supervised speech model)
- Student network learns to predict HuBERT layer 9 features (768-dim) from raw audio
- Projects to 128-dim continuous phonetic features at 100 fps
- Architecture based on wav2vec 2.0 feature extractor (6 conv layers, stride 160)

**Training details**:
| Component | Choice |
|---|---|
| Teacher | `torchaudio.pipelines.HUBERT_BASE`, layer 9 |
| Loss | Cosine similarity + MSE |
| Input | 16 kHz audio, 4s crops |
| Output | 128-dim features @ 100 fps |

**Pretrained file**: `pretrained/122_checkpoint_03000000.pt` (Japanese-leaning corpus)

**Retraining**: See `phone_extractor_trainer/README.md` for English-only retraining.

---

## Stage 2: Pitch Estimator

**Purpose**: Estimates fundamental frequency (F0) from speech for pitch-aware voice conversion.

**How it works**:
- Supervised training using pyworld's DIO+StoneMask as ground truth
- Classifies into 448 pitch bins:
  - Bin 0 = unvoiced
  - Bins 1-447 = 55 Hz (A1) to ~1390 Hz (F6), 96 bins/octave
- Uses instantaneous frequency features + autocorrelation (YIN-style)
- ConvNeXt backbone (9 blocks, 192 channels)

**Training details**:
| Component | Choice |
|---|---|
| Teacher | pyworld DIO + StoneMask |
| Loss | Cross-entropy over 448 bins |
| Input | 16 kHz audio |
| Output | Pitch classification @ 100 fps |

**Pretrained file**: `pretrained/104_3_checkpoint_00300000.pt`

**Retraining**: See `pitch_estimator_trainer/README.md` for improved pitch coverage.

---

## Stage 3: Beatrice Trainer (Final Voice Conversion)

**Purpose**: Trains the voice conversion model that transforms source speech to target speaker voice.

**How it works**:
1. **Feature extraction** (frozen models):
   - PhoneExtractor → phonetic content (what is being said)
   - PitchEstimator → pitch contour (intonation)

2. **Voice conversion** (trainable):
   - Per-speaker VQ codebooks (kNN-VC style) for speaker identity
   - Cross-attention speaker embeddings
   - WaveGenerator vocoder with FIR postfilter

3. **Data augmentation** (robustness):
   - IR convolution for reverb simulation
   - Noise mixing at various SNRs
   - Formant shifting
   - Low-pass filtering

**Training details**:
| Component | Choice |
|---|---|
| Generator | ConverterNetwork with cross-attention |
| Discriminator | Multi-period + multi-resolution |
| Loss | Mel + loudness + adversarial + feature matching + D4C |
| Augmentation | Reverb, noise, formant shift, LPF |

**Pretrained file**: `pretrained/151_checkpoint_libritts_r_200_02750000.pt.gz`

---

## How the Models Work Together

At **inference time**, the pipeline processes audio as follows:

```
Input Audio (any speaker)
        │
        ▼
┌───────────────────┐
│  Phone Extractor  │──▶ Phonetic features (what is said)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Pitch Estimator  │──▶ Pitch contour (how it's said)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  VQ Codebook      │──▶ Map to target speaker's phonetic space
│  (per-speaker)    │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  WaveGenerator    │──▶ Synthesize waveform in target voice
│  + Speaker Embed  │
└───────────────────┘
        │
        ▼
Output Audio (target speaker)
```

**Key insight**: The PhoneExtractor and PitchEstimator are **frozen** during Beatrice training. They provide a stable, speaker-independent representation that the ConverterNetwork learns to map to target speakers.

---

## Augmentation Data

### IR (Impulse Responses)

[Room Impulse Response Generator](https://github.com/audiolabs/rir-generator) によって生成されたインパルス応答データです。

**Used for**: Simulating room acoustics during training. A random IR is convolved with audio (50% probability) to make the model robust to reverberant recordings.

### Noise

[DNS-Challenge](https://github.com/microsoft/DNS-Challenge) で提供されているノイズデータのサブセットをダウンサンプルしたものであり、以下を含みます。

* Audioset: https://research.google.com/audioset/index.html; License: https://creativecommons.org/licenses/by/4.0/
* Freesound: https://freesound.org/ Only files with CC0 licenses were selected; License: https://creativecommons.org/publicdomain/zero/1.0/
* Demand: https://zenodo.org/record/1227121#.XRKKxYhKiUk; License: https://creativecommons.org/licenses/by-sa/3.0/deed.en_CA

**Used for**: Mixing background noise at random SNRs (20–45 dB) to make the model robust to noisy input.

### Augmentation Summary

The `augment_audio()` function in `beatrice_trainer/__main__.py` applies:

| Augmentation | Probability | Purpose |
|---|---|---|
| **Reverb (IR convolution)** | 50% | Room acoustics robustness |
| **Noise mixing** | Always | Background noise robustness |
| **Formant shift** | 50% | Vocal timbre variation |
| **Low-pass filter** | 20% | Bandwidth-limited recording simulation |
| **Random filtering** | Always | Channel variation robustness |

---

## Pretrained Models

Beatrice の事前学習済みモデルです。
[ReazonSpeech](https://huggingface.co/datasets/reazon-research/reazonspeech), [VocalSet](https://zenodo.org/records/1193957), [DNS-Challenge](https://github.com/microsoft/DNS-Challenge), [LibriTTS-R](https://www.openslr.org/141/) のデータを使用して学習されています。

| File | Stage | Description |
|---|---|---|
| `122_checkpoint_03000000.pt` | Phone Extractor | HuBERT distillation, Japanese-leaning corpus |
| `104_3_checkpoint_00300000.pt` | Pitch Estimator | pyworld supervision, VocalSet + speech data |
| `151_checkpoint_libritts_r_200_02750000.pt.gz` | Beatrice (base) | LibriTTS-R pretrained generator |

---

## Test

[Common Voice](https://commonvoice.mozilla.org) のサブセットをダウンサンプルしたものであり、オリジナルのデータは CC0 でライセンスされています。
読み上げられている文は、[青空文庫](https://www.aozora.gr.jp)に掲載されている著作権保護期間の満了した作品の一部です。

* common_voice_ja_38833628
  * "「やっぱりお化けや幽霊じゃないんだ。ああして歩いているところをみると、人間にちがいない」"
  * 江戸川乱歩 『少年探偵団』 https://www.aozora.gr.jp/cards/001779/files/56669_58756.html
* common_voice_ja_38843402
  * "「こりゃきっと仲間によくないことがあったにちがいない。」と小悪魔は考えました。"
  * トルストイ 『イワンの馬鹿』 菊池寛訳 https://www.aozora.gr.jp/cards/000361/files/42941_15672.html
* common_voice_ja_38852485
  * "すると、ブランコ乗りは突然泣き始めた。すっかり驚いた興行主は飛び上がり、いったいどうしたのか、とたずねた。"
  * フランツ・カフカ 『最初の苦悩』 原田義人訳 https://www.aozora.gr.jp/cards/001235/files/49861_41921.html
* common_voice_ja_38853932
  * "王もこのやり方は喜んでいません。それにもう一つ、これには困ることがあるのです。"
  * ジョナサン・スイフト 『ガリバー旅行記』 原民喜訳 https://www.aozora.gr.jp/cards/000912/files/4673_9768.html
* common_voice_ja_38864552
  * "ヘンゼルは屋根が、とてもおいしかったので、大きなやつを、一枚、そっくりめくってもって来ました。"
  * グリム兄弟 『ヘンゼルとグレーテル』 楠山正雄訳 https://www.aozora.gr.jp/cards/001091/files/42315_15931.html
* common_voice_ja_38878413
  * "私があまりあけすけに、陛下に申し上げたので、それが、皇帝のお気にさわったらしいのです。陛下は議会で、私の考えを、それとなく非難されました。"
  * ジョナサン・スイフト 『ガリバー旅行記』 原民喜訳 https://www.aozora.gr.jp/cards/000912/files/4673_9768.html
* common_voice_ja_38898180
  * "「君となら話すかい？」と、Ｋはきいた。「わたしもだめよ」と、フリーダがいう。「あなたもだめよ、わたしもだめよ。まったくできないことなのよ」"
  * フランツ・カフカ 『城』 原田義人訳 https://www.aozora.gr.jp/cards/001235/files/49862_45839.html
* common_voice_ja_38925334
  * "それで海の中へ落ちたことがはじめてわかりました。箱は私の身体や家具などの重みで、水の中に浸りながら浮いています。"
  * ジョナサン・スイフト 『ガリバー旅行記』 原民喜訳 https://www.aozora.gr.jp/cards/000912/files/4673_9768.html
