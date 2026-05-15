# beatrice_trainer

The main voice conversion model trainer for **Beatrice 2**. This module trains
the `ConverterNetwork` (generator) and `MultiPeriodDiscriminator` that perform
the actual voice conversion, using frozen `PhoneExtractor` and `PitchEstimator`
as feature extractors.

---

## Architecture Overview

```
                    ┌─────────────────────┐
                    │   Input Waveform    │
                    │    (16 kHz mono)    │
                    └──────────┬──────────┘
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
           ▼                   ▼                   ▼
   ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
   │PhoneExtractor │   │PitchEstimator │   │    Energy     │
   │   (frozen)    │   │   (frozen)    │   │  Extraction   │
   └───────┬───────┘   └───────┬───────┘   └───────┬───────┘
           │                   │                   │
           │ 128-dim           │ 448 bins          │ 1-dim
           │ phones            │ pitch             │ energy
           │                   │                   │
           └───────────────────┼───────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   VectorQuantizer   │
                    │ (per-speaker, 512)  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  ConverterNetwork   │
                    │    (WaveGenerator)  │
                    │  + Speaker Embed    │
                    │  + Cross-Attention  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Output Waveform   │
                    │    (24 kHz mono)    │
                    └─────────────────────┘
```

### Key Components

| Component | Description |
|-----------|-------------|
| **PhoneExtractor** | Extracts 128-dim phonetic features at 100 fps. Frozen during training. |
| **PitchEstimator** | Predicts pitch bins (448 classes) at 100 fps. Frozen during training. |
| **VectorQuantizer** | Per-speaker codebook (512 entries) that captures speaker identity from phone features. Initialized via k-means on speaker's audio. |
| **ConverterNetwork** | The main generator. Uses ConvNeXt backbone + cross-attention for speaker conditioning. Outputs 24 kHz waveform. |
| **MultiPeriodDiscriminator** | GAN discriminator with multiple period sub-discriminators for adversarial training. |

---

## Training Recipe

| Setting | Value | Notes |
|---------|-------|-------|
| **Input sample rate** | 16 kHz | Fixed, do not change |
| **Output sample rate** | 24 kHz | Fixed, do not change |
| **Batch size** | 8 | Optimal for RTX 3080 Ti / 4070 Ti (12 GB VRAM) |
| **Default steps** | 50,000 | ~6 hours on RTX 4090, ~12 hours on RTX 3080 Ti |
| **Optimizer** | AdamW | Separate for generator and discriminator |
| **LR schedule** | Exponential decay | Easier to extend training |
| **Gradient clipping** | 1.0 | |
| **AMP** | Enabled | fp16 mixed precision |

### Loss Functions

| Loss | Weight | Purpose |
|------|--------|---------|
| **Adversarial (G)** | 1.0 | GAN generator loss |
| **Feature matching** | 2.0 | Match discriminator intermediate features |
| **Multi-scale mel** | 45.0 | Spectral reconstruction |
| **D4C aperiodicity** | 1.0 | Aperiodic component quality |
| **Loudness** | 1.0 | Volume consistency |

### Data Augmentation

| Augmentation | Probability | Purpose |
|--------------|-------------|---------|
| **Noise injection** | 0.5 | Robustness to noisy input |
| **Reverb** | 0.5 | Room acoustics robustness |
| **Formant shift** | 0.5 | Speaker identity disentanglement |
| **Low-pass filter** | 0.2 | Bandwidth robustness |

---

## Data Layout

```
your_training_data/
├── speaker_a/
│   ├── audio_001.wav
│   ├── audio_002.flac
│   └── ...
├── speaker_b/
│   ├── recording_01.wav
│   └── ...
└── speaker_c/
    └── ...
```

**Requirements:**
- One subdirectory per speaker (speaker name = directory name)
- Audio files can be nested within speaker directories
- Supported formats: `.wav`, `.flac`, `.mp3`, `.ogg`, `.m4a`, `.aac`
- Minimum: **~2 minutes** of varied speech per speaker
- Recommended: **5-10 minutes** per speaker for best quality

### Single Speaker Training

Even for single-speaker training, you must create a speaker subdirectory:

```
# WRONG - won't work
your_data/
├── audio_001.wav
└── audio_002.wav

# CORRECT
your_data/
└── my_speaker/
    ├── audio_001.wav
    └── audio_002.wav
```

---

## Usage

### Basic Training

```bash
# Using Makefile (recommended)
make DATASET=my_dataset

# Or directly
uv run python -m beatrice_trainer \
    -d preprocessed/my_dataset \
    -o outputs/my_dataset
```

### Resume Training

```bash
# Resume without re-preprocessing (recommended)
make resume DATASET=my_dataset

# Resume with re-preprocessing (if you changed input audio)
make train RESUME=1 DATASET=my_dataset
```

### Monitor Progress

```bash
make tensorboard DATASET=my_dataset
```

### Custom Configuration

1. Copy the default config:
   ```bash
   cp assets/default_config.json outputs/my_dataset/config.json
   ```

2. Edit the config file to change hyperparameters

3. Train with custom config:
   ```bash
   uv run python -m beatrice_trainer \
       -d preprocessed/my_dataset \
       -o outputs/my_dataset \
       -c outputs/my_dataset/config.json
   ```

---

## Configuration Reference

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_steps` | 50000 | Total training steps |
| `batch_size` | 8 | Batch size (reduce if OOM) |
| `num_workers` | 16 | DataLoader workers |
| `learning_rate` | 2e-4 | Initial learning rate |
| `lr_decay` | 0.999 | Exponential decay factor |

### Pretrained Models

| Parameter | Default | Description |
|-----------|---------|-------------|
| `phone_extractor_file` | `assets/pretrained/122_checkpoint_03000000.pt` | PhoneExtractor checkpoint |
| `pitch_estimator_file` | `assets/pretrained/104_3_checkpoint_00300000.pt` | PitchEstimator checkpoint |
| `pretrained_file` | `assets/pretrained/151_checkpoint_libritts_r_200_02750000.pt.gz` | Generator pretrain |

### Augmentation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `augmentation_noise_probability` | 0.5 | Noise injection probability |
| `augmentation_reverb_probability` | 0.5 | Reverb probability |
| `augmentation_formant_shift_probability` | 0.5 | Formant shift probability |
| `augmentation_lpf_probability` | 0.2 | Low-pass filter probability |

---

## Output Files

After training, the output directory contains:

| File/Directory | Description |
|----------------|-------------|
| `paraphernalia_<name>_<step>/` | **Deployable model** — load this in VST/VCClient |
| `checkpoint_<name>_<step>.pt.gz` | Training checkpoint (can resume from) |
| `checkpoint_latest.pt.gz` | Latest checkpoint (auto-updated) |
| `config.json` | Configuration used for training |
| `events.out.tfevents.*` | TensorBoard logs |

### Using the Trained Model

The `paraphernalia_*` directory can be loaded directly in:
- [Beatrice 2 VST](https://prj-beatrice.com)
- [VCClient](https://github.com/w-okada/voice-changer)
- [beatrice-client](https://github.com/aq2r/beatrice-client)

---

## Multi-Speaker Training

The trainer handles multiple speakers natively. Each speaker gets:
- Separate VQ codebook (512 entries)
- Separate speaker embedding
- Separate key-value embedding for cross-attention

**Scaling guidance:**
- `n_steps` should scale with speaker count
- Aim for **≥20k updates per speaker**
- Formula: `per_speaker_updates ≈ n_steps × batch_size ÷ n_speakers`

| Speakers | Recommended `n_steps` |
|----------|----------------------|
| 1-2 | 30,000-50,000 |
| 5-10 | 50,000-80,000 |
| 10-20 | 80,000-120,000 |

---

## Performance Notes

### RTX 3080 Ti / RTX 4070 Ti (12 GB VRAM)

| Metric | Value |
|--------|-------|
| Throughput | ~2.4 it/s |
| Time per 10k steps | ~70 min |
| VRAM usage | ~9 GB |
| Bottleneck | CPU (data augmentation) |

### Optimization Tips

1. **Don't increase batch size beyond 8** — VRAM pressure causes cuDNN to pick slow algorithms
2. **Keep `num_workers: 16`** — lower values cause data starvation
3. **`torch.compile` doesn't work on Windows** — Triton unavailable
4. **Reduce `augmentation_reverb_probability`** to 0.2 for faster training (quality tradeoff)

---

## Troubleshooting

### Out of Memory (OOM)

- Reduce `batch_size` to 4 or 6
- Reduce `num_workers` to 8
- Disable `compile_*` options

### Poor Voice Quality

- Increase `n_steps` (try 80,000-100,000)
- Add more training data per speaker
- Check that audio is clean (no background noise/music)

### Japanese Accent in English Conversion

The default PhoneExtractor was trained on Japanese data. Solutions:
1. Use the English-retrained PhoneExtractor:
   ```json
   "phone_extractor_file": "assets/pretrained/phone_extractor_en.pt"
   ```
2. Or retrain it yourself: `make phone-train-en`

### Poor Pitch Tracking (Cross-Gender)

The default PitchEstimator may struggle with extreme pitch ranges. Solutions:
1. Use a retrained PitchEstimator:
   ```json
   "pitch_estimator_file": "assets/pretrained/pitch_estimator_v2.pt"
   ```
2. Or retrain it yourself: `make pitch-train-vctk`

---

## References

Key papers and implementations used in Beatrice 2:

| Component | Reference |
|-----------|-----------|
| PhoneExtractor | [Soft-VC](https://arxiv.org/abs/2111.02392) + [wav2vec 2.0](https://arxiv.org/abs/2006.11477) |
| PitchEstimator | [Subramani et al., 2024](https://arxiv.org/abs/2309.14507) |
| VQ Codebook | [kNN-VC](https://arxiv.org/abs/2305.18975) |
| Generator | [HiFi-GAN](https://arxiv.org/abs/2010.05646) + [Vocos](https://arxiv.org/abs/2306.00814) |
| Discriminator | [HiFi-GAN](https://arxiv.org/abs/2010.05646) + [UnivNet](https://arxiv.org/abs/2106.07889) |
| Cross-Attention | [FragmentVC](https://arxiv.org/abs/2010.14150) |
| D4C Loss | [WORLD vocoder](https://www.sciencedirect.com/science/article/pii/S0167639316300413) |
| Multi-scale Mel | [Descript Audio Codec](https://arxiv.org/abs/2306.06546) |

---

## License

MIT License. See [LICENSE](../LICENSE) for details.
