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

---

## 3. Directory Layout Conventions

- `inputs/<dataset_name>/<speaker_name>/`: Raw source audio files.
- `preprocessed/<dataset_name>/<speaker_name>/`: Segmented clips outputted by `preprocess.py` (24kHz, mono, PCM_16).
- `outputs/<dataset_name>/`: Checkpoints and exported paraphernalia voice packages.

---

## 4. Current Work & Goals
- **Active Task**: Fine-tuning Beatrice voice models for `new_lol_data` (6 speakers: sion, teemo, demacia_male, noxus_male, yordle_female, yordle_male).
- **Setup**: Training is running with the 580k English PhoneExtractor, the V2 VCTK Pitch Estimator, and a robust, balanced regularization config (60,000 steps).
