# Training Log & Experimental Findings

A running log of experiments, hyperparameter tuning, data scaling, and model behavior for the Beatrice Trainer fork.

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

### Phase B: Balanced Regularized Fine-Tuning (Current Run)
* **Configuration**: `phone_noise_ratio: 0.5`, `floor_noise_level: 1e-4` (-80dB), `augmentation_reverb_probability: 0.5`, `augmentation_formant_shift_probability: 0.5`
* **Theory**: Keep the original robust augmentation probabilities so the model has enough regularization to handle the small dataset size without overfitting. Reduce the static footprint by dropping `floor_noise_level` from `1e-3` (too loud) to `1e-4` (quiet but present for numerical gradient stability).
* **Result**: **In Progress (Active)**.

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
