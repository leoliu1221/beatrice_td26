"""Shared augmentation helper for noise-robust feature-extractor distillation.

Background
----------
Beatrice's main trainer (`beatrice_trainer.__main__`) calls `augment_audio()` on
the input waveform before feeding it to the PhoneExtractor and PitchEstimator.
This adds noise, reverb, LPF, formant shifts, etc. — the network learns to
extract content/pitch despite that corruption.

However, both the phone_extractor_trainer and pitch_estimator_trainer originally
trained their networks on **clean** audio only. This created a train/test gap:
the feature extractors had never seen noisy inputs, but Beatrice (and real
inference) always gives them noisy inputs. The result is unstable features
during conversion → audible artifacts in the converted voice.

This helper wraps Beatrice's exact augmentation so both feature-extractor
trainers can re-train with `student(noisy)` matched against the teacher / F0
label computed from `clean`. This is a standard "noise-robust" or "consistency"
distillation setup.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Iterable

import torch

# Default augmentation hyperparameters lifted from assets/default_config.json
# so the feature-extractor training matches Beatrice's inference pipeline.
DEFAULT_AUG_KWARGS = dict(
    snr_candidates=[20.0, 25.0, 30.0, 35.0, 40.0, 45.0],
    formant_shift_probability=0.5,
    formant_shift_semitone_min=-3.0,
    formant_shift_semitone_max=3.0,
    reverb_probability=0.5,
    lpf_probability=0.2,
    lpf_cutoff_freq_candidates=[2000.0, 3000.0, 4000.0, 6000.0],
)

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".opus"}


def discover_aux_files(root: os.PathLike | str) -> list[Path]:
    """Recursively gather audio files under `root`. Used for both noise/ and ir/."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"aux dir not found: {root}")
    out = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
    if not out:
        raise RuntimeError(f"no audio files found under {root}")
    return out


def load_default_aux(repo_root: os.PathLike | str = ".") -> tuple[list[Path], list[Path]]:
    """Load `assets/noise/` and `assets/ir/` from the Beatrice repo."""
    repo = Path(repo_root)
    return discover_aux_files(repo / "assets" / "noise"), discover_aux_files(repo / "assets" / "ir")


def apply_augmentation(
    clean_16k: torch.Tensor,
    noise_files: list[Path],
    ir_files: list[Path],
    aug_kwargs: dict | None = None,
) -> torch.Tensor:
    """Run Beatrice's augment_audio on a 16 kHz clean mono waveform.

    Args:
        clean_16k: float tensor of shape [T] (1-D) or [1, T] at 16 kHz.
        noise_files / ir_files: lists from discover_aux_files() or load_default_aux().
        aug_kwargs: override any DEFAULT_AUG_KWARGS entries.

    Returns:
        noisy waveform, shape [T] at 16 kHz, float.
    """
    # Import lazily to avoid pulling beatrice_trainer at module import time
    # in CPU-only data workers.
    from beatrice_trainer.__main__ import augment_audio

    if clean_16k.dim() == 1:
        clean = clean_16k.unsqueeze(0)
    elif clean_16k.dim() == 2 and clean_16k.size(0) == 1:
        clean = clean_16k
    else:
        raise ValueError(f"clean_16k must be [T] or [1, T], got {tuple(clean_16k.shape)}")

    kw = dict(DEFAULT_AUG_KWARGS)
    if aug_kwargs:
        kw.update(aug_kwargs)

    noisy = augment_audio(
        clean,
        16000,
        noise_files,
        ir_files,
        snr_candidates=kw["snr_candidates"],
        formant_shift_probability=kw["formant_shift_probability"],
        formant_shift_semitone_min=kw["formant_shift_semitone_min"],
        formant_shift_semitone_max=kw["formant_shift_semitone_max"],
        reverb_probability=kw["reverb_probability"],
        lpf_probability=kw["lpf_probability"],
        lpf_cutoff_freq_candidates=kw["lpf_cutoff_freq_candidates"],
    )

    if noisy.dim() == 2:
        noisy = noisy.squeeze(0)

    # Safety: augment_audio can occasionally clip slightly; rescale if so.
    peak = noisy.abs().max()
    if peak > 0.999:
        noisy = noisy * (0.999 / peak)
    return noisy
