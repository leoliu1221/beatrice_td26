"""Audio dataset for PitchEstimator training.

Recursively scans a root directory for audio files, resamples each on the fly
to 16 kHz mono, extracts ground-truth F0 using pyworld DIO, and yields
(waveform, f0_bins) pairs for supervised pitch estimation training.

For best results, use diverse pitch data:
- Male and female speakers across age ranges
- Singing voice datasets (VocalSet, NUS-48E, etc.)
- Speech datasets with varied pitch ranges

Noise-robust mode
-----------------
When `noise_files` and `ir_files` are passed, each yielded waveform is
augmented with Beatrice's `augment_audio()` (noise + reverb + LPF + formant
shift) while the F0 label is still computed from the **clean** waveform.
This teaches the estimator to predict pitch from corrupted audio, closing
the train/test gap with Beatrice's main trainer (which always feeds the
estimator noisy audio).
"""
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pyworld
import torch
import torchaudio
from torch.utils.data import Dataset

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".opus"}

# Make distill_augment importable from worker processes.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Cache per-process resamplers (one per source sample rate) to avoid rebuilding
# the kernel on every __getitem__ call.
_RESAMPLER_CACHE: dict[tuple[int, int], torchaudio.transforms.Resample] = {}


def _get_resampler(src_sr: int, dst_sr: int) -> torchaudio.transforms.Resample:
    key = (src_sr, dst_sr)
    if key not in _RESAMPLER_CACHE:
        _RESAMPLER_CACHE[key] = torchaudio.transforms.Resample(src_sr, dst_sr)
    return _RESAMPLER_CACHE[key]


def discover_audio_files(root: Path) -> list[Path]:
    """Return every audio file beneath `root` (recursive)."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"data dir not found: {root}")
    files = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    ]
    if not files:
        raise RuntimeError(f"no audio files found under {root}")
    return sorted(files)


def f0_to_pitch_bin(
    f0: np.ndarray,
    f0_floor: float = 55.0,
    pitch_bins_per_octave: int = 96,
) -> np.ndarray:
    """Convert F0 in Hz to pitch bin indices.
    
    Bin 0 = unvoiced (f0 <= 0)
    Bins 1-447 = voiced pitches starting at f0_floor Hz
    
    Formula: bin = round(log2(f0 / f0_floor) * pitch_bins_per_octave) + 1
    """
    bins = np.zeros_like(f0, dtype=np.int64)
    voiced_mask = f0 > 0
    if voiced_mask.any():
        voiced_f0 = f0[voiced_mask]
        # Clamp to valid range before log
        voiced_f0 = np.clip(voiced_f0, f0_floor, None)
        bin_values = np.round(
            np.log2(voiced_f0 / f0_floor) * pitch_bins_per_octave
        ).astype(np.int64) + 1
        # Clamp to valid bin range [1, 447]
        bin_values = np.clip(bin_values, 1, 447)
        bins[voiced_mask] = bin_values
    return bins


def extract_f0_pyworld(
    wav: np.ndarray,
    sample_rate: int = 16000,
    hop_length: int = 160,
    f0_floor: float = 55.0,
    f0_ceil: float = 1400.0,
) -> np.ndarray:
    """Extract F0 using pyworld DIO + StoneMask refinement.
    
    Returns F0 in Hz at 10ms frame rate (hop_length=160 @ 16kHz).
    Unvoiced frames have F0 = 0.
    """
    wav_np = wav.astype(np.float64)
    frame_period = hop_length * 1000 / sample_rate  # ms
    
    f0, t = pyworld.dio(
        wav_np,
        sample_rate,
        f0_floor=f0_floor,
        f0_ceil=f0_ceil,
        frame_period=frame_period,
    )
    f0 = pyworld.stonemask(wav_np, f0, t, sample_rate)
    return f0.astype(np.float32)


class PitchDataset(Dataset):
    """Random fixed-length 16 kHz mono crops with ground-truth pitch bins.

    `__len__` is set by `samples_per_epoch` rather than `len(files)`, so each
    epoch is a fixed number of random crops regardless of dataset size.
    """

    def __init__(
        self,
        files: Sequence[Path],
        wav_length: int = 64000,  # 4s @ 16 kHz
        samples_per_epoch: int = 50000,
        sample_rate: int = 16000,
        hop_length: int = 160,
        f0_floor: float = 55.0,
        f0_ceil: float = 1400.0,
        pitch_bins_per_octave: int = 96,
        seed: int | None = None,
        noise_files: Sequence[Path] | None = None,
        ir_files: Sequence[Path] | None = None,
        aug_kwargs: dict | None = None,
    ):
        if wav_length <= 0:
            raise ValueError("wav_length must be positive")
        if wav_length % hop_length != 0:
            raise ValueError(f"wav_length must be divisible by hop_length ({hop_length})")
        self.files = list(files)
        self.wav_length = wav_length
        self.samples_per_epoch = samples_per_epoch
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.f0_floor = f0_floor
        self.f0_ceil = f0_ceil
        self.pitch_bins_per_octave = pitch_bins_per_octave
        self._rng = random.Random(seed)
        # Noise-robust mode: when set, the returned waveform is augmented while
        # the F0 label is still computed from the clean wav. We default
        # formant_shift_probability to 0 because formant shifts can subtly
        # alter pyworld's F0 estimate (the `random_formant_shift` in Beatrice
        # is a spectral-envelope-warp + resample combo, not pure LPC), which
        # would silently corrupt the labels. The other Beatrice augmentations
        # (noise/reverb/LPF) leave F0 untouched.
        if (noise_files is None) != (ir_files is None):
            raise ValueError("noise_files and ir_files must be both set or both None")
        self.noise_files = list(noise_files) if noise_files is not None else None
        self.ir_files = list(ir_files) if ir_files is not None else None
        merged_aug = {"formant_shift_probability": 0.0}
        if aug_kwargs:
            merged_aug.update(aug_kwargs)
        self.aug_kwargs = merged_aug
        self.augment_enabled = self.noise_files is not None

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _load_random_crop(self, path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        wav, sr = torchaudio.load(str(path), backend="soundfile")
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)  # to mono
        if sr != self.sample_rate:
            wav = _get_resampler(sr, self.sample_rate)(wav)
        wav = wav.squeeze(0)  # [T]
        total = wav.size(0)
        
        # Check for pre-computed F0
        f0_path = path.with_suffix(path.suffix + ".f0.npy")
        has_precomputed_f0 = f0_path.exists()
        
        if has_precomputed_f0:
            # Load pre-computed F0 (much faster!)
            f0_full = np.load(f0_path)
            f0_frames = len(f0_full)
            wav_frames = total // self.hop_length
            
            # Determine crop position
            crop_frames = self.wav_length // self.hop_length
            if wav_frames <= crop_frames:
                start_frame = 0
                start_sample = 0
            else:
                max_start_frame = wav_frames - crop_frames
                start_frame = self._rng.randint(0, max_start_frame)
                start_sample = start_frame * self.hop_length
            
            # Crop waveform
            if total < self.wav_length:
                if total < self.wav_length // 2:
                    repeats = (self.wav_length // total) + 1
                    wav = wav.repeat(repeats)[:self.wav_length]
                else:
                    pad = self.wav_length - total
                    wav = torch.nn.functional.pad(wav, (0, pad), mode="constant", value=0.0)
            else:
                wav = wav[start_sample : start_sample + self.wav_length]
            
            # Crop F0
            if f0_frames <= crop_frames:
                f0 = f0_full
            else:
                f0 = f0_full[start_frame : start_frame + crop_frames]
        else:
            # Fall back to on-the-fly extraction
            if total < self.wav_length:
                if total < self.wav_length // 2:
                    repeats = (self.wav_length // total) + 1
                    wav = wav.repeat(repeats)[:self.wav_length]
                else:
                    pad = self.wav_length - total
                    wav = torch.nn.functional.pad(wav, (0, pad), mode="constant", value=0.0)
            else:
                start = self._rng.randint(0, total - self.wav_length)
                wav = wav[start : start + self.wav_length]
            
            wav_np = wav.numpy()
            f0 = extract_f0_pyworld(
                wav_np,
                sample_rate=self.sample_rate,
                hop_length=self.hop_length,
                f0_floor=self.f0_floor,
                f0_ceil=self.f0_ceil,
            )
        
        # Convert F0 to pitch bins
        pitch_bins = f0_to_pitch_bin(
            f0,
            f0_floor=self.f0_floor,
            pitch_bins_per_octave=self.pitch_bins_per_octave,
        )
        
        # Expected length: wav_length // hop_length
        expected_len = self.wav_length // self.hop_length
        if len(pitch_bins) < expected_len:
            pitch_bins = np.pad(pitch_bins, (0, expected_len - len(pitch_bins)), mode='edge')
        elif len(pitch_bins) > expected_len:
            pitch_bins = pitch_bins[:expected_len]
        
        return wav, torch.from_numpy(pitch_bins).long()

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (wav, pitch_bins). In noise-robust mode `wav` is augmented
        but `pitch_bins` are computed from the clean source."""
        last_err: Exception | None = None
        for _ in range(8):
            path = self._rng.choice(self.files)
            try:
                clean_wav, pitch_bins = self._load_random_crop(path)
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue
            if not self.augment_enabled:
                return clean_wav, pitch_bins
            # Apply Beatrice's augment_audio to a clone; keep clean pitch_bins.
            from distill_augment import apply_augmentation

            try:
                noisy_wav = apply_augmentation(
                    clean_wav.detach().clone(),
                    self.noise_files,
                    self.ir_files,
                    self.aug_kwargs,
                )
            except Exception as e:  # noqa: BLE001
                last_err = e
                noisy_wav = clean_wav.detach().clone()
            return noisy_wav, pitch_bins
        raise RuntimeError(f"could not load any audio after 8 tries; last error: {last_err}")
