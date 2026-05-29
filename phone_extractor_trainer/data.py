"""Audio dataset for PhoneExtractor distillation training.

Recursively scans a root directory for audio files, resamples each on the fly
to 16 kHz mono, and yields random fixed-length crops as 1D float32 tensors.

This dataset is intentionally minimal: distillation does not require
silence-trimmed clips or speaker labels. Any English-speaking audio works
(audiobooks, LibriSpeech, CommonVoice, podcasts, ...).

Noise-robust mode
-----------------
When `noise_files` and `ir_files` are passed, each crop also produces a
**noisy** counterpart augmented with Beatrice's `augment_audio()` (noise,
reverb, LPF, formant shift). The trainer then matches `student(noisy)` against
`teacher(clean)` for consistency distillation. This closes the train/test gap
where Beatrice feeds the phone extractor noisy audio at conversion time but
the original distillation only saw clean LibriSpeech.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Sequence

import torch
import torchaudio
from torch.utils.data import Dataset

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".opus"}

# Make distill_augment importable when this file is loaded from a worker
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


class WavCropDataset(Dataset):
    """Random fixed-length 16 kHz mono crops from a flat pool of audio files.

    `__len__` is set by `samples_per_epoch` rather than `len(files)`, so each
    epoch is a fixed number of random crops regardless of dataset size. This
    is the convention used by most large-scale audio distillation pipelines.
    """

    def __init__(
        self,
        files: Sequence[Path],
        wav_length: int = 64000,  # 4s @ 16 kHz
        samples_per_epoch: int = 50000,
        sample_rate: int = 16000,
        seed: int | None = None,
        noise_files: Sequence[Path] | None = None,
        ir_files: Sequence[Path] | None = None,
        aug_kwargs: dict | None = None,
        aux_files: Sequence[Path] | None = None,
        aux_mix_ratio: float = 0.0,
    ):
        if wav_length <= 0:
            raise ValueError("wav_length must be positive")
        self.files = list(files)
        self.wav_length = wav_length
        self.samples_per_epoch = samples_per_epoch
        self.sample_rate = sample_rate
        self._rng = random.Random(seed)
        # Noise-robust mode: when noise_files and ir_files are both given,
        # __getitem__ returns (clean, noisy) at 16 kHz; otherwise returns
        # the clean crop alone (backward compatible).
        if (noise_files is None) != (ir_files is None):
            raise ValueError("noise_files and ir_files must be both set or both None")
        self.noise_files = list(noise_files) if noise_files is not None else None
        self.ir_files = list(ir_files) if ir_files is not None else None
        self.aug_kwargs = aug_kwargs
        self.augment_enabled = self.noise_files is not None
        # Auxiliary (target-domain) pool: on each __getitem__ call we sample
        # from `aux_files` with probability `aux_mix_ratio`, else from `files`.
        # Used to expose the student to the actual inference distribution
        # (e.g., the LoL TTS voices) during distillation, fighting the
        # LibriSpeech-only specialization seen at step 580k -> 704k.
        if aux_files is not None and not 0.0 <= aux_mix_ratio <= 1.0:
            raise ValueError("aux_mix_ratio must be in [0, 1]")
        self.aux_files = list(aux_files) if aux_files else None
        self.aux_mix_ratio = aux_mix_ratio if self.aux_files else 0.0

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _load_random_crop(self, path: Path) -> torch.Tensor:
        # frame_offset=-1 means "auto"; we'll random-crop after load. For very
        # long files, reading the whole thing is wasteful, but soundfile is
        # fast enough for typical audiobook-sized files. If you have hour-long
        # FLACs, prefer pre-segmenting them.
        wav, sr = torchaudio.load(str(path), backend="soundfile")
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)  # to mono
        if sr != self.sample_rate:
            wav = _get_resampler(sr, self.sample_rate)(wav)
        wav = wav.squeeze(0)  # [T]
        total = wav.size(0)
        if total < self.wav_length:
            # Pad short clips with reflection so the model always sees a
            # full-length crop. Short-clip-heavy corpora (CommonVoice) still
            # train fine this way.
            pad = self.wav_length - total
            wav = torch.nn.functional.pad(wav, (0, pad), mode="reflect" if total > 1 else "constant")
            return wav[: self.wav_length]
        start = self._rng.randint(0, total - self.wav_length)
        return wav[start : start + self.wav_length]

    def __getitem__(self, idx: int):
        """Return a clean crop, or (clean, noisy) if augmentation is enabled.

        Backward compatible: trainers that never opted into noise-robust mode
        still get a single tensor.
        """
        # Try a few times in case a file is corrupt/unreadable.
        last_err: Exception | None = None
        for _ in range(8):
            # Weighted choice between the main pool and the (optional) aux pool.
            if self.aux_files and self._rng.random() < self.aux_mix_ratio:
                path = self._rng.choice(self.aux_files)
            else:
                path = self._rng.choice(self.files)
            try:
                clean = self._load_random_crop(path)
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue
            if not self.augment_enabled:
                return clean
            # Apply Beatrice's augment_audio to a clone so the clean target
            # is untouched. Import lazily so this module stays importable in
            # environments without beatrice_trainer (e.g. CI on CPU).
            from distill_augment import apply_augmentation

            try:
                noisy = apply_augmentation(
                    clean.detach().clone(),
                    self.noise_files,
                    self.ir_files,
                    self.aug_kwargs,
                )
            except Exception as e:  # noqa: BLE001
                # Augmentation can fail for very short noise files; fall back
                # to identity so training never hangs on a bad sample.
                last_err = e
                noisy = clean.detach().clone()
            return clean, noisy
        raise RuntimeError(f"could not load any audio after 8 tries; last error: {last_err}")
