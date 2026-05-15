"""Pre-extract F0 for all audio files to speed up training.

This script extracts F0 using pyworld DIO and saves it alongside the audio.
During training, the dataset can load pre-computed F0 instead of computing
it on-the-fly, dramatically improving GPU utilization.

Usage:
    uv run python -m pitch_estimator_trainer.preprocess \
        --data-dir /path/to/audio \
        --num-workers 24
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path
from functools import partial

import numpy as np
import torchaudio

from pitch_estimator_trainer.data import (
    discover_audio_files,
    extract_f0_pyworld,
    _get_resampler,
    AUDIO_EXTS,
)


def process_file(
    path: Path,
    sample_rate: int = 16000,
    hop_length: int = 160,
    f0_floor: float = 55.0,
    f0_ceil: float = 1400.0,
) -> tuple[Path, bool, str]:
    """Extract F0 for a single file and save as .f0.npy alongside it."""
    f0_path = path.with_suffix(path.suffix + ".f0.npy")
    
    # Skip if already processed
    if f0_path.exists():
        return path, True, "skipped (exists)"
    
    try:
        wav, sr = torchaudio.load(str(path), backend="soundfile")
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != sample_rate:
            wav = _get_resampler(sr, sample_rate)(wav)
        wav = wav.squeeze(0).numpy()
        
        f0 = extract_f0_pyworld(
            wav,
            sample_rate=sample_rate,
            hop_length=hop_length,
            f0_floor=f0_floor,
            f0_ceil=f0_ceil,
        )
        
        np.save(f0_path, f0)
        return path, True, "ok"
    except Exception as e:
        return path, False, str(e)


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-extract F0 for pitch estimator training")
    ap.add_argument("--data-dir", required=True, type=Path, help="root dir of audio files")
    ap.add_argument("--num-workers", type=int, default=mp.cpu_count(), help="parallel workers")
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--hop-length", type=int, default=160)
    ap.add_argument("--f0-floor", type=float, default=55.0)
    ap.add_argument("--f0-ceil", type=float, default=1400.0)
    args = ap.parse_args()

    print(f"Scanning {args.data_dir} ...")
    files = discover_audio_files(args.data_dir)
    print(f"  Found {len(files)} audio files")

    process_fn = partial(
        process_file,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        f0_floor=args.f0_floor,
        f0_ceil=args.f0_ceil,
    )

    print(f"Extracting F0 with {args.num_workers} workers...")
    
    success = 0
    skipped = 0
    failed = 0
    
    with mp.Pool(args.num_workers) as pool:
        for i, (path, ok, msg) in enumerate(pool.imap_unordered(process_fn, files)):
            if ok:
                if "skipped" in msg:
                    skipped += 1
                else:
                    success += 1
            else:
                failed += 1
                print(f"  FAILED: {path.name}: {msg}")
            
            if (i + 1) % 1000 == 0:
                print(f"  Progress: {i + 1}/{len(files)} (success={success}, skipped={skipped}, failed={failed})")
    
    print(f"\nDone! success={success}, skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    main()
