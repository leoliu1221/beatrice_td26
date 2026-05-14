"""Split long single-speaker recordings into Beatrice-ready training clips.

Usage:
    uv run python preprocess.py
"""

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from auditok import split

SOURCES = [
    ("sion", Path("noxus_new/sion_new/Sion Edit for TD.wav")),
    ("teemo", Path("yordle_new/teemo_new/Teemo Edit for TD.wav")),
]
OUT_ROOT = Path("lol_data")
TARGET_SR = 24000  # Beatrice out_sample_rate
MIN_DUR = 4.0  # seconds; must be >= wav_length (4s)
MAX_DUR = 15.0
MAX_SILENCE = 0.3  # seconds of silence allowed inside a region
ENERGY_THRESHOLD = 45  # auditok energy threshold (lower = more permissive)


def process_one(speaker: str, src: Path):
    out_dir = OUT_ROOT / speaker
    out_dir.mkdir(parents=True, exist_ok=True)
    # wipe any previous run
    for p in out_dir.glob("*.wav"):
        p.unlink()

    print(f"\n[{speaker}] reading {src} ...")
    wav, sr = torchaudio.load(str(src), backend="soundfile")
    # mono
    if wav.size(0) > 1:
        wav = wav.mean(0, keepdim=True)
    # resample to 24k once
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    # peak-normalize lightly to avoid clipping after resample
    peak = wav.abs().max().item()
    if peak > 0:
        wav = wav * min(1.0, 0.95 / peak)

    pcm16 = (wav.squeeze(0).numpy() * 32767.0).clip(-32768, 32767).astype(np.int16)

    # auditok wants a file or bytes; feed bytes for speed
    regions = split(
        pcm16.tobytes(),
        sampling_rate=TARGET_SR,
        sample_width=2,
        channels=1,
        min_dur=MIN_DUR,
        max_dur=MAX_DUR,
        max_silence=MAX_SILENCE,
        energy_threshold=ENERGY_THRESHOLD,
    )

    n = 0
    total = 0.0
    for i, region in enumerate(regions):
        start_s = region.start
        end_s = region.end
        dur = end_s - start_s
        if dur < MIN_DUR:
            continue
        s0 = int(start_s * TARGET_SR)
        s1 = int(end_s * TARGET_SR)
        clip = pcm16[s0:s1]
        out_path = out_dir / f"{speaker}_{i:04d}.wav"
        sf.write(str(out_path), clip, TARGET_SR, subtype="PCM_16")
        n += 1
        total += dur

    print(f"[{speaker}] wrote {n} clips, {total:.1f}s total speech -> {out_dir}")


def main():
    for speaker, src in SOURCES:
        if not src.is_file():
            print(f"MISSING: {src}")
            continue
        process_one(speaker, src)


if __name__ == "__main__":
    main()
