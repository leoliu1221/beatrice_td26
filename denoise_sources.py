"""Denoise raw TTS source recordings before Beatrice preprocessing.

Uses spectral gating (noisereduce) which is well-suited to remove the
stationary buzz / aliasing artifacts common in TTS outputs, especially
on low-pitch / male voices.

Strategy: for each top-level speaker concatenated *.wav under inputs/<dataset>/<speaker>/,
write a denoised copy in place at inputs/<dataset>_denoised/<speaker>/<original_name>.
The directory structure mirrors the source so preprocess.py works unchanged.

Only the top-level <speaker>/<name>.wav files are processed (the ones
preprocess.py actually consumes); raw/ and processed/ subdirs are skipped.

Usage:
    uv run python denoise_sources.py --dataset new_lol_data
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import noisereduce as nr
import numpy as np
import soundfile as sf
import torchaudio

INPUTS_ROOT = Path("inputs")

# Strength of denoising. 1.0 = aggressive, 0.5-0.7 = moderate (preserves voice
# character better). TTS buzz is stationary, so moderate values clean well
# without dropping speech energy below auditok's VAD threshold downstream.
PROP_DECREASE = 0.6

# Stationary noise estimation works best for TTS artifacts (buzz that doesn't
# change much over time). non-stationary mode is for real-world room noise.
STATIONARY = True


def _first_token(s: str) -> str:
    m = re.match(r"[a-zA-Z0-9_]+", s.strip())
    return m.group(0).lower() if m else ""


def find_speaker_sources(dataset_dir: Path) -> list[tuple[str, Path]]:
    """Mirror preprocess.discover() exactly: return only files whose stem-prefix
    matches their parent folder name (the concatenated speaker recordings)."""
    out: list[tuple[str, Path]] = []
    for path in sorted(dataset_dir.rglob("*.wav")):
        rel = path.relative_to(dataset_dir)
        if len(rel.parts) == 1:
            speaker = _first_token(path.stem)
            out.append((speaker, path))
        else:
            speaker = _first_token(rel.parts[0])
            file_word = _first_token(path.stem)
            if file_word == speaker and len(rel.parts) == 2:
                # only consider speaker_dir/file.wav (skip raw/ and processed/)
                out.append((speaker, path))
    return out


def denoise_file(src: Path, dst: Path) -> None:
    print(f"  reading {src.relative_to(INPUTS_ROOT)} ...")
    wav, sr = torchaudio.load(str(src), backend="soundfile")
    if wav.size(0) > 1:
        wav = wav.mean(0, keepdim=True)
    y = wav.squeeze(0).numpy().astype(np.float32)
    dur = len(y) / sr
    print(f"    dur={dur:.1f}s sr={sr} -> denoising (stationary, prop={PROP_DECREASE}) ...")
    y_denoised = nr.reduce_noise(
        y=y,
        sr=sr,
        stationary=STATIONARY,
        prop_decrease=PROP_DECREASE,
    )
    # Preserve peak level
    peak_in = float(np.abs(y).max())
    peak_out = float(np.abs(y_denoised).max())
    if peak_out > 0 and peak_in > 0:
        y_denoised = y_denoised * (peak_in / peak_out) * 0.98
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), y_denoised, sr, subtype="PCM_16")
    print(f"    wrote {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="dataset folder under inputs/")
    ap.add_argument("--suffix", default="_denoised", help="suffix for the output dataset folder")
    args = ap.parse_args()

    src_root = INPUTS_ROOT / args.dataset
    dst_root = INPUTS_ROOT / f"{args.dataset}{args.suffix}"

    if not src_root.is_dir():
        raise SystemExit(f"missing source dataset: {src_root}")

    sources = find_speaker_sources(src_root)
    if not sources:
        raise SystemExit(f"no top-level speaker wavs found under {src_root}")

    print(f"=== denoising {len(sources)} speaker files: {src_root} -> {dst_root} ===\n")
    for speaker, src in sources:
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        print(f"[{speaker}] {rel}")
        denoise_file(src, dst)
        print()

    print(f"done. now run:  make DATASET={args.dataset}{args.suffix} preprocess")


if __name__ == "__main__":
    main()
