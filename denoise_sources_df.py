"""DeepFilterNet-based denoiser for raw TTS source recordings.

DeepFilterNet (DF) is a deep-learning denoiser that handles non-stationary
artifacts (buzz tied to phonemes, transient clicks, vocoder ringing) that
spectral-subtraction methods like `noisereduce` miss.

Trade-offs vs. noisereduce:
  - DF: stronger removal, better on speech-correlated artifacts, but may
    slightly soften voice character.
  - noisereduce: gentler, better for stationary hum, weaker on transients.

DF runs natively at 48 kHz. Inputs at other rates are upsampled, denoised,
then saved at 48 kHz; downstream preprocess.py resamples to TARGET_SR.

Usage:
    uv run python denoise_sources_df.py --dataset new_lol_data
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from df.enhance import enhance, init_df

INPUTS_ROOT = Path("inputs")


def _first_token(s: str) -> str:
    m = re.match(r"[a-zA-Z0-9_]+", s.strip())
    return m.group(0).lower() if m else ""


def find_speaker_sources(dataset_dir: Path) -> list[tuple[str, Path]]:
    """Mirror preprocess.discover() exactly: only top-level <speaker>/<name>.wav
    where <name> begins with <speaker> (skip raw/ and processed/ subdirs)."""
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
                out.append((speaker, path))
    return out


# Chunk size in seconds. Long files crash cuDNN GRU; 30s chunks keep
# memory + sequence length manageable while letting DF maintain context.
CHUNK_SEC = 30.0


def denoise_file(src: Path, dst: Path, model, df_state, target_sr: int) -> None:
    print(f"  reading {src.relative_to(INPUTS_ROOT)} ...")
    wav, sr = torchaudio.load(str(src), backend="soundfile")
    if wav.size(0) > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    wav = wav.contiguous()
    dur = wav.size(1) / target_sr
    print(f"    dur={dur:.1f}s sr_in={sr} -> denoising at {target_sr} Hz with DeepFilterNet (chunked {CHUNK_SEC}s) ...")
    peak_in = wav.abs().max().item()

    chunk_samples = int(CHUNK_SEC * target_sr)
    chunks = []
    n_chunks = (wav.size(1) + chunk_samples - 1) // chunk_samples
    for i in range(n_chunks):
        start = i * chunk_samples
        end = min(start + chunk_samples, wav.size(1))
        seg = wav[:, start:end].contiguous()
        enhanced_seg = enhance(model, df_state, seg)
        chunks.append(enhanced_seg)
    enhanced = torch.cat(chunks, dim=1)

    # Preserve original peak (DF can shift gain slightly)
    peak_out = enhanced.abs().max().item()
    if peak_out > 0 and peak_in > 0:
        enhanced = enhanced * (peak_in / peak_out) * 0.98

    y = enhanced.squeeze(0).cpu().numpy().astype(np.float32)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), y, target_sr, subtype="PCM_16")
    print(f"    wrote {dst}  ({n_chunks} chunks)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="dataset folder under inputs/")
    ap.add_argument("--suffix", default="_df", help="suffix for the output dataset folder")
    args = ap.parse_args()

    src_root = INPUTS_ROOT / args.dataset
    dst_root = INPUTS_ROOT / f"{args.dataset}{args.suffix}"

    if not src_root.is_dir():
        raise SystemExit(f"missing source dataset: {src_root}")

    sources = find_speaker_sources(src_root)
    if not sources:
        raise SystemExit(f"no top-level speaker wavs found under {src_root}")

    print("=== loading DeepFilterNet ...")
    model, df_state, _ = init_df()
    target_sr = df_state.sr()
    print(f"    model loaded, native sr = {target_sr} Hz")

    print(f"\n=== denoising {len(sources)} speaker files: {src_root} -> {dst_root} ===\n")
    with torch.inference_mode():
        for speaker, src in sources:
            rel = src.relative_to(src_root)
            dst = dst_root / rel
            print(f"[{speaker}] {rel}")
            denoise_file(src, dst, model, df_state, target_sr)
            print()

    print(f"done. now run:  uv run python preprocess.py --dataset {args.dataset}{args.suffix} --energy-threshold 35")


if __name__ == "__main__":
    main()
