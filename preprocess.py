"""Auto-discover long recordings under inputs/ and segment them into
Beatrice-ready training clips written to preprocessed/.

Layout conventions:
  inputs/<dataset>/<speaker>.<ext>          -> speaker = first word of stem, lowercased
  inputs/<dataset>/<speaker>/<anything>.<ext>  -> speaker = parent dir name, lowercased
  inputs/<dataset>/<speaker>/sub/.../*.<ext>   -> speaker = top-level subdir under dataset

Output:
  preprocessed/<dataset>/<speaker>/<speaker>_NNNN.wav (mono, 24 kHz, PCM_16)

Usage:
    uv run python preprocess.py                 # process every dataset under inputs/
    uv run python preprocess.py --dataset lol_data  # only one
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torchaudio
from auditok import split

INPUTS_ROOT = Path("inputs")
PREPROCESSED_ROOT = Path("preprocessed")
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}

TARGET_SR = 24000  # Beatrice out_sample_rate
MIN_DUR = 4.0  # seconds; must be >= wav_length (4s)
MAX_DUR = 15.0
MAX_SILENCE = 0.3
ENERGY_THRESHOLD = 45  # lower = more permissive


def _first_token(s: str) -> str:
    """First alphanumeric run of `s`, lowercased. e.g. 'Sion Edit' -> 'sion'."""
    m = re.match(r"[a-zA-Z0-9]+", s.strip())
    return m.group(0).lower() if m else ""


def discover(dataset_dir: Path):
    """Yield (speaker, source_file) tuples for one dataset directory.

    Layout rules:
      <dataset>/<file>.<ext>                  -> speaker = first word of filename
      <dataset>/<speaker>/<file>.<ext>        -> speaker = subfolder name; the file's
                                                 first word must match <speaker>,
                                                 otherwise the file is skipped.
    """
    for path in sorted(dataset_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTS:
            continue
        rel = path.relative_to(dataset_dir)
        if len(rel.parts) == 1:
            speaker = _first_token(path.stem)
        else:
            speaker = _first_token(rel.parts[0])
            file_word = _first_token(path.stem)
            if file_word != speaker:
                print(
                    f"  skip (filename '{path.name}' doesn't start with '{speaker}')"
                )
                continue
        if not speaker:
            continue
        yield speaker, path


def segment_file(src: Path, start_index: int, out_dir: Path, speaker: str) -> tuple[int, float]:
    print(f"  reading {src.name} ...")
    wav, sr = torchaudio.load(str(src), backend="soundfile")
    if wav.size(0) > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    peak = wav.abs().max().item()
    if peak > 0:
        wav = wav * min(1.0, 0.95 / peak)

    pcm16 = (wav.squeeze(0).numpy() * 32767.0).clip(-32768, 32767).astype(np.int16)

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
    for region in regions:
        dur = region.end - region.start
        if dur < MIN_DUR:
            continue
        s0 = int(region.start * TARGET_SR)
        s1 = int(region.end * TARGET_SR)
        out_path = out_dir / f"{speaker}_{start_index + n:04d}.wav"
        sf.write(str(out_path), pcm16[s0:s1], TARGET_SR, subtype="PCM_16")
        n += 1
        total += dur
    return n, total


def process_dataset(dataset_dir: Path):
    name = dataset_dir.name
    out_root = PREPROCESSED_ROOT / name
    print(f"\n=== dataset: {name} ===")

    # group source files by speaker so multiple files per speaker concatenate
    grouped: dict[str, list[Path]] = defaultdict(list)
    for speaker, src in discover(dataset_dir):
        grouped[speaker].append(src)

    if not grouped:
        print(f"  (no audio files found under {dataset_dir})")
        return

    for speaker, sources in grouped.items():
        out_dir = out_root / speaker
        out_dir.mkdir(parents=True, exist_ok=True)
        # wipe previous run for this speaker
        for p in out_dir.glob("*.wav"):
            p.unlink()

        print(f"\n[{name}/{speaker}] {len(sources)} source file(s)")
        total_clips = 0
        total_speech = 0.0
        for src in sources:
            n, secs = segment_file(src, total_clips, out_dir, speaker)
            total_clips += n
            total_speech += secs
        print(
            f"[{name}/{speaker}] wrote {total_clips} clips, "
            f"{total_speech:.1f}s total speech -> {out_dir}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="only process this dataset name (folder under inputs/)")
    ap.add_argument("--inputs-root", default=str(INPUTS_ROOT))
    args = ap.parse_args()

    inputs_root = Path(args.inputs_root)
    if not inputs_root.is_dir():
        raise SystemExit(f"inputs root not found: {inputs_root}")

    if args.dataset:
        candidates = [inputs_root / args.dataset]
    else:
        candidates = [p for p in sorted(inputs_root.iterdir()) if p.is_dir()]

    if not candidates:
        raise SystemExit(f"no datasets under {inputs_root}")

    for d in candidates:
        if not d.is_dir():
            print(f"skip (not a dir): {d}")
            continue
        process_dataset(d)


if __name__ == "__main__":
    main()
