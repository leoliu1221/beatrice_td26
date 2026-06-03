"""Fetch a small phone-aligned LibriSpeech subset for the ABX / phone probe.

Downloads ONE parquet shard of `gilkeyio/librispeech-alignments`
(train_clean_100) and reads only the first N rows locally. This is far faster
than HF row-by-row streaming (which fetches+decodes audio over HTTP per sample).
The shard is cached by huggingface_hub, so reruns are instant.

Output pickle:
    [{ "id": str, "wav": np.float32[T] @16k, "phonemes": [(label, start_s, end_s), ...] }, ...]

Run standalone (no beatrice imports) so the repo's local `datasets/` folder
does not shadow anything; we also strip cwd / '' from sys.path defensively.
"""
from __future__ import annotations

import argparse
import io
import os
import pickle
import sys

sys.path = [p for p in sys.path if p not in ("", os.getcwd(), os.path.dirname(os.path.abspath(__file__)))]

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DEFAULT_OUT = os.path.join(DATA_DIR, "ls_aligned.pkl")
REPO_ID = "gilkeyio/librispeech-alignments"
SHARD = "data/train_clean_100-00000-of-00014.parquet"
MIN_SEC, MAX_SEC = 2.0, 10.0
MIN_PHONEMES = 5


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch a phone-aligned LibriSpeech subset.")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output pickle path")
    ap.add_argument("--n-utt", type=int, default=200, help="number of utterances to collect")
    ap.add_argument("--skip", type=int, default=0,
                    help="skip this many PASSING utterances first (use to build a "
                         "training split disjoint from the default 200-utt eval set)")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"downloading shard {SHARD} (cached after first run) ...")
    path = hf_hub_download(REPO_ID, SHARD, repo_type="dataset")
    print(f"  -> {path}\nreading rows (skip={args.skip}, n_utt={args.n_utt}) ...")

    pf = pq.ParquetFile(path)
    out = []
    seen = 0  # count of passing utterances (before skip)
    for batch in pf.iter_batches(batch_size=64, columns=["id", "audio", "phonemes"]):
        d = batch.to_pylist()
        for ex in d:
            ph = ex.get("phonemes") or []
            if len(ph) < MIN_PHONEMES:
                continue
            audio = ex["audio"]
            wav, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(1)
            if sr != 16000:
                continue
            dur = len(wav) / sr
            if not (MIN_SEC <= dur <= MAX_SEC):
                continue
            seen += 1
            if seen <= args.skip:
                continue
            phonemes = [(p["phoneme"], float(p["start"]), float(p["end"])) for p in ph]
            out.append({"id": ex.get("id", str(len(out))), "wav": wav, "phonemes": phonemes})
            if len(out) % 50 == 0:
                print(f"  collected {len(out)}/{args.n_utt}")
            if len(out) >= args.n_utt:
                break
        if len(out) >= args.n_utt:
            break

    with open(args.out, "wb") as f:
        pickle.dump(out, f)
    n_ph = sum(len(d["phonemes"]) for d in out)
    labels = sorted({lab for d in out for lab, _, _ in d["phonemes"]})
    print(f"\nsaved {len(out)} utterances, {n_ph} phoneme segments -> {args.out}")
    print(f"distinct phoneme labels ({len(labels)}): {labels}")


if __name__ == "__main__":
    main()
