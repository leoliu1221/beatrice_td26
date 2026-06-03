"""Convert ENGLISH source clips through a trained Beatrice checkpoint.

Why this exists: the trainer's built-in eval set (assets/test) is JAPANESE
Common Voice, so it cannot reveal English-pronunciation quality (the phone
extractor reads content from the *source*). This harness converts English
sources (from analysis/data/ls_aligned.pkl, which also carries reference MFA
phonemes) to one or more target speakers, so the downstream scorer can measure
PER / WER / UTMOS / speaker-similarity on English content.

Output: <out>/source/<id>.wav (16k) + <out>/<target>/<id>.wav (24k converted)
        + <out>/manifest.json  (per-id ref phonemes, target list)

Usage:
    uv run python analysis/convert_eval.py \
        --ckpt outputs/path_a_en_clean/checkpoint_latest.pt.gz \
        --targets sion,noxus_male --n 40 --tag path_a_5k
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from beatrice_trainer.__main__ import (  # noqa: E402
    ConverterNetwork,
    PhoneExtractor,
    PitchEstimator,
)

DATA = REPO / "analysis/data/ls_aligned.pkl"
# speaker dirs in sorted order define the speaker-id mapping used at training.
SPEAKER_DIRS = REPO / "preprocessed/new_lol_data_df"


def speaker_names() -> list[str]:
    return sorted(p.name for p in SPEAKER_DIRS.iterdir() if p.is_dir())


def load_converter(ckpt_path: Path, device) -> tuple[ConverterNetwork, dict]:
    with gzip.open(ckpt_path, "rb") as f:
        ck = torch.load(f, map_location="cpu", weights_only=False)
    h = ck["h"]
    n_speakers = ck["net_g"]["embed_speaker.weight"].shape[0]
    phone = PhoneExtractor()
    pitch = PitchEstimator()
    phone.load_state_dict(ck["phone_extractor"])
    pitch.load_state_dict(ck["pitch_estimator"])
    net_g = ConverterNetwork(
        phone, pitch, n_speakers,
        h["pitch_bins"], h["hidden_channels"], h["vq_topk"],
        h["training_time_vq"], h["phone_noise_ratio"],
        augment_pitch=False, floor_noise_level=h["floor_noise_level"],
    )
    missing, unexpected = net_g.load_state_dict(ck["net_g"], strict=False)
    print(f"load net_g: missing={len(missing)} unexpected={len(unexpected)} | iter={ck.get('iteration')} n_speakers={n_speakers}")
    net_g.to(device).eval()
    # frozen_modules is a plain dict (kept out of state_dict), so net_g.to()
    # does NOT move the phone/pitch models -- move them explicitly.
    phone.to(device).eval()
    pitch.to(device).eval()
    net_g.enable_hook()  # activate VQ codebook hook on phone_extractor.head
    return net_g, h


@torch.inference_mode()
def convert(net_g, wav16k: np.ndarray, spk_id: int, device) -> np.ndarray:
    x = torch.from_numpy(wav16k).float().to(device)
    n = (x.numel() // 160) * 160
    x = x[:n]
    y = net_g(
        x[None, None],                                   # [1,1,T]
        torch.tensor([spk_id], device=device),
        torch.tensor([0.0], device=device),             # formant shift
        torch.tensor([0.0], device=device),             # pitch shift
    ).squeeze(1)[0]                                       # [T*240/160] @24k
    return y.float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--targets", default="sion", help="comma-separated speaker names")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--tag", required=True, help="output subdir name")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = speaker_names()
    name_to_id = {n: i for i, n in enumerate(names)}
    targets = [t.strip() for t in args.targets.split(",")]
    for t in targets:
        if t not in name_to_id:
            raise SystemExit(f"unknown target '{t}'. choices: {names}")

    utts = pickle.load(open(DATA, "rb"))[: args.n]
    net_g, h = load_converter(Path(args.ckpt), device)

    out = REPO / "analysis/converted" / args.tag
    (out / "source").mkdir(parents=True, exist_ok=True)
    for t in targets:
        (out / t).mkdir(parents=True, exist_ok=True)

    manifest = {"ckpt": str(args.ckpt), "targets": targets, "speaker_names": names, "items": []}
    for i, u in enumerate(utts):
        cid = f"{i:03d}"
        wav = u["wav"].astype(np.float32)
        sf.write(out / "source" / f"{cid}.wav", wav, 16000)
        for t in targets:
            y = convert(net_g, wav, name_to_id[t], device)
            sf.write(out / t / f"{cid}.wav", y, 24000)
        manifest["items"].append({
            "id": cid, "src_id": u["id"],
            "phonemes": [[p, s, e] for (p, s, e) in u["phonemes"]],
        })
        if (i + 1) % 10 == 0:
            print(f"  converted {i+1}/{len(utts)}")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {len(utts)} source + {len(utts)*len(targets)} converted clips -> {out}")


if __name__ == "__main__":
    main()
