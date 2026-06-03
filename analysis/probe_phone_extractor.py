"""Phonetic-contrast probe for PhoneExtractor variants.

Goal: test whether the v2 noise-robust phone extractor collapsed phonetic
distinctions ("big tongue") relative to the v1 clean extractor.

A healthy phone extractor produces features that *change* as the phoneme
changes (high temporal contrast, block structure in the self-similarity
matrix, high effective rank). A collapsed/over-smoothed extractor produces
nearly-constant features regardless of phoneme — every frame looks alike, so
the converter cannot tell /s/ from /a/, yielding mumbled output.

We feed the SAME clean clips (what inference actually sees) through each
extractor via the exact Beatrice inference path (`units()`), then compute
contrast metrics. No retraining required.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from beatrice_trainer.__main__ import PhoneExtractor  # noqa: E402

VARIANTS = {
    "v1_clean_580k":      REPO / "assets/pretrained/phone_extractor/en_clean_580k.pt",
    "v2_nr_targetmix_300k": REPO / "assets/pretrained/phone_extractor/en_nr_targetmix02_300k.pt",
    "v1_clean_200k":      REPO / "assets/pretrained/phone_extractor/en_clean_200k.pt",
    "jp_122_upstream":    REPO / "assets/pretrained/phone_extractor/jp_122_3000k.pt",
    "en_v3_anchored_80k": REPO / "outputs/phone_extractor_en_v3/checkpoint_00080000.pt",
    "en_v4_softvc_40k":   REPO / "outputs/phone_extractor_en_v4/_probe_snapshot.pt",
}

# Clean clips: a mix of target-domain (LoL) and natural English (LibriSpeech).
CLIPS = [
    REPO / "preprocessed/new_lol_data_df/sion/sion_0001.wav",
    REPO / "preprocessed/new_lol_data_df/sion/sion_0004.wav",
    REPO / "datasets/librispeech/LibriSpeech/train-clean-100/196/122152/196-122152-0020.flac",
    REPO / "datasets/librispeech/LibriSpeech/train-clean-100/196/122152/196-122152-0024.flac",
]


def load_extractor(path: Path, device: torch.device) -> PhoneExtractor:
    model = PhoneExtractor().to(device).eval().requires_grad_(False)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    sd = ckpt["phone_extractor"] if "phone_extractor" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  WARN {path.name}: missing={len(missing)} unexpected={len(unexpected)}")
    return model


def load_clip_16k(path: Path, device: torch.device, max_sec: float = 6.0) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path), backend="soundfile")
    if wav.size(0) > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav[:, : int(max_sec * 16000)]
    n = (wav.size(1) // 160) * 160
    return wav[:, :n].to(device)  # [1, T]


def contrast_metrics(feats: torch.Tensor) -> dict[str, float]:
    """feats: [T, C] phone features for one clip (already on cpu, float32)."""
    f = feats - feats.mean(0, keepdim=True)
    T, C = f.shape

    # 1. Temporal contrast: mean L2 of frame-to-frame deltas, normalized by
    #    per-frame norm. Low = over-smoothed.
    norms = feats.norm(dim=1).clamp_min(1e-6)
    deltas = (feats[1:] - feats[:-1]).norm(dim=1)
    temporal_contrast = float((deltas / norms[:-1]).mean())

    # 2. Mean pairwise cosine similarity between frames. High = collapsed
    #    (all frames look alike regardless of phoneme).
    fn = feats / feats.norm(dim=1, keepdim=True).clamp_min(1e-6)
    sim = fn @ fn.t()  # [T, T]
    off = sim[~torch.eye(T, dtype=torch.bool)]
    mean_pairwise_cos = float(off.mean())

    # 3. Effective rank (participation ratio) of the feature covariance.
    #    High = features spread across many dims (rich); low = collapsed.
    cov = (f.t() @ f) / max(T - 1, 1)
    eig = torch.linalg.eigvalsh(cov).clamp_min(0)
    pr = float((eig.sum() ** 2) / (eig.square().sum() + 1e-12))  # participation ratio

    # 4. Raw temporal variance (mean over channels of per-channel variance).
    temporal_var = float(f.var(dim=0).mean())

    return {
        "temporal_contrast": temporal_contrast,
        "mean_pairwise_cos": mean_pairwise_cos,
        "effective_rank": pr,
        "temporal_var": temporal_var,
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")

    clips = []
    for p in CLIPS:
        if p.is_file():
            clips.append((p.parent.parent.name + "/" + p.name, load_clip_16k(p, device)))
        else:
            print(f"  (skip missing clip {p})")
    print(f"loaded {len(clips)} clips\n")

    agg: dict[str, dict[str, list[float]]] = {}
    for name, path in VARIANTS.items():
        if not path.is_file():
            print(f"(skip missing variant {name}: {path})")
            continue
        model = load_extractor(path, device)
        per_metric: dict[str, list[float]] = {}
        for clip_name, wav in clips:
            feats = model.units(wav.unsqueeze(0)).squeeze(0).float().cpu()  # [T, C]
            m = contrast_metrics(feats)
            for k, v in m.items():
                per_metric.setdefault(k, []).append(v)
        agg[name] = per_metric
        del model

    # Report
    metrics = ["temporal_contrast", "mean_pairwise_cos", "effective_rank", "temporal_var"]
    hint = {
        "temporal_contrast": "higher=sharper articulation",
        "mean_pairwise_cos": "LOWER=sharper (high=collapsed/big-tongue)",
        "effective_rank":    "higher=richer representation",
        "temporal_var":      "higher=more phonetic movement",
    }
    print(f"{'variant':<24}" + "".join(f"{m:>20}" for m in metrics))
    print("-" * (24 + 20 * len(metrics)))
    for name, per_metric in agg.items():
        row = f"{name:<24}"
        for m in metrics:
            row += f"{np.mean(per_metric[m]):>20.4f}"
        print(row)
    print()
    for m in metrics:
        print(f"  {m}: {hint[m]}")


if __name__ == "__main__":
    main()
