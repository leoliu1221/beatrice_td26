"""F0-accuracy probe for PitchEstimator variants.

Scores each estimator against the pyworld DIO+StoneMask F0 that the trainer
uses as ground truth, on the SAME clips, both clean and augmented. This tells
us whether reverting pitch to upstream jp_104 is safe (clean accuracy) and what
the noise-robust vctk_nr actually buys (augmented accuracy).

Metrics (on frames where both reference and prediction are voiced):
  - voicing_acc    : agreement of voiced/unvoiced decision (higher better)
  - voiced_err_st  : mean |pred-ref| in semitones (lower better)
  - gross_err_rate : fraction of co-voiced frames off by > 1 semitone, i.e.
                     octave/halving errors (lower better)
  - corr           : Pearson correlation of pred vs ref pitch (higher better)

Since the bin->Hz convention (f0_floor=55, 96 bins/octave) is defined by the
upstream architecture, a LOW error for jp_104 validates the convention; the
other variants are then directly comparable.
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

from beatrice_trainer.__main__ import PitchEstimator  # noqa: E402
from pitch_estimator_trainer.data import extract_f0_pyworld, f0_to_pitch_bin  # noqa: E402
from distill_augment import apply_augmentation, discover_aux_files  # noqa: E402

VARIANTS = {
    "jp_104_upstream": REPO / "assets/pretrained/pitch_estimator/jp_104_3_300k.pt",
    "vctk_clean_300k": REPO / "assets/pretrained/pitch_estimator/vctk_clean_300k.pt",
    "vctk_nr_300k":    REPO / "assets/pretrained/pitch_estimator/vctk_nr_300k.pt",
}

CLIPS = [
    REPO / "preprocessed/new_lol_data_df/sion/sion_0001.wav",
    REPO / "preprocessed/new_lol_data_df/sion/sion_0004.wav",
    REPO / "datasets/librispeech/LibriSpeech/train-clean-100/196/122152/196-122152-0020.flac",
    REPO / "datasets/librispeech/LibriSpeech/train-clean-100/196/122152/196-122152-0024.flac",
]

HOP = 160
F0_FLOOR = 55.0


def load_estimator(path: Path, device: torch.device) -> PitchEstimator:
    model = PitchEstimator().to(device).eval().requires_grad_(False)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    sd = ckpt["pitch_estimator"] if "pitch_estimator" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  WARN {path.name}: missing={len(missing)} unexpected={len(unexpected)}")
    return model


def load_clip_16k(path: Path, max_sec: float = 6.0) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path), backend="soundfile")
    if wav.size(0) > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav[:, : int(max_sec * 16000)]
    n = (wav.size(1) // HOP) * HOP
    return wav[:, :n]  # [1, T] cpu


@torch.inference_mode()
def predict_bins(model: PitchEstimator, wav_1t: torch.Tensor, device) -> np.ndarray:
    logits, _ = model(wav_1t.unsqueeze(0).to(device))  # [1, 448, T]
    bins = model.sample_pitch(logits)  # [1, T]
    return bins.squeeze(0).cpu().numpy()


def ref_bins(wav_1t: torch.Tensor) -> np.ndarray:
    f0 = extract_f0_pyworld(wav_1t.squeeze(0).numpy(), 16000, HOP, F0_FLOOR, 1400.0)
    return f0_to_pitch_bin(f0, F0_FLOOR, 96)


def score(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    n = min(len(pred), len(ref))
    pred, ref = pred[:n], ref[:n]
    pv, rv = pred > 0, ref > 0
    voicing_acc = float((pv == rv).mean())
    both = pv & rv
    if both.sum() < 2:
        return {"voicing_acc": voicing_acc, "voiced_err_st": float("nan"),
                "gross_err_rate": float("nan"), "corr": float("nan")}
    err_bins = np.abs(pred[both] - ref[both]).astype(np.float64)
    err_st = err_bins / 8.0
    gross = float((err_st > 1.0).mean())
    p, r = pred[both].astype(np.float64), ref[both].astype(np.float64)
    corr = float(np.corrcoef(p, r)[0, 1]) if p.std() > 0 and r.std() > 0 else float("nan")
    return {
        "voicing_acc": voicing_acc,
        "voiced_err_st": float(err_st.mean()),
        "gross_err_rate": gross,
        "corr": corr,
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")
    noise_files = discover_aux_files(REPO / "assets/noise")
    ir_files = discover_aux_files(REPO / "assets/ir")
    rng = np.random.default_rng(0)

    clips = []
    for p in CLIPS:
        if p.is_file():
            wav = load_clip_16k(p)
            # F0 label always from clean; build a noisy counterpart (no formant
            # shift, matching pitch training) for the robustness test.
            torch.manual_seed(0)
            noisy = apply_augmentation(wav.clone(), noise_files, ir_files,
                                       {"formant_shift_probability": 0.0}).unsqueeze(0)
            clips.append((p.name, wav, noisy, ref_bins(wav)))
        else:
            print(f"  (skip missing {p})")
    print(f"loaded {len(clips)} clips\n")

    for cond in ("clean", "noisy"):
        print(f"==================== {cond.upper()} input ====================")
        print(f"{'variant':<20}{'voicing_acc':>13}{'voiced_err_st':>15}{'gross_err':>11}{'corr':>8}")
        print("-" * 67)
        for name, path in VARIANTS.items():
            if not path.is_file():
                print(f"(skip missing {name})")
                continue
            model = load_estimator(path, device)
            agg: dict[str, list[float]] = {}
            for clip_name, wav, noisy, rb in clips:
                inp = wav if cond == "clean" else noisy
                s = score(predict_bins(model, inp, device), rb)
                for k, v in s.items():
                    if not np.isnan(v):
                        agg.setdefault(k, []).append(v)
            del model
            row = f"{name:<20}"
            for k in ("voicing_acc", "voiced_err_st", "gross_err_rate", "corr"):
                row += f"{np.mean(agg.get(k, [float('nan')])):>{13 if k=='voicing_acc' else (15 if k=='voiced_err_st' else (11 if k=='gross_err_rate' else 8))}.4f}"
            print(row)
        print()
    print("voicing_acc/corr: higher better.  voiced_err_st/gross_err: lower better.")


if __name__ == "__main__":
    main()
