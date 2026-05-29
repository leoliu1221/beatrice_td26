"""Held-out evaluation of a distilled PhoneExtractor checkpoint.

Computes cos_sim(student, teacher) on four conditions to detect overfitting,
loss of clean-input representation, and out-of-domain generalization:

    (a) Clean LibriSpeech       -- sanity check; should be >= 0.78
    (b) Augmented LibriSpeech   -- matches training distribution
    (c) Clean target domain     -- e.g. LoL TTS clips; out-of-domain clean
    (d) Augmented target domain -- the real inference distribution

If (a) drops below the pre-augmentation level (~0.78), the noise-robust
training has degraded the clean representation -- bad sign.
If (b) tracks `train/cos_sim` from TensorBoard, training is consistent.
If (c) and (d) are much lower than (a) and (b), the model overfit to the
LibriSpeech-specific acoustics and won't generalize to the target speakers.

This eval requires the `projection` head, which is only saved in training
checkpoints (NOT exported `.pt` files in `assets/pretrained/`).

Usage:
    uv run python -m phone_extractor_trainer.eval \\
        --ckpt outputs/phone_extractor_en/checkpoint_latest.pt
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from statistics import mean, stdev

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from beatrice_trainer.__main__ import PhoneExtractor  # noqa: E402
from phone_extractor_trainer.train import HubertTeacher  # noqa: E402
from phone_extractor_trainer.data import discover_audio_files  # noqa: E402
from distill_augment import apply_augmentation, discover_aux_files  # noqa: E402


def load_wavs_16k(
    files: list[Path],
    n_samples: int,
    wav_length: int,
    rng: random.Random,
) -> torch.Tensor:
    """Load `n_samples` random 4-s crops at 16 kHz mono."""
    out: list[torch.Tensor] = []
    files = list(files)
    rng.shuffle(files)
    for path in files:
        if len(out) >= n_samples:
            break
        try:
            wav, sr = torchaudio.load(str(path), backend="soundfile")
        except Exception:
            continue
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        wav = wav.squeeze(0)
        if wav.size(0) < wav_length:
            wav = F.pad(wav, (0, wav_length - wav.size(0)), mode="constant")
        else:
            start = rng.randint(0, wav.size(0) - wav_length)
            wav = wav[start : start + wav_length]
        # Skip near-silent crops; they yield meaningless cos_sim.
        if wav.abs().max() < 1e-3:
            continue
        out.append(wav)
    if len(out) < n_samples:
        raise RuntimeError(
            f"only loaded {len(out)} usable crops from {len(files)} files; "
            f"need {n_samples}"
        )
    return torch.stack(out)


@torch.inference_mode()
def cos_sim_student_teacher(
    student: nn.Module,
    teacher: nn.Module,
    projection: nn.Module,
    wavs: torch.Tensor,
    device: torch.device,
    batch_size: int = 16,
) -> tuple[float, float]:
    """Mean cos_sim averaged over time-frames, plus per-sample std."""
    per_sample_means: list[float] = []
    for i in range(0, wavs.size(0), batch_size):
        batch = wavs[i : i + batch_size].to(device)
        teacher_feat = teacher(batch.float())                       # [B, T_t, 768]
        student_feat = student(batch.unsqueeze(1), return_stats=False)  # [B, 128, T_s]
        student_feat = student_feat.transpose(1, 2)                 # [B, T_s, 128]
        student_proj = projection(student_feat)                     # [B, T_s, 768]
        if teacher_feat.size(1) != student_proj.size(1):
            teacher_feat = F.interpolate(
                teacher_feat.transpose(1, 2),
                size=student_proj.size(1),
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        cos = F.cosine_similarity(student_proj.float(), teacher_feat.float(), dim=-1)
        # one mean per sample, time-averaged
        per_sample_means.extend(cos.mean(dim=-1).tolist())
    if len(per_sample_means) <= 1:
        return per_sample_means[0] if per_sample_means else float("nan"), 0.0
    return mean(per_sample_means), stdev(per_sample_means)


def build_eval_context(
    librispeech_dir: str,
    target_dir: str,
    noise_dir: str,
    ir_dir: str,
    n_samples: int,
    wav_length: int,
    device: torch.device,
    seed: int,
) -> dict:
    """Build a re-usable eval context: teacher + audio pools + cached eval crops.

    Caching the crops means the same audio is used across all checkpoints in a
    sweep, giving an apples-to-apples comparison (no sample-pool variance).
    """
    rng = random.Random(seed)
    teacher = HubertTeacher().to(device).eval()
    librispeech_files = discover_audio_files(Path(librispeech_dir))
    target_files = discover_audio_files(Path(target_dir))
    noise_files = discover_aux_files(noise_dir)
    ir_files = discover_aux_files(ir_dir)
    ls_clean = load_wavs_16k(librispeech_files, n_samples, wav_length, rng)
    tgt_clean = load_wavs_16k(target_files, n_samples, wav_length, rng)
    ls_noisy = torch.stack([apply_augmentation(w, noise_files, ir_files) for w in ls_clean])
    tgt_noisy = torch.stack([apply_augmentation(w, noise_files, ir_files) for w in tgt_clean])
    return dict(
        teacher=teacher,
        device=device,
        ls_clean=ls_clean,
        ls_noisy=ls_noisy,
        tgt_clean=tgt_clean,
        tgt_noisy=tgt_noisy,
        librispeech_files=librispeech_files,
        target_files=target_files,
        noise_files=noise_files,
        ir_files=ir_files,
    )


def eval_checkpoint(
    ckpt_path: str | Path,
    ctx: dict,
) -> dict:
    """Run the 4-condition eval on one checkpoint, returning a dict of metrics.

    Returns: {step, a_mean, a_std, b_mean, b_std, c_mean, c_std, d_mean, d_std}
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if "projection" not in ckpt:
        raise ValueError(f"{ckpt_path}: missing 'projection'; not a training ckpt")
    device = ctx["device"]
    student = PhoneExtractor().to(device).eval()
    student.load_state_dict(ckpt["phone_extractor"])
    projection = nn.Linear(128, ctx["teacher"].feature_dim).to(device).eval()
    projection.load_state_dict(ckpt["projection"])
    teacher = ctx["teacher"]

    a_m, a_s = cos_sim_student_teacher(student, teacher, projection, ctx["ls_clean"], device)
    b_m, b_s = cos_sim_student_teacher(student, teacher, projection, ctx["ls_noisy"], device)
    c_m, c_s = cos_sim_student_teacher(student, teacher, projection, ctx["tgt_clean"], device)
    d_m, d_s = cos_sim_student_teacher(student, teacher, projection, ctx["tgt_noisy"], device)
    return {
        "step": ckpt.get("step", -1),
        "a_mean": a_m, "a_std": a_s,
        "b_mean": b_m, "b_std": b_s,
        "c_mean": c_m, "c_std": c_s,
        "d_mean": d_m, "d_std": d_s,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt", required=True, type=str,
                   help="training checkpoint (must contain 'projection')")
    p.add_argument("--librispeech-dir", type=str,
                   default="datasets/librispeech/LibriSpeech/train-clean-100",
                   help="in-distribution clean audio pool")
    p.add_argument("--target-dir", type=str,
                   default="inputs/new_lol_data",
                   help="held-out target-domain audio (LoL TTS)")
    p.add_argument("--noise-dir", type=str, default="assets/noise")
    p.add_argument("--ir-dir", type=str, default="assets/ir")
    p.add_argument("--n-samples", type=int, default=64,
                   help="crops per condition (default 64 -> ~30 s on RTX)")
    p.add_argument("--wav-length-sec", type=float, default=4.0)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device(args.device)
    wav_length = (int(round(args.wav_length_sec * 16000)) // 160) * 160

    print(f"loading {args.ckpt} ...")
    print("scanning audio pools and pre-augmenting eval crops ...")
    ctx = build_eval_context(
        args.librispeech_dir, args.target_dir, args.noise_dir, args.ir_dir,
        args.n_samples, wav_length, device, args.seed,
    )
    print(f"  librispeech: {len(ctx['librispeech_files'])}  target: {len(ctx['target_files'])}")
    print(f"  noise: {len(ctx['noise_files'])}  ir: {len(ctx['ir_files'])}")

    r = eval_checkpoint(args.ckpt, ctx)
    print(f"\n=== held-out eval (n_samples={args.n_samples}, step={r['step']}) ===\n")
    rows = [
        ("(a) Clean LibriSpeech (sanity)", r["a_mean"], r["a_std"], ">= 0.78"),
        ("(b) Aug LibriSpeech (~train)",   r["b_mean"], r["b_std"], "~ train/cos_sim"),
        ("(c) Clean target-domain",        r["c_mean"], r["c_std"], ">= 0.65 ideal"),
        ("(d) Aug target-domain (real)",   r["d_mean"], r["d_std"], ">= 0.60 ideal"),
    ]
    print(f"{'Condition':<35} {'cos_sim':>9} {'± std':>8}   expected")
    print("-" * 76)
    for name, m, s, exp in rows:
        print(f"{name:<35} {m:>9.4f} {s:>8.4f}   {exp}")
    print()
    print("Heuristics:")
    print("  * (a) << 0.78 -> noise-robust training degraded clean representation")
    print("  * (b) much lower than train/cos_sim -> eval pool differs from train")
    print("  * (c) << (a) by > 0.05 -> overfit to LibriSpeech acoustics")
    print("  * (d) << (b) by > 0.05 -> overfit to specific noise patterns")


if __name__ == "__main__":
    main()
