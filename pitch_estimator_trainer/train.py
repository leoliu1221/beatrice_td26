"""Supervised trainer for Beatrice's PitchEstimator.

Architecture: imported as-is from `beatrice_trainer.__main__.PitchEstimator` so
the trained checkpoint is bit-compatible with Beatrice's loader.

Training recipe:
  - Student  : PitchEstimator() with default ctor (matches Beatrice exactly).
               Outputs 448-dim logits at 100 fps from 16 kHz audio.
  - Teacher  : pyworld DIO + StoneMask (robust F0 estimation algorithm).
               Provides ground-truth pitch bins for supervised training.
  - Loss     : Cross-entropy over 448 pitch bins (bin 0 = unvoiced).
  - Optimizer: AdamW with cosine schedule + warmup.

The key insight: the shipped pitch estimator was trained on data that may not
cover the full pitch range needed for cross-gender voice conversion. Retraining
on diverse pitch data (male/female speech + singing) improves coverage.

Usage:
    uv run python -m pitch_estimator_trainer.train \
        --data-dir /path/to/diverse_pitch_audio \
        --out-dir  outputs/pitch_estimator_v2 \
        --steps 300000

To swap into Beatrice afterwards (see assets/pretrained/README.md for naming):
    uv run python -m pitch_estimator_trainer.export \
        outputs/pitch_estimator_v2/checkpoint_00300000.pt \
        assets/pretrained/pitch_estimator/<dataset>_<tag>_<steps>.pt
    # then retarget the current symlink:
    #   ln -sfn <dataset>_<tag>_<steps>.pt assets/pretrained/pitch_estimator/current.pt
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Make `beatrice_trainer.__main__` importable when running this file directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from beatrice_trainer.__main__ import PitchEstimator  # noqa: E402
from pitch_estimator_trainer.data import (  # noqa: E402
    PitchDataset,
    discover_audio_files,
)


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

def cosine_warmup_lr(step: int, warmup: int, total: int, base_lr: float, min_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def pitch_estimation_loss(
    logits: torch.Tensor,      # [B, 448, T]
    targets: torch.Tensor,     # [B, T] Long
    label_smoothing: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Cross-entropy loss for pitch bin classification.
    
    Returns total loss and stats dict with:
    - ce: raw cross-entropy loss
    - acc: overall accuracy
    - voiced_acc: accuracy on voiced frames only
    - unvoiced_acc: accuracy on unvoiced frames only
    """
    batch_size, num_bins, length = logits.size()
    
    # Reshape for cross_entropy: [B*T, 448] vs [B*T]
    logits_flat = logits.transpose(1, 2).contiguous().view(-1, num_bins)  # [B*T, 448]
    targets_flat = targets.view(-1)  # [B*T]
    
    ce_loss = F.cross_entropy(logits_flat, targets_flat, label_smoothing=label_smoothing)
    
    # Compute accuracies
    with torch.no_grad():
        preds = logits_flat.argmax(dim=1)
        correct = (preds == targets_flat).float()
        acc = correct.mean()
        
        voiced_mask = targets_flat > 0
        unvoiced_mask = targets_flat == 0
        
        voiced_acc = correct[voiced_mask].mean() if voiced_mask.any() else torch.tensor(0.0)
        unvoiced_acc = correct[unvoiced_mask].mean() if unvoiced_mask.any() else torch.tensor(0.0)
        
        # Pitch error in semitones (for voiced frames only)
        if voiced_mask.any():
            pred_voiced = preds[voiced_mask].float()
            target_voiced = targets_flat[voiced_mask].float()
            # Each bin is 1/96 octave = 1/8 semitone
            pitch_error_bins = (pred_voiced - target_voiced).abs()
            pitch_error_semitones = pitch_error_bins / 8.0  # 96 bins/octave = 8 bins/semitone
            mean_pitch_error = pitch_error_semitones.mean()
        else:
            mean_pitch_error = torch.tensor(0.0)
    
    return ce_loss, {
        "ce": ce_loss.detach(),
        "acc": acc,
        "voiced_acc": voiced_acc,
        "unvoiced_acc": unvoiced_acc,
        "pitch_error_st": mean_pitch_error,
    }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"device: {device}")

    # ---- data
    print(f"scanning {args.data_dir} ...")
    files = discover_audio_files(Path(args.data_dir))
    print(f"  found {len(files)} audio files")
    wav_length = int(round(args.wav_length_sec * 16000))
    # PitchEstimator's feature extraction requires wav_length % 160 == 0.
    wav_length = (wav_length // 160) * 160
    # Optional noise-robust mode: when --augment is on, the student sees
    # noisy audio while the F0 label is still computed from clean audio.
    # This matches Beatrice's main trainer which feeds augmented audio to
    # the pitch estimator at training time.
    noise_files = None
    ir_files = None
    if args.augment:
        from distill_augment import discover_aux_files
        noise_files = discover_aux_files(args.noise_dir)
        ir_files = discover_aux_files(args.ir_dir)
        print(f"noise-robust mode: {len(noise_files)} noise files, {len(ir_files)} IR files")
    dataset = PitchDataset(
        files=files,
        wav_length=wav_length,
        samples_per_epoch=args.batch_size * args.steps_per_epoch,
        sample_rate=16000,
        hop_length=160,
        f0_floor=55.0,
        f0_ceil=1400.0,
        pitch_bins_per_octave=96,
        seed=args.seed,
        noise_files=noise_files,
        ir_files=ir_files,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )

    # ---- model
    student = PitchEstimator().to(device)

    if args.init_from:
        ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
        sd = ckpt.get("pitch_estimator", ckpt)
        missing, unexpected = student.load_state_dict(sd, strict=False)
        print(f"init_from {args.init_from}: missing={len(missing)} unexpected={len(unexpected)}")

    student.train()

    # ---- optimizer
    params = list(student.parameters())
    optim = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.98), weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    # ---- resume
    start_step = 0
    ckpt_latest = out_dir / "checkpoint_latest.pt"
    if args.resume and ckpt_latest.is_file():
        print(f"resuming from {ckpt_latest}")
        ckpt = torch.load(ckpt_latest, map_location="cpu", weights_only=False)
        student.load_state_dict(ckpt["pitch_estimator"])
        optim.load_state_dict(ckpt["optim"])
        if "scaler" in ckpt and scaler.is_enabled():
            scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt["step"]
        print(f"  resumed at step {start_step}")

    # ---- logging
    writer = SummaryWriter(log_dir=str(out_dir))
    print(f"tensorboard logs -> {out_dir}")

    # ---- loop
    step = start_step
    t_start = time.time()
    pbar = tqdm(total=args.steps, initial=step, desc="pitch_train", dynamic_ncols=True)
    while step < args.steps:
        for wav, pitch_bins in loader:
            if step >= args.steps:
                break
            wav = wav.to(device, non_blocking=True)  # [B, wav_length]
            pitch_bins = pitch_bins.to(device, non_blocking=True)  # [B, T]

            # update LR
            lr = cosine_warmup_lr(step, args.warmup_steps, args.steps, args.lr, args.min_lr)
            for g in optim.param_groups:
                g["lr"] = lr

            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                # Student: [B, 1, wav_length] -> ([B, 448, T], [B, 1, T])
                logits, energy = student(wav.unsqueeze(1))
                loss, stats = pitch_estimation_loss(
                    logits, pitch_bins,
                    label_smoothing=args.label_smoothing,
                )

            optim.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                grad_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                optim.step()

            step += 1
            pbar.update(1)

            if step % args.log_interval == 0:
                elapsed = time.time() - t_start
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/ce_loss", stats["ce"].item(), step)
                writer.add_scalar("train/acc", stats["acc"].item(), step)
                writer.add_scalar("train/voiced_acc", stats["voiced_acc"].item(), step)
                writer.add_scalar("train/unvoiced_acc", stats["unvoiced_acc"].item(), step)
                writer.add_scalar("train/pitch_error_st", stats["pitch_error_st"].item(), step)
                writer.add_scalar("train/lr", lr, step)
                writer.add_scalar("train/grad_norm", float(grad_norm), step)
                writer.add_scalar("train/it_per_s", step / max(elapsed, 1e-9), step)
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    acc=f"{stats['acc'].item():.3f}",
                    err=f"{stats['pitch_error_st'].item():.2f}st",
                    lr=f"{lr:.2e}",
                )

            if step % args.save_interval == 0 or step == args.steps:
                ckpt = {
                    "step": step,
                    "pitch_estimator": student.state_dict(),
                    "optim": optim.state_dict(),
                    "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                    "args": vars(args),
                }
                tmp = out_dir / "checkpoint_latest.pt.tmp"
                torch.save(ckpt, tmp)
                os.replace(tmp, ckpt_latest)
                # also keep step-numbered copies at major milestones
                if step % (args.save_interval * 5) == 0 or step == args.steps:
                    torch.save(ckpt, out_dir / f"checkpoint_{step:08d}.pt")
                pbar.write(f"[step {step}] saved checkpoint")

    pbar.close()
    writer.close()
    print("done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a PitchEstimator with pyworld supervision.")
    p.add_argument("--data-dir", required=True, type=str, help="root dir of audio files (recursive)")
    p.add_argument("--out-dir", required=True, type=str, help="output dir for checkpoints & TB logs")
    p.add_argument("--steps", type=int, default=300_000)
    p.add_argument("--warmup-steps", type=int, default=5_000)
    p.add_argument("--steps-per-epoch", type=int, default=2_000,
                   help="DataLoader epoch length; total samples = batch_size * steps_per_epoch")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--wav-length-sec", type=float, default=4.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.1,
                   help="Label smoothing for cross-entropy loss")
    p.add_argument("--amp", action="store_true", default=True, help="use mixed-precision (cuda)")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--init-from", type=str, default="",
                   help="optional .pt to warm-start the student PitchEstimator")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--save-interval", type=int, default=2_000)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--augment", action="store_true",
        help="enable noise-robust training: student sees augmented audio, "
             "F0 label still computed from clean audio. Closes the train/test "
             "gap with Beatrice's main trainer.",
    )
    p.add_argument("--noise-dir", type=str, default="assets/noise",
                   help="dir of background noise files (used with --augment)")
    p.add_argument("--ir-dir", type=str, default="assets/ir",
                   help="dir of impulse-response files for reverb (used with --augment)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
