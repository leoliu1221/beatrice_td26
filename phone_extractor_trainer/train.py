"""HuBERT-distillation trainer for Beatrice's PhoneExtractor.

Architecture: imported as-is from `beatrice_trainer.__main__.PhoneExtractor` so
the trained checkpoint is bit-compatible with Beatrice's loader.

Training recipe (inferred from the published 122_checkpoint_03000000.pt; see
phone_extractor_trainer/README.md for the reasoning):
  - Teacher  : torchaudio.pipelines.HUBERT_BASE (English LibriSpeech-960h SSL),
               frozen, eval mode. Outputs 768-dim features at 50 fps from
               16 kHz audio.
  - Student  : PhoneExtractor() with default ctor (matches Beatrice exactly).
               Outputs 128-dim features at 100 fps from 16 kHz audio.
  - Loss     : 1 - cos_sim(proj(student), teacher) + lambda_mse * MSE
               where `proj` is a trainable linear (128 -> 768) used only
               during training; it is NOT exported.
               Teacher is linearly upsampled 2x along time to match student.
  - Optimizer: AdamW with cosine schedule + warmup.

Usage:
    uv run python -m phone_extractor_trainer.train \
        --data-dir /path/to/english_audio \
        --out-dir  outputs/phone_extractor_en \
        --steps 200000

To swap into Beatrice afterwards:
    uv run python -m phone_extractor_trainer.export \
        outputs/phone_extractor_en/checkpoint_latest.pt \
        assets/pretrained/phone_extractor_en.pt
    # then point `phone_extractor_file` in assets/default_config.json at it.
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
import torchaudio
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Make `beatrice_trainer.__main__` importable when running this file directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from beatrice_trainer.__main__ import PhoneExtractor  # noqa: E402
from phone_extractor_trainer.data import (  # noqa: E402
    WavCropDataset,
    discover_audio_files,
)


# ---------------------------------------------------------------------------
# Teacher
# ---------------------------------------------------------------------------

class HubertTeacher(nn.Module):
    """Wraps torchaudio's HUBERT_BASE for feature extraction.

    Returns features from a specific transformer layer (default: layer 9 of 12,
    which is consistently the most phonetically informative in published
    probing studies of HuBERT BASE).
    """

    def __init__(self, layer_index: int = 9):
        super().__init__()
        bundle = torchaudio.pipelines.HUBERT_BASE
        self.model = bundle.get_model()
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.layer_index = layer_index
        self.sample_rate = bundle.sample_rate  # 16000
        self.feature_dim = 768

    @torch.inference_mode()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: [B, T] at 16 kHz
        # extract_features returns (List[Tensor[B, T', 768]], lengths)
        features, _ = self.model.extract_features(wav)
        # Layer indexing: features[i] is the output of transformer layer i.
        idx = min(self.layer_index, len(features) - 1)
        return features[idx]  # [B, T_teacher, 768]


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

def distillation_loss(
    student_proj: torch.Tensor,  # [B, T_s, 768]
    teacher: torch.Tensor,       # [B, T_t, 768]
    lambda_mse: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # Upsample teacher in time to match student.
    if teacher.size(1) != student_proj.size(1):
        teacher = F.interpolate(
            teacher.transpose(1, 2),  # [B, 768, T_t]
            size=student_proj.size(1),
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)  # [B, T_s, 768]
    cos = F.cosine_similarity(student_proj, teacher, dim=-1)  # [B, T]
    cos_loss = (1.0 - cos).mean()
    mse_loss = F.mse_loss(student_proj, teacher)
    total = cos_loss + lambda_mse * mse_loss
    return total, {"cos": cos_loss.detach(), "mse": mse_loss.detach(), "cos_sim": cos.mean().detach()}


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
    # PhoneExtractor's FeatureExtractor requires wav_length % 160 == 0.
    wav_length = (wav_length // 160) * 160
    dataset = WavCropDataset(
        files=files,
        wav_length=wav_length,
        samples_per_epoch=args.batch_size * args.steps_per_epoch,
        sample_rate=16000,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )

    # ---- models
    student = PhoneExtractor().to(device)
    teacher = HubertTeacher(layer_index=args.teacher_layer).to(device)
    projection = nn.Linear(128, teacher.feature_dim).to(device)

    if args.init_from:
        ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
        sd = ckpt.get("phone_extractor", ckpt)
        missing, unexpected = student.load_state_dict(sd, strict=False)
        print(f"init_from {args.init_from}: missing={len(missing)} unexpected={len(unexpected)}")

    student.train()

    # ---- optimizer
    params = list(student.parameters()) + list(projection.parameters())
    optim = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.98), weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    # ---- resume
    start_step = 0
    ckpt_latest = out_dir / "checkpoint_latest.pt"
    if args.resume and ckpt_latest.is_file():
        print(f"resuming from {ckpt_latest}")
        ckpt = torch.load(ckpt_latest, map_location="cpu", weights_only=False)
        student.load_state_dict(ckpt["phone_extractor"])
        projection.load_state_dict(ckpt["projection"])
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
    pbar = tqdm(total=args.steps, initial=step, desc="distill", dynamic_ncols=True)
    while step < args.steps:
        for wav in loader:
            if step >= args.steps:
                break
            wav = wav.to(device, non_blocking=True)  # [B, wav_length]

            # update LR
            lr = cosine_warmup_lr(step, args.warmup_steps, args.steps, args.lr, args.min_lr)
            for g in optim.param_groups:
                g["lr"] = lr

            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                # Teacher in fp32 (HuBERT is small enough; keeps targets stable)
                with torch.amp.autocast("cuda", enabled=False):
                    teacher_feat = teacher(wav.float())  # [B, T_t, 768]
                # Student: [B, 1, wav_length] -> [B, 128, T_s]
                student_feat = student(wav.unsqueeze(1), return_stats=False)
                student_feat = student_feat.transpose(1, 2)  # [B, T_s, 128]
                student_proj = projection(student_feat)       # [B, T_s, 768]
                loss, stats = distillation_loss(
                    student_proj, teacher_feat.to(student_proj.dtype),
                    lambda_mse=args.lambda_mse,
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
                writer.add_scalar("train/cos_loss", stats["cos"].item(), step)
                writer.add_scalar("train/mse_loss", stats["mse"].item(), step)
                writer.add_scalar("train/cos_sim", stats["cos_sim"].item(), step)
                writer.add_scalar("train/lr", lr, step)
                writer.add_scalar("train/grad_norm", float(grad_norm), step)
                writer.add_scalar("train/it_per_s", step / max(elapsed, 1e-9), step)
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    cos=f"{stats['cos_sim'].item():.3f}",
                    lr=f"{lr:.2e}",
                )

            if step % args.save_interval == 0 or step == args.steps:
                ckpt = {
                    "step": step,
                    "phone_extractor": student.state_dict(),
                    "projection": projection.state_dict(),
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
    p = argparse.ArgumentParser(description="Train a PhoneExtractor by HuBERT distillation.")
    p.add_argument("--data-dir", required=True, type=str, help="root dir of audio files (recursive)")
    p.add_argument("--out-dir", required=True, type=str, help="output dir for checkpoints & TB logs")
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--warmup-steps", type=int, default=5_000)
    p.add_argument("--steps-per-epoch", type=int, default=2_000,
                   help="DataLoader epoch length; total samples = batch_size * steps_per_epoch")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--wav-length-sec", type=float, default=4.0)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--min-lr", type=float, default=5e-6)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--lambda-mse", type=float, default=0.1)
    p.add_argument("--teacher-layer", type=int, default=9,
                   help="HuBERT BASE transformer layer to distill from (0-11). 9 is best for phones.")
    p.add_argument("--amp", action="store_true", default=True, help="use mixed-precision (cuda)")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--init-from", type=str, default="",
                   help="optional .pt to warm-start the student PhoneExtractor")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--save-interval", type=int, default=2_000)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
