"""PhoneExtractor v4: Soft-VC categorical objective (the REAL Beatrice recipe).

Finding (2026-06-03, see project_context.md lesson 2.10 / experimental_log.md):
the upstream jp_122 PhoneExtractor follows **Soft-VC**, whose content encoder is
trained by **predicting a distribution over discrete units via cross-entropy** -
a CLASSIFICATION objective - NOT the `cos + MSE` feature *regression* our v1/v2
(`train.py`) reimplemented. Regression collapses toward the conditional mean of
all sounds at a frame -> low effective_rank -> the muffled "big tongue". A
categorical target forces phoneme-discriminative features -> rich, clear words.
This is why jp_122 (categorical) is rich while en_clean (regression) is muffled.

v4 replicates Soft-VC faithfully, on ENGLISH:
  1. Cluster frozen HuBERT-BASE layer-9 features into K discrete units (k-means)
     over a sample of English audio. (Same teacher information as en_clean used,
     so the ONLY changed variable vs en_clean is regression -> classification.)
  2. Train the PhoneExtractor + a small linear head to PREDICT each frame's unit
     id by cross-entropy. The 128-dim PhoneExtractor output is the "soft unit";
     the head is training-only and dropped at export (like the old projection).
  3. Because HuBERT-L9 is English and separates R/L, TH, V, the unit labels do
     too -> the student should gain BOTH richness AND English correctness, with
     no anchor/contrastive hacks (those were crude stand-ins for this).

Usage:
    uv run python -m phone_extractor_trainer.train_v4 \
        --data-dir datasets/librispeech/LibriSpeech/train-clean-100 \
        --out-dir outputs/phone_extractor_en_v4 \
        --n-clusters 500 --steps 200000

Export + judge exactly like v1/v2 (checkpoint stores `phone_extractor`).
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

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from beatrice_trainer.__main__ import PhoneExtractor  # noqa: E402
from phone_extractor_trainer.data import WavCropDataset, discover_audio_files  # noqa: E402
from phone_extractor_trainer.train import HubertTeacher, cosine_warmup_lr  # noqa: E402

SR = 16000
SAMPLES_PER_FRAME = 160  # PhoneExtractor 100 fps; HuBERT 50 fps -> student = 2x teacher


# ---------------------------------------------------------------------------
# k-means over HuBERT-L9 features (the Soft-VC discrete units)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def build_kmeans(teacher, loader, n_frames, k, device, iters=25, seed=0):
    """Collect ~n_frames HuBERT-L9 frames and Lloyd-cluster into k centroids."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    feats = []
    got = 0
    print(f"k-means: collecting ~{n_frames} HuBERT-L9 frames ...")
    for batch in loader:
        wav = (batch[0] if isinstance(batch, (tuple, list)) else batch).to(device)
        f = teacher(wav.float()).reshape(-1, 768)              # [B*T, 768]
        feats.append(f.float().cpu())
        got += f.size(0)
        if got >= n_frames:
            break
    X = torch.cat(feats, 0)
    if X.size(0) > n_frames:
        idx = torch.randperm(X.size(0), generator=g)[:n_frames]
        X = X[idx]
    X = X.to(device)
    print(f"k-means: clustering {X.size(0)} frames into {k} units ({iters} iters) ...")
    cen = X[torch.randperm(X.size(0), device=device)[:k]].clone()
    for it in range(iters):
        # chunked nearest-centroid assignment to bound memory
        assign = torch.empty(X.size(0), dtype=torch.long, device=device)
        for s in range(0, X.size(0), 100_000):
            assign[s:s + 100_000] = torch.cdist(X[s:s + 100_000], cen).argmin(1)
        for c in range(k):
            m = assign == c
            if m.any():
                cen[c] = X[m].mean(0)
    return cen  # [k, 768]


@torch.inference_mode()
def assign_units(teacher_feat, centroids):
    """teacher_feat [B,T,768] -> unit ids [B,T] (nearest centroid)."""
    B, T, D = teacher_feat.shape
    f = teacher_feat.reshape(-1, D)
    ids = torch.cdist(f.float(), centroids).argmin(1)
    return ids.reshape(B, T)


def effective_rank(feats):
    f = feats.reshape(-1, feats.size(-1)).float()
    f = f - f.mean(0, keepdim=True)
    cov = (f.t() @ f) / max(f.size(0) - 1, 1)
    eig = torch.linalg.eigvalsh(cov).clamp_min(0)
    return float((eig.sum() ** 2) / (eig.square().sum() + 1e-12))


def _infinite(loader):
    while True:
        yield from loader


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

def train(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"device: {device}")

    wav_length = (int(round(args.wav_length_sec * SR)) // SAMPLES_PER_FRAME) * SAMPLES_PER_FRAME

    files = discover_audio_files(Path(args.data_dir))
    print(f"data: {len(files)} files")
    dataset = WavCropDataset(files=files, wav_length=wav_length,
                             samples_per_epoch=args.batch_size * args.steps_per_epoch,
                             sample_rate=SR, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda", drop_last=True,
                        persistent_workers=args.num_workers > 0)

    teacher = HubertTeacher(layer_index=args.teacher_layer).to(device).eval()

    # ---- discrete units (cache to out_dir)
    km_path = out_dir / f"kmeans_k{args.n_clusters}_l{args.teacher_layer}.pt"
    if km_path.is_file():
        centroids = torch.load(km_path, map_location=device)
        print(f"loaded k-means centroids {tuple(centroids.shape)} from {km_path}")
    else:
        km_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                               drop_last=True)
        centroids = build_kmeans(teacher, km_loader, args.kmeans_frames, args.n_clusters,
                                 device, seed=args.seed)
        torch.save(centroids, km_path)
        print(f"saved k-means centroids -> {km_path}")

    # ---- models
    student = PhoneExtractor().to(device).train()
    head = nn.Linear(128, args.n_clusters).to(device)   # training-only; dropped at export
    if args.init_from:
        ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
        sd = ck.get("phone_extractor", ck)
        missing, unexpected = student.load_state_dict(sd, strict=False)
        print(f"init_from {args.init_from}: missing={len(missing)} unexpected={len(unexpected)}")

    params = list(student.parameters()) + list(head.parameters())
    optim = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.98), weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    start_step = 0
    ckpt_latest = out_dir / "checkpoint_latest.pt"
    if args.resume and ckpt_latest.is_file():
        ck = torch.load(ckpt_latest, map_location="cpu", weights_only=False)
        student.load_state_dict(ck["phone_extractor"])
        head.load_state_dict(ck["head"])
        optim.load_state_dict(ck["optim"])
        start_step = ck["step"]
        print(f"resumed at step {start_step}")

    writer = SummaryWriter(log_dir=str(out_dir))
    it = _infinite(loader)
    pbar = tqdm(total=args.steps, initial=start_step, desc="v4-softvc", dynamic_ncols=True)
    t0 = time.time()
    for step in range(start_step, args.steps):
        lr = cosine_warmup_lr(step, args.warmup_steps, args.steps, args.lr, args.min_lr)
        for g in optim.param_groups:
            g["lr"] = lr

        wav = next(it).to(device, non_blocking=True)        # [B, L]
        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            with torch.amp.autocast("cuda", enabled=False):
                t_feat = teacher(wav.float())               # [B, Tt, 768] @50fps
                labels = assign_units(t_feat, centroids)    # [B, Tt]
            # upsample labels 2x to student's 100 fps
            labels_up = labels.repeat_interleave(2, dim=1)  # [B, 2*Tt]
            s_feat = student(wav.unsqueeze(1), return_stats=False).transpose(1, 2)  # [B, Ts, 128]
            T = min(s_feat.size(1), labels_up.size(1))
            logits = head(s_feat[:, :T])                    # [B, T, K]
            loss = F.cross_entropy(logits.reshape(-1, args.n_clusters),
                                   labels_up[:, :T].reshape(-1),
                                   label_smoothing=args.label_smoothing)

        optim.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            gnorm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            scaler.step(optim); scaler.update()
        else:
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            optim.step()
        pbar.update(1)

        if step % args.log_interval == 0:
            with torch.no_grad():
                acc = (logits.reshape(-1, args.n_clusters).argmax(1)
                       == labels_up[:, :T].reshape(-1)).float().mean().item()
                er = effective_rank(s_feat.detach())
            writer.add_scalar("train/ce_loss", float(loss), step)
            writer.add_scalar("train/unit_acc", acc, step)
            writer.add_scalar("train/effective_rank", er, step)
            writer.add_scalar("train/lr", lr, step)
            writer.add_scalar("train/grad_norm", float(gnorm), step)
            writer.add_scalar("train/it_per_s", (step - start_step + 1) / max(time.time() - t0, 1e-9), step)
            pbar.set_postfix(ce=f"{float(loss):.3f}", acc=f"{acc:.3f}", eff_rank=f"{er:.1f}", lr=f"{lr:.1e}")

        if (step % args.save_interval == 0 and step > start_step) or step == args.steps - 1:
            ck = {"step": step + 1, "phone_extractor": student.state_dict(),
                  "head": head.state_dict(), "optim": optim.state_dict(),
                  "centroids": centroids.cpu(), "args": vars(args)}
            tmp = out_dir / "checkpoint_latest.pt.tmp"
            torch.save(ck, tmp)
            os.replace(tmp, ckpt_latest)
            if (step + 1) % (args.save_interval * 4) == 0 or step == args.steps - 1:
                torch.save(ck, out_dir / f"checkpoint_{step + 1:08d}.pt")
            pbar.write(f"[step {step + 1}] saved checkpoint")

    pbar.close(); writer.close()
    print("done.")


def parse_args():
    p = argparse.ArgumentParser(description="PhoneExtractor v4: Soft-VC unit-classification objective.")
    p.add_argument("--data-dir", required=True, help="root dir of English audio (recursive)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-clusters", type=int, default=500, help="k-means units (Soft-VC uses 100; HuBERT used 100-500)")
    p.add_argument("--kmeans-frames", type=int, default=500_000, help="HuBERT frames sampled to fit k-means")
    p.add_argument("--teacher-layer", type=int, default=9)
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--warmup-steps", type=int, default=5_000)
    p.add_argument("--steps-per-epoch", type=int, default=2_000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--wav-length-sec", type=float, default=4.0)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--min-lr", type=float, default=5e-6)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--init-from", type=str, default="", help="optional warm-start PhoneExtractor")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--save-interval", type=int, default=5_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
