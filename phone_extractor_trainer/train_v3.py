"""PhoneExtractor v3: anchored, targeted English correction of jp_122.

Strategy (project_context.md lesson 2.9, experimental_log.md Experiment 5):
the user's listening verdict is that jp_122 is the better extractor because it
captures ALL phonetic features with no muffled "big tongue" mask. So v3 treats
jp_122's richness/articulation as a HARD CONSTRAINT and applies only a small,
targeted English-contrast correction on top of it. From-scratch distillation
against a clean SSL teacher is abandoned (it averages phonemes -> muffle).

Loss = alpha * L_anchor + beta * L_supcon
  - L_anchor (DOMINANT): 1 - cos(student(x), jp_122(x)) on a large pool of
    diverse unlabeled English audio. Pins the rich jp_122 geometry everywhere;
    this is the structural safeguard against the muffle.
  - L_supcon (SMALL nudge): supervised contrastive loss over frame features of
    a DISJOINT labelled set (MFA phonemes, ls_aligned_train.pkl). Pulls same-
    phone frames together and pushes different-phone frames apart, directly
    improving the English contrasts jp_122 merges (R/L, TH, V, DH ...), WITHOUT
    asking the 128-dim student to imitate HuBERT's full 768-dim manifold.

The student is warm-started from jp_122; jp_122 is also held frozen as anchor.
`effective_rank` of the student is logged every interval as the richness gate
(must not regress below ~jp_122's 18.8 -- abort the run if it does).

Usage (Go/No-Go pilot):
    uv run python -m phone_extractor_trainer.train_v3 \
        --init-from assets/pretrained/phone_extractor/jp_122_3000k.pt \
        --anchor-data-dir datasets/librispeech/LibriSpeech/train-clean-100 \
        --train-pkl analysis/data/ls_aligned_train.pkl \
        --out-dir outputs/phone_extractor_en_v3 \
        --steps 80000

Export + judge afterwards exactly like v1/v2 (checkpoint stores
`phone_extractor` state dict, so phone_extractor_trainer.export and the probes
read it unchanged).
"""
from __future__ import annotations

import argparse
import math
import os
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from beatrice_trainer.__main__ import PhoneExtractor  # noqa: E402
from phone_extractor_trainer.data import WavCropDataset, discover_audio_files  # noqa: E402

DROP = {"spn", "sil", ""}
SR = 16000
SAMPLES_PER_FRAME = 160  # PhoneExtractor runs at 100 fps from 16 kHz

# The specific English contrasts jp_122 merges (the "Japanese accent"); a
# diffuse SupCon under a dominant anchor does NOT move these (v3 80k result),
# so we push them apart explicitly. Collapsed (stress-stripped) phone labels.
HARD_PAIRS = [("R", "L"), ("TH", "S"), ("V", "B"), ("DH", "D"),
              ("S", "SH"), ("IH", "IY"), ("AE", "EH")]


def collapse(lbl: str) -> str:
    return re.sub(r"\d+$", "", lbl)


# ---------------------------------------------------------------------------
# Labelled dataset for the supervised-contrastive correction term
# ---------------------------------------------------------------------------

class LabeledPhoneDataset(Dataset):
    """Yields (wav_crop[L], frame_labels[T]) from an MFA-aligned pickle.

    frame_labels[t] is the collapsed-phoneme id covering the centre of frame t,
    or -1 for silence / out-of-segment / dropped frames (ignored by the loss).
    """

    def __init__(self, items, label2id, wav_length, samples_per_epoch, seed=0):
        self.items = items
        self.label2id = label2id
        self.wav_length = wav_length
        self.n_frames = wav_length // SAMPLES_PER_FRAME
        self.samples_per_epoch = samples_per_epoch
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        u = self.items[self.rng.integers(len(self.items))]
        wav = u["wav"].astype(np.float32)
        T = len(wav)
        if T < self.wav_length:
            wav = np.pad(wav, (0, self.wav_length - T), mode="reflect" if T > 1 else "constant")
            start = 0
        else:
            start = int(self.rng.integers(0, T - self.wav_length + 1))
        crop = wav[start : start + self.wav_length]

        labels = np.full(self.n_frames, -1, dtype=np.int64)
        segs = u["phonemes"]
        for t in range(self.n_frames):
            centre_s = (start + t * SAMPLES_PER_FRAME + SAMPLES_PER_FRAME / 2) / SR
            for lab, s, e in segs:
                if s <= centre_s < e:
                    c = collapse(lab)
                    if c not in DROP and c in self.label2id:
                        labels[t] = self.label2id[c]
                    break
        return torch.from_numpy(crop), torch.from_numpy(labels)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def anchor_loss(student_feat, anchor_feat):
    # [B, C, T] each. cosine over channel dim per frame, averaged.
    cos = F.cosine_similarity(student_feat, anchor_feat, dim=1)  # [B, T]
    return (1.0 - cos).mean(), cos.mean().detach()


def supcon_loss(feats, labels, tau, max_frames, rng):
    """Supervised contrastive loss (Khosla et al. 2020) over labelled frames.

    feats: [B, T, C] student features. labels: [B, T] long, -1 = ignore.
    """
    f = feats.reshape(-1, feats.size(-1))
    y = labels.reshape(-1)
    valid = y >= 0
    f, y = f[valid], y[valid]
    if f.size(0) < 4:
        return feats.new_zeros(()), 0
    if f.size(0) > max_frames:
        sel = torch.from_numpy(rng.choice(f.size(0), size=max_frames, replace=False)).to(f.device)
        f, y = f[sel], y[sel]
    f = F.normalize(f, dim=1)
    sim = (f @ f.t()) / tau                       # [M, M]
    M = f.size(0)
    eye = torch.eye(M, dtype=torch.bool, device=f.device)
    sim = sim.masked_fill(eye, -1e9)              # exclude self
    logits = sim - sim.max(dim=1, keepdim=True).values.detach()
    exp = logits.exp()
    denom = exp.sum(dim=1)                         # over all a != i
    pos_mask = (y[:, None] == y[None, :]) & ~eye   # same-label, not self
    has_pos = pos_mask.any(dim=1)
    if has_pos.sum() == 0:
        return feats.new_zeros(()), 0
    log_prob = logits - denom.clamp_min(1e-12).log()[:, None]
    pos_log_prob = (log_prob * pos_mask).sum(1) / pos_mask.sum(1).clamp_min(1)
    loss = -pos_log_prob[has_pos].mean()
    return loss, int(has_pos.sum())


def hardpair_loss(feats, labels, pair_ids, margin, max_per_class, rng):
    """Push merged English contrasts apart (surgical accent correction).

    For each (a,b) confusable pair, penalise high cosine similarity between
    a-frames and b-frames: loss = mean relu(cos(a_i, b_j) - margin). Drives the
    two phone clouds below `margin` cosine similarity (default 0 -> orthogonal),
    targeting exactly R/L, TH/S, V/B, DH/D that the anchor pins together.
    """
    f = F.normalize(feats.reshape(-1, feats.size(-1)), dim=1)
    y = labels.reshape(-1)
    total = feats.new_zeros(())
    n_active = 0
    for ai, bi in pair_ids:
        ia = (y == ai).nonzero(as_tuple=True)[0]
        ib = (y == bi).nonzero(as_tuple=True)[0]
        if ia.numel() == 0 or ib.numel() == 0:
            continue
        if ia.numel() > max_per_class:
            ia = ia[torch.from_numpy(rng.choice(ia.cpu().numpy(), max_per_class, replace=False)).to(ia.device)]
        if ib.numel() > max_per_class:
            ib = ib[torch.from_numpy(rng.choice(ib.cpu().numpy(), max_per_class, replace=False)).to(ib.device)]
        sim = f[ia] @ f[ib].t()                       # [na, nb] cross-pair cosine
        total = total + F.relu(sim - margin).mean()
        n_active += 1
    if n_active == 0:
        return feats.new_zeros(()), 0
    return total / n_active, n_active


def effective_rank(feats):
    """Participation ratio of the feature covariance (richness gate)."""
    f = feats.reshape(-1, feats.size(-1)).float()
    f = f - f.mean(0, keepdim=True)
    cov = (f.t() @ f) / max(f.size(0) - 1, 1)
    eig = torch.linalg.eigvalsh(cov).clamp_min(0)
    return float((eig.sum() ** 2) / (eig.square().sum() + 1e-12))


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

def cosine_warmup_lr(step, warmup, total, base_lr, min_lr):
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    p = min(max((step - warmup) / max(1, total - warmup), 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * p))


def _infinite(loader):
    while True:
        yield from loader


def load_phone_extractor(path, device):
    m = PhoneExtractor().to(device)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("phone_extractor", ck)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    print(f"load {Path(path).name}: missing={len(missing)} unexpected={len(unexpected)}")
    return m


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

def train(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"device: {device}")

    wav_length = (int(round(args.wav_length_sec * SR)) // SAMPLES_PER_FRAME) * SAMPLES_PER_FRAME

    # ---- labelled data (SupCon)
    items = pickle.load(open(args.train_pkl, "rb"))
    labelset = sorted({collapse(l) for u in items for l, _, _ in u["phonemes"]} - DROP)
    label2id = {c: i for i, c in enumerate(labelset)}
    pair_ids = [(label2id[a], label2id[b]) for a, b in HARD_PAIRS
                if a in label2id and b in label2id]
    print(f"labelled utts: {len(items)} | phone classes: {len(label2id)} | "
          f"hard pairs present: {len(pair_ids)}/{len(HARD_PAIRS)}")
    lab_ds = LabeledPhoneDataset(items, label2id, wav_length,
                                 samples_per_epoch=args.labeled_batch_size * args.steps_per_epoch,
                                 seed=args.seed)
    lab_loader = DataLoader(lab_ds, batch_size=args.labeled_batch_size,
                            num_workers=args.num_workers, drop_last=True,
                            pin_memory=device.type == "cuda")

    # ---- anchor data (diverse unlabelled English)
    files = discover_audio_files(Path(args.anchor_data_dir))
    print(f"anchor pool: {len(files)} files")
    anc_ds = WavCropDataset(files=files, wav_length=wav_length,
                            samples_per_epoch=args.batch_size * args.steps_per_epoch,
                            sample_rate=SR, seed=args.seed)
    anc_loader = DataLoader(anc_ds, batch_size=args.batch_size,
                            num_workers=args.num_workers, drop_last=True,
                            pin_memory=device.type == "cuda")

    # ---- models
    student = load_phone_extractor(args.init_from, device).train()
    anchor = load_phone_extractor(args.init_from, device).eval().requires_grad_(False)

    optim = torch.optim.AdamW(student.parameters(), lr=args.lr,
                              betas=(0.9, 0.98), weight_decay=args.weight_decay)

    start_step = 0
    ckpt_latest = out_dir / "checkpoint_latest.pt"
    if args.resume and ckpt_latest.is_file():
        ck = torch.load(ckpt_latest, map_location="cpu", weights_only=False)
        student.load_state_dict(ck["phone_extractor"])
        optim.load_state_dict(ck["optim"])
        start_step = ck["step"]
        print(f"resumed at step {start_step}")

    writer = SummaryWriter(log_dir=str(out_dir))
    rng = np.random.default_rng(args.seed)
    lab_it, anc_it = _infinite(lab_loader), _infinite(anc_loader)

    pbar = tqdm(total=args.steps, initial=start_step, desc="v3", dynamic_ncols=True)
    t0 = time.time()
    for step in range(start_step, args.steps):
        lr = cosine_warmup_lr(step, args.warmup_steps, args.steps, args.lr, args.min_lr)
        for g in optim.param_groups:
            g["lr"] = lr

        anc_wav = next(anc_it).to(device, non_blocking=True)          # [B, L]
        lab_wav, lab_y = next(lab_it)
        lab_wav = lab_wav.to(device, non_blocking=True)
        lab_y = lab_y.to(device, non_blocking=True)

        student_anc = student(anc_wav.unsqueeze(1), return_stats=False)        # [B,128,T]
        with torch.no_grad():
            anchor_anc = anchor(anc_wav.unsqueeze(1), return_stats=False)
        l_anchor, cos_anc = anchor_loss(student_anc, anchor_anc)

        student_lab = student(lab_wav.unsqueeze(1), return_stats=False)        # [B,128,T]
        lab_feat = student_lab.transpose(1, 2)                                 # [B,T,128]
        l_supcon, n_anchors = supcon_loss(lab_feat, lab_y,
                                          args.tau, args.max_supcon_frames, rng)
        l_hardpair, _ = hardpair_loss(lab_feat, lab_y, pair_ids,
                                      args.hardpair_margin, args.max_supcon_frames, rng)

        loss = args.alpha * l_anchor + args.beta * l_supcon + args.gamma * l_hardpair
        optim.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
        optim.step()
        pbar.update(1)

        if step % args.log_interval == 0:
            er = effective_rank(student_anc.transpose(1, 2).detach())
            writer.add_scalar("train/loss", float(loss), step)
            writer.add_scalar("train/l_anchor", float(l_anchor), step)
            writer.add_scalar("train/l_supcon", float(l_supcon), step)
            writer.add_scalar("train/l_hardpair", float(l_hardpair), step)
            writer.add_scalar("train/anchor_cos", float(cos_anc), step)
            writer.add_scalar("train/effective_rank", er, step)
            writer.add_scalar("train/lr", lr, step)
            writer.add_scalar("train/grad_norm", float(gnorm), step)
            writer.add_scalar("train/it_per_s", (step - start_step + 1) / max(time.time() - t0, 1e-9), step)
            pbar.set_postfix(anc=f"{cos_anc:.3f}", sup=f"{float(l_supcon):.3f}",
                             hp=f"{float(l_hardpair):.3f}", eff_rank=f"{er:.1f}", lr=f"{lr:.1e}")

        if step % args.save_interval == 0 and step > start_step or step == args.steps - 1:
            ck = {"step": step + 1, "phone_extractor": student.state_dict(),
                  "optim": optim.state_dict(), "args": vars(args), "label2id": label2id}
            tmp = out_dir / "checkpoint_latest.pt.tmp"
            torch.save(ck, tmp)
            os.replace(tmp, ckpt_latest)
            if (step + 1) % (args.save_interval * 4) == 0 or step == args.steps - 1:
                torch.save(ck, out_dir / f"checkpoint_{step + 1:08d}.pt")
            pbar.write(f"[step {step + 1}] saved checkpoint")

    pbar.close()
    writer.close()
    print("done.")


def parse_args():
    p = argparse.ArgumentParser(description="PhoneExtractor v3: anchored English correction of jp_122.")
    p.add_argument("--init-from", required=True, help="jp_122 checkpoint (warm-start AND frozen anchor)")
    p.add_argument("--anchor-data-dir", required=True, help="diverse unlabelled English audio for the anchor term")
    p.add_argument("--train-pkl", required=True, help="MFA-aligned labelled pickle (DISJOINT from eval set)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--steps", type=int, default=80_000)
    p.add_argument("--warmup-steps", type=int, default=1_000)
    p.add_argument("--steps-per-epoch", type=int, default=2_000)
    p.add_argument("--batch-size", type=int, default=16, help="anchor batch")
    p.add_argument("--labeled-batch-size", type=int, default=8, help="SupCon batch")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--wav-length-sec", type=float, default=4.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=1.0, help="anchor (richness preservation) weight")
    p.add_argument("--beta", type=float, default=0.3, help="SupCon (broad English correction) weight")
    p.add_argument("--gamma", type=float, default=0.5,
                   help="hard-pair separation weight -- surgical accent fix on R/L,TH/S,V/B,DH/D")
    p.add_argument("--hardpair-margin", type=float, default=0.0,
                   help="target max cosine sim between confusable-pair frames (0 = orthogonal)")
    p.add_argument("--tau", type=float, default=0.1, help="SupCon temperature")
    p.add_argument("--max-supcon-frames", type=int, default=1024)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--save-interval", type=int, default=5_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
