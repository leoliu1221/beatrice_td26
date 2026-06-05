"""Richness monitor + early-terminator for the v4 PhoneExtractor training.

Watches an out-dir for new numbered checkpoints (checkpoint_XXXXXXXX.pt). For
each new one it computes the **richness score = effective_rank** (participation
ratio of the feature covariance, identical to analysis/probe_phone_extractor.py)
averaged over a FIXED held-out clip set, logs all richness metrics to a CSV, and
applies an early-stop rule:

    drop = score strictly below the best score seen so far (running max).
    `--patience` consecutive checkpoints below the running peak -> SIGTERM the
    training process and exit.

A new running-max (score >= best) resets the consecutive-drop counter.

Usage:
    uv run python -m phone_extractor_trainer.watch_richness \
        --out-dir outputs/phone_extractor_en_v4_k2000 \
        --train-pid 12345 --patience 5 --n-clips 40
"""
from __future__ import annotations

import argparse
import csv
import os
import signal
import sys
import time
from pathlib import Path

import torch
import torchaudio

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from beatrice_trainer.__main__ import PhoneExtractor  # noqa: E402
from phone_extractor_trainer.data import discover_audio_files  # noqa: E402

SR = 16000


def load_clip_16k(path: Path, device, max_sec: float = 6.0):
    wav, sr = torchaudio.load(str(path), backend="soundfile")
    if wav.size(0) > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    wav = wav[:, : int(max_sec * SR)]
    n = (wav.size(1) // 160) * 160
    return wav[:, :n].to(device)


def contrast_metrics(feats: torch.Tensor) -> dict:
    """feats: [T, C] for one clip. Matches analysis/probe_phone_extractor.py."""
    f = feats - feats.mean(0, keepdim=True)
    T, _ = f.shape
    norms = feats.norm(dim=1).clamp_min(1e-6)
    deltas = (feats[1:] - feats[:-1]).norm(dim=1)
    temporal_contrast = float((deltas / norms[:-1]).mean())
    fn = feats / feats.norm(dim=1, keepdim=True).clamp_min(1e-6)
    sim = fn @ fn.t()
    off = sim[~torch.eye(T, dtype=torch.bool)]
    mean_pairwise_cos = float(off.mean())
    cov = (f.t() @ f) / max(T - 1, 1)
    eig = torch.linalg.eigvalsh(cov).clamp_min(0)
    eff_rank = float((eig.sum() ** 2) / (eig.square().sum() + 1e-12))
    return {"effective_rank": eff_rank,
            "temporal_contrast": temporal_contrast,
            "mean_pairwise_cos": mean_pairwise_cos}


@torch.inference_mode()
def score_checkpoint(ckpt_path: Path, clips, device) -> dict:
    model = PhoneExtractor().to(device).eval().requires_grad_(False)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["phone_extractor"] if "phone_extractor" in ck else ck
    model.load_state_dict(sd, strict=False)
    accum: dict[str, list] = {}
    for wav in clips:
        feats = model.units(wav.unsqueeze(0)).squeeze(0).float().cpu()
        for k, v in contrast_metrics(feats).items():
            accum.setdefault(k, []).append(v)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {k: sum(v) / len(v) for k, v in accum.items()}


def step_of(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except ValueError:
        return -1


def numbered_checkpoints(out_dir: Path):
    cks = [p for p in out_dir.glob("checkpoint_*.pt")
           if p.stem.split("_")[-1].isdigit()]
    return sorted(cks, key=step_of)


def wait_stable(path: Path, tries: int = 10, delay: float = 3.0) -> bool:
    """Wait until a checkpoint file's size stops changing (write finished)."""
    last = -1
    for _ in range(tries):
        try:
            sz = path.stat().st_size
        except FileNotFoundError:
            return False
        if sz == last and sz > 0:
            return True
        last = sz
        time.sleep(delay)
    return True


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return True  # not tracking a pid
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser(description="v4 richness monitor + early stop")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train-pid", type=int, default=0, help="training PID to SIGTERM on trigger")
    ap.add_argument("--clips-dir", default="datasets/librispeech/LibriSpeech/train-clean-100")
    ap.add_argument("--n-clips", type=int, default=40)
    ap.add_argument("--patience", type=int, default=5, help="consecutive checkpoints below running max -> stop")
    ap.add_argument("--poll-interval", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-final-step", type=int, default=200000, help="exit after scoring this step")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fixed held-out clip set (deterministic).
    files = discover_audio_files(Path(args.clips_dir))
    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(files), generator=g)[: args.n_clips].tolist()
    clips = [load_clip_16k(files[i], device) for i in idx]
    print(f"[monitor] device={device} | {len(clips)} fixed clips | "
          f"patience={args.patience} (drops below running-max) | pid={args.train_pid}",
          flush=True)

    csv_path = out_dir / "richness_monitor.csv"
    new_csv = not csv_path.is_file()
    csv_f = open(csv_path, "a", newline="")
    writer = csv.writer(csv_f)
    if new_csv:
        writer.writerow(["step", "effective_rank", "best_so_far", "below_peak_streak",
                         "temporal_contrast", "mean_pairwise_cos", "ts"])
        csv_f.flush()

    seen: set[int] = set()
    best = float("-inf")
    streak = 0

    while True:
        # exit if training process has ended and we've drained checkpoints
        train_done = not pid_alive(args.train_pid)

        pending = [p for p in numbered_checkpoints(out_dir) if step_of(p) not in seen]
        for ck in pending:
            step = step_of(ck)
            if not wait_stable(ck):
                continue
            try:
                m = score_checkpoint(ck, clips, device)
            except Exception as e:  # noqa: BLE001
                print(f"[monitor] step {step}: scoring failed ({e}); will retry", flush=True)
                continue
            seen.add(step)
            er = m["effective_rank"]
            if er >= best:
                best = er
                streak = 0
                flag = "NEW PEAK"
            else:
                streak += 1
                flag = f"below peak {streak}/{args.patience}"
            print(f"[monitor] step {step:>7} | eff_rank={er:6.2f} | best={best:6.2f} | "
                  f"tc={m['temporal_contrast']:.3f} cos={m['mean_pairwise_cos']:.3f} | {flag}",
                  flush=True)
            writer.writerow([step, f"{er:.4f}", f"{best:.4f}", streak,
                             f"{m['temporal_contrast']:.4f}", f"{m['mean_pairwise_cos']:.4f}",
                             time.strftime("%Y-%m-%d %H:%M:%S")])
            csv_f.flush()

            if streak >= args.patience:
                print(f"[monitor] EARLY STOP: eff_rank below running-max {streak} checkpoints "
                      f"in a row. Terminating training pid {args.train_pid}.", flush=True)
                if args.train_pid > 0 and pid_alive(args.train_pid):
                    os.kill(args.train_pid, signal.SIGTERM)
                csv_f.close()
                return
            if step >= args.max_final_step:
                print("[monitor] reached final step; exiting.", flush=True)
                csv_f.close()
                return

        if train_done and not pending:
            print("[monitor] training process ended and all checkpoints scored; exiting.", flush=True)
            csv_f.close()
            return
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
