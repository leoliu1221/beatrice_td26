"""Sweep `eval.py` across every step-numbered checkpoint in a training dir.

Useful for detecting:
  * **peak before overfit**: target-domain cos_sim (c, d) climbed then declined
    before the run ended -- the "final" checkpoint is past optimal.
  * **plateau**: metrics flat over many checkpoints -> training is wasting compute.
  * **slow regression**: subtle multi-checkpoint drop that single-point eval misses.

Audio pools and pre-augmented eval crops are built ONCE and reused across all
checkpoints, so all numbers are directly comparable.

Outputs a markdown table to stdout and a CSV next to the trainer output dir.

Usage:
    uv run python -m phone_extractor_trainer.eval_sweep \\
        --ckpt-glob 'outputs/phone_extractor_en/checkpoint_*.pt' \\
        --every 50000
"""
from __future__ import annotations

import argparse
import csv
import glob
import re
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from phone_extractor_trainer.eval import build_eval_context, eval_checkpoint  # noqa: E402


_STEP_RE = re.compile(r"checkpoint_(\d+)\.pt$")


def discover_checkpoints(pattern: str) -> list[tuple[int, Path]]:
    """Return [(step, path), ...] for every matching numbered checkpoint."""
    out: list[tuple[int, Path]] = []
    for s in glob.glob(pattern):
        p = Path(s)
        m = _STEP_RE.search(p.name)
        if not m:
            continue
        out.append((int(m.group(1)), p))
    out.sort()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--ckpt-glob",
        default="outputs/phone_extractor_en/checkpoint_*.pt",
        help="glob pattern for step-numbered checkpoints (excludes _latest)",
    )
    p.add_argument("--every", type=int, default=50_000,
                   help="only eval checkpoints whose step is a multiple of this")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=10_000_000)
    p.add_argument("--librispeech-dir", type=str,
                   default="datasets/librispeech/LibriSpeech/train-clean-100")
    p.add_argument("--target-dir", type=str, default="inputs/new_lol_data")
    p.add_argument("--noise-dir", type=str, default="assets/noise")
    p.add_argument("--ir-dir", type=str, default="assets/ir")
    p.add_argument("--n-samples", type=int, default=32,
                   help="smaller default than eval.py since we run many ckpts")
    p.add_argument("--wav-length-sec", type=float, default=4.0)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-csv", type=str, default="",
                   help="optional CSV path; defaults to <ckpt-dir>/eval_sweep.csv")
    args = p.parse_args()

    ckpts = discover_checkpoints(args.ckpt_glob)
    ckpts = [
        (step, path) for step, path in ckpts
        if args.start <= step <= args.end and step % args.every == 0
    ]
    if not ckpts:
        raise SystemExit(f"no matching checkpoints under {args.ckpt_glob}")

    print(f"evaluating {len(ckpts)} checkpoints (every {args.every:,} steps "
          f"in [{args.start:,}, {args.end:,}]) with n_samples={args.n_samples}")
    for step, path in ckpts:
        print(f"  {step:>9,}  {path}")

    device = torch.device(args.device)
    wav_length = (int(round(args.wav_length_sec * 16000)) // 160) * 160

    print("\nbuilding eval context (audio pools + pre-augmented crops) ...")
    ctx = build_eval_context(
        args.librispeech_dir, args.target_dir, args.noise_dir, args.ir_dir,
        args.n_samples, wav_length, device, args.seed,
    )
    print(f"  librispeech: {len(ctx['librispeech_files'])}  "
          f"target: {len(ctx['target_files'])}")

    rows: list[dict] = []
    for step, path in ckpts:
        try:
            r = eval_checkpoint(path, ctx)
        except Exception as e:  # noqa: BLE001
            print(f"  [step {step}] FAILED: {e}")
            continue
        rows.append(r)
        print(
            f"  step {r['step']:>9,}: "
            f"a={r['a_mean']:.4f}  b={r['b_mean']:.4f}  "
            f"c={r['c_mean']:.4f}  d={r['d_mean']:.4f}"
        )

    # ---- markdown summary -----------------------------------------------------
    print()
    print("| step | (a) clean LS | (b) aug LS | (c) clean target | (d) aug target |")
    print("|---:|---:|---:|---:|---:|")
    best_c = max(rows, key=lambda r: r["c_mean"])
    best_d = max(rows, key=lambda r: r["d_mean"])
    for r in rows:
        bc = " <-- best (c)" if r is best_c else ""
        bd = " <-- best (d)" if r is best_d else ""
        flag = bc + bd
        print(f"| {r['step']:,} | {r['a_mean']:.4f} | {r['b_mean']:.4f} "
              f"| {r['c_mean']:.4f} | {r['d_mean']:.4f} |{flag}")

    print()
    print(f"Best (c) clean target: step {best_c['step']:,} -> {best_c['c_mean']:.4f}")
    print(f"Best (d) aug   target: step {best_d['step']:,} -> {best_d['d_mean']:.4f}")

    # ---- CSV ------------------------------------------------------------------
    if args.out_csv:
        csv_path = Path(args.out_csv)
    else:
        # Place CSV in the same dir as the checkpoints.
        any_path = ckpts[0][1]
        csv_path = any_path.parent / "eval_sweep.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "a_mean", "a_std", "b_mean", "b_std",
                    "c_mean", "c_std", "d_mean", "d_std"])
        for r in rows:
            w.writerow([r["step"],
                        f"{r['a_mean']:.4f}", f"{r['a_std']:.4f}",
                        f"{r['b_mean']:.4f}", f"{r['b_std']:.4f}",
                        f"{r['c_mean']:.4f}", f"{r['c_std']:.4f}",
                        f"{r['d_mean']:.4f}", f"{r['d_std']:.4f}"])
    print(f"\nCSV written -> {csv_path}")


if __name__ == "__main__":
    main()
