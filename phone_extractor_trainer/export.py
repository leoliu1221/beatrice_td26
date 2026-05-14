"""Convert a phone_extractor_trainer checkpoint into a Beatrice-compatible
pretrained file.

The training checkpoint contains:
    {"step", "phone_extractor", "projection", "optim", "scaler", "args"}

Beatrice's loader (beatrice_trainer/__main__.py: prepare_training) expects:
    {"phone_extractor": <state_dict>}

This script strips everything else (training-only projection head, optimizer
state, etc.) and verifies the result loads cleanly into a fresh PhoneExtractor.

Usage:
    uv run python -m phone_extractor_trainer.export <input_ckpt> <output.pt>

Example:
    uv run python -m phone_extractor_trainer.export \
        outputs/phone_extractor_en/checkpoint_latest.pt \
        assets/pretrained/phone_extractor_en.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from beatrice_trainer.__main__ import PhoneExtractor  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path, help="trainer checkpoint (.pt)")
    ap.add_argument("output", type=Path, help="output Beatrice-compatible .pt")
    args = ap.parse_args()

    print(f"loading {args.input} ...")
    ckpt = torch.load(args.input, map_location="cpu", weights_only=False)
    if "phone_extractor" not in ckpt:
        raise SystemExit(f"input ckpt has no 'phone_extractor' key (keys={list(ckpt)})")
    state_dict = ckpt["phone_extractor"]
    step = ckpt.get("step", "?")
    print(f"  step: {step}, tensors: {len(state_dict)}")

    # Sanity check: must load cleanly into a fresh PhoneExtractor().
    model = PhoneExtractor()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"WARNING missing keys ({len(missing)}): {missing[:5]}{' ...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"WARNING unexpected keys ({len(unexpected)}): {unexpected[:5]}{' ...' if len(unexpected)>5 else ''}")
    if not missing and not unexpected:
        print("  load_state_dict: CLEAN (0 missing, 0 unexpected)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"phone_extractor": state_dict}, args.output)
    size_mb = args.output.stat().st_size / 1024 / 1024
    print(f"wrote {args.output} ({size_mb:.1f} MB)")
    print()
    print("Next: point Beatrice at it. Edit your dataset's config.json (or")
    print("assets/default_config.json) and set:")
    print(f'    "phone_extractor_file": "{args.output.as_posix()}"')


if __name__ == "__main__":
    main()
