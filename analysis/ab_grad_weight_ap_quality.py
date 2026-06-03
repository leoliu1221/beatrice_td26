"""Extract per-step converted audio from each A/B run's TensorBoard events,
score every clip with UTMOS (the same metric the trainer uses internally),
and plot ap0 vs ap100 quality curves.

This is the substitute for `record_metrics: true` scalar curves: we
reconstruct the validation/utmos signal from the audio that was saved.
"""
from __future__ import annotations

import io
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from tensorboard.backend.event_processing.event_accumulator import (
    AUDIO,
    EventAccumulator,
)

REPO = Path(__file__).resolve().parent.parent
RUNS = [
    ("grad_weight_ap=0",   REPO / "outputs/ab_ap0_v2"),
    ("grad_weight_ap=100", REPO / "outputs/ab_ap100_v2"),
]
OUT_DIR = REPO / "analysis"
OUT_DIR.mkdir(exist_ok=True)


def load_utmos(device: torch.device) -> torch.nn.Module:
    model = torch.hub.load(
        "tarepan/SpeechMOS:v1.0.0", "utmos22_strong", trust_repo=True
    ).eval().to(device)
    return model


def extract_audio(run_dir: Path) -> dict[str, dict[int, tuple[np.ndarray, int]]]:
    """{tag: {step: (waveform[float32, mono], sample_rate)}}"""
    ea = EventAccumulator(str(run_dir), size_guidance={AUDIO: 0})
    ea.Reload()
    out: dict[str, dict[int, tuple[np.ndarray, int]]] = defaultdict(dict)
    for tag in sorted(ea.Tags().get("audio", [])):
        if not tag.startswith("converted/"):
            continue
        for ev in ea.Audio(tag):
            wav, sr = sf.read(io.BytesIO(ev.encoded_audio_string), dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            out[tag][ev.step] = (wav, sr)
    return out


def utmos_score(model: torch.nn.Module, wav: np.ndarray, sr: int,
                device: torch.device) -> float:
    x = torch.from_numpy(wav).float().to(device)
    if sr != 16000:
        x = AF.resample(x, sr, 16000)
    x = x.unsqueeze(0)  # [1, T]
    with torch.inference_mode():
        return float(model(x, sr=16000).item())


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device {device}")
    model = load_utmos(device)

    # {run_label: {step: [utmos_per_clip, ...]}}
    results: dict[str, dict[int, list[float]]] = {}
    # also keep per-clip breakdown for joint listening / inspection
    per_clip: dict[str, dict[str, dict[int, float]]] = {}

    for label, run_dir in RUNS:
        print(f"\n=== {label}  ({run_dir.name}) ===")
        audio = extract_audio(run_dir)
        steps_for_run: dict[int, list[float]] = defaultdict(list)
        clip_for_run: dict[str, dict[int, float]] = {}
        for tag, by_step in sorted(audio.items()):
            clip_for_run[tag] = {}
            for step, (wav, sr) in sorted(by_step.items()):
                score = utmos_score(model, wav, sr, device)
                steps_for_run[step].append(score)
                clip_for_run[tag][step] = score
            steps = sorted(by_step)
            avg_last = np.mean([clip_for_run[tag][s] for s in steps[-3:]])
            print(f"  {tag}: {len(steps)} steps; mean(last 3) UTMOS={avg_last:.3f}")
        results[label] = dict(steps_for_run)
        per_clip[label] = clip_for_run

    # Aggregate + summary
    summary_rows = []
    for label, by_step in results.items():
        for step, scores in sorted(by_step.items()):
            summary_rows.append({
                "run": label,
                "step": step,
                "n_clips": len(scores),
                "utmos_mean": float(np.mean(scores)),
                "utmos_std":  float(np.std(scores)),
            })
    json_out = OUT_DIR / "ab_grad_weight_ap_utmos.json"
    json_out.write_text(json.dumps(
        {"summary": summary_rows, "per_clip": per_clip}, indent=2
    ))
    print(f"\nwrote {json_out.relative_to(REPO)}")

    # ---- Plot ---------------------------------------------------------------
    labels = [lbl for lbl, _ in RUNS]
    steps_all = sorted({s for by_step in results.values() for s in by_step})
    colors = {"grad_weight_ap=0": "#1f77b4", "grad_weight_ap=100": "#d62728"}

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), dpi=130,
                             gridspec_kw={"height_ratios": [3, 2]})

    # Panel 1: mean UTMOS with stderr (stderr eliminates clip-difficulty noise
    # better than std band, since N=12 clips per step)
    ax = axes[0]
    for label in labels:
        by_step = results[label]
        steps = sorted(by_step)
        means = np.array([np.mean(by_step[s]) for s in steps])
        sems  = np.array([np.std(by_step[s], ddof=1) / np.sqrt(len(by_step[s]))
                          for s in steps])
        ax.plot(steps, means, marker="o", label=label, color=colors[label],
                linewidth=2)
        ax.fill_between(steps, means - sems, means + sems,
                        color=colors[label], alpha=0.18)
    ax.set_ylabel("UTMOS (mean ± SEM over 12 clips)")
    ax.set_title(
        "A/B: grad_weight_ap on Beatrice fine-tune of new_lol_data_df\n"
        "(both runs use v2 noise-robust extractors, 60k steps, identical seed/data)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_xticks(steps_all)
    ax.set_xticklabels([f"{s//1000}k" if s >= 1000 else str(s) for s in steps_all])

    # Panel 2: paired difference (ap100 − ap0) per clip per step, then mean
    # ± SEM of those paired diffs. This cancels clip-difficulty variance and is
    # the right statistical test for the A/B.
    ax = axes[1]
    paired_means: list[float] = []
    paired_sems:  list[float] = []
    for s in steps_all:
        diffs = []
        for tag in per_clip[labels[0]]:
            if (s in per_clip[labels[0]][tag] and s in per_clip[labels[1]][tag]):
                diffs.append(per_clip[labels[1]][tag][s] -
                             per_clip[labels[0]][tag][s])
        diffs = np.array(diffs)
        paired_means.append(float(diffs.mean()))
        paired_sems.append(float(diffs.std(ddof=1) / np.sqrt(len(diffs))))
    paired_means_arr = np.array(paired_means)
    paired_sems_arr  = np.array(paired_sems)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.plot(steps_all, paired_means_arr, marker="o", color="#444444",
            linewidth=2, label="ap100 − ap0 (paired by clip)")
    ax.fill_between(steps_all, paired_means_arr - paired_sems_arr,
                    paired_means_arr + paired_sems_arr,
                    color="#444444", alpha=0.18)
    ax.set_xlabel("training step")
    ax.set_ylabel("paired ΔUTMOS\n(positive = ap100 better)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_xticks(steps_all)
    ax.set_xticklabels([f"{s//1000}k" if s >= 1000 else str(s) for s in steps_all])

    fig.tight_layout()
    fig_out = OUT_DIR / "ab_grad_weight_ap_utmos.png"
    fig.savefig(fig_out)
    print(f"wrote {fig_out.relative_to(REPO)}")

    # Console summary
    print("\n=== UTMOS summary (mean over 12 clips) ===")
    print(f"{'step':>6} | {'ap=0':>6} | {'ap=100':>7} | {'Δ (ap100-ap0)':>14}")
    for s, dm, dsem in zip(steps_all, paired_means_arr, paired_sems_arr):
        m0 = np.mean(results[labels[0]][s]) if s in results[labels[0]] else float("nan")
        m1 = np.mean(results[labels[1]][s]) if s in results[labels[1]] else float("nan")
        print(f"{s:>6} | {m0:>6.3f} | {m1:>7.3f} | {dm:>+8.4f} ± {dsem:.4f}")


if __name__ == "__main__":
    main()
