"""English phonetic-faithfulness probe (label-free).

Question this answers: does a PhoneExtractor's feature geometry match the
*English* phonetic geometry, or has it carved its space around the wrong
(e.g. Japanese) categories? We use the English HuBERT-BASE layer-9 teacher as
the ground-truth English geometry (it is trained on LibriSpeech-960 and is the
backbone of English VC), and measure how well each candidate preserves that
structure on English speech.

Metrics (computed on energy-filtered, frame-aligned features pooled across many
LibriSpeech clips):
  - knn_overlap : for each frame, fraction of its k nearest neighbours in
                  HuBERT space that are also among its k nearest neighbours in
                  the candidate space (cosine distance). 1.0 = identical
                  phonetic neighbourhoods; ~k/N = random. THE key metric for
                  "are English phonemes organised the same way".
  - lin_cka     : linear Centered Kernel Alignment between the HuBERT and
                  candidate feature matrices (rotation/linear-map invariant
                  representational similarity). 1.0 = same structure.

Interpretation:
  - HuBERT-L9 vs itself -> knn_overlap=1.0, cka=1.0 (sanity).
  - A faithful English distillation should score HIGH (it was trained to match
    HuBERT). If en_clean scores LOW, our distillation FAILED to transfer English
    structure (a method problem). jp_122's score quantifies how English-aligned
    (vs Japanese-leaning) the upstream extractor is.
  - NOTE: en_* were trained against this teacher, so their score is partly
    "fit to objective"; this probe is for English-alignment diagnosis, not final
    VC quality. The TIMIT ABX probe (next) gives a label-based, non-circular read.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from beatrice_trainer.__main__ import PhoneExtractor  # noqa: E402
from phone_extractor_trainer.train import HubertTeacher  # noqa: E402

PHONE_VARIANTS = {
    "jp_122_upstream":      REPO / "assets/pretrained/phone_extractor/jp_122_3000k.pt",
    "en_clean_580k":        REPO / "assets/pretrained/phone_extractor/en_clean_580k.pt",
    "en_clean_200k":        REPO / "assets/pretrained/phone_extractor/en_clean_200k.pt",
    "en_nr_targetmix_300k": REPO / "assets/pretrained/phone_extractor/en_nr_targetmix02_300k.pt",
}
LIBRI_ROOT = REPO / "datasets/librispeech/LibriSpeech/train-clean-100"

N_CLIPS = 40
MAX_SEC = 6.0
N_FRAMES = 4000      # pooled frames for the structural comparison
K = 10               # neighbourhood size
ENERGY_PCTL = 35     # drop the quietest X% of frames (silence/low-energy)
SEED = 0


def discover_clips(n: int) -> list[Path]:
    files = sorted(LIBRI_ROOT.rglob("*.flac"))
    if not files:
        raise RuntimeError(f"no flac under {LIBRI_ROOT}")
    idx = np.linspace(0, len(files) - 1, num=min(n, len(files))).astype(int)
    return [files[i] for i in idx]


def load_clip_16k(path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path), backend="soundfile")
    if wav.size(0) > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav[:, : int(MAX_SEC * 16000)]
    n = (wav.size(1) // 160) * 160
    return wav[:, :n]  # [1, T]


def load_phone_extractor(path: Path, device) -> PhoneExtractor:
    m = PhoneExtractor().to(device).eval().requires_grad_(False)
    ck = torch.load(path, map_location="cpu", weights_only=True)
    sd = ck["phone_extractor"] if "phone_extractor" in ck else ck
    m.load_state_dict(sd, strict=False)
    return m


def frame_energy(wav_1t: torch.Tensor, n_frames: int) -> np.ndarray:
    # mean-square energy per 160-sample hop, resampled to n_frames
    w = wav_1t.squeeze(0)
    T = (w.numel() // 160) * 160
    e = w[:T].reshape(-1, 160).pow(2).mean(1).sqrt()  # [T/160]
    e = F.interpolate(e[None, None], size=n_frames, mode="linear", align_corners=False)
    return e.squeeze().cpu().numpy()


@torch.inference_mode()
def collect_features(device) -> dict[str, np.ndarray]:
    clips = discover_clips(N_CLIPS)
    teacher = HubertTeacher(layer_index=9).to(device).eval()
    extractors = {n: load_phone_extractor(p, device) for n, p in PHONE_VARIANTS.items() if p.is_file()}

    pooled: dict[str, list[np.ndarray]] = {"_hubert": [], "_energy": []}
    for n in extractors:
        pooled[n] = []

    for path in clips:
        wav = load_clip_16k(path).to(device)
        # HuBERT [1, T_h, 768]; align to student frame rate (100 fps) below.
        h = teacher(wav.float())  # [1, T_h, 768]
        # Use the first candidate to define target T_s (all share 100 fps).
        any_ext = next(iter(extractors.values()))
        s = any_ext.units(wav.unsqueeze(0)).squeeze(0)  # [T_s, 128]
        T_s = s.size(0)
        h_al = F.interpolate(h.transpose(1, 2), size=T_s, mode="linear",
                             align_corners=False).squeeze(0).transpose(0, 1)  # [T_s, 768]
        pooled["_hubert"].append(h_al.float().cpu().numpy())
        pooled["_energy"].append(frame_energy(wav, T_s))
        for n, ext in extractors.items():
            f = ext.units(wav.unsqueeze(0)).squeeze(0).float().cpu().numpy()  # [T_s, 128]
            pooled[n].append(f)
    return {k: (np.concatenate(v, 0) if k != "_energy" else np.concatenate(v, 0)) for k, v in pooled.items()}


def subsample_voiced(feats: dict[str, np.ndarray], n: int, seed: int) -> dict[str, np.ndarray]:
    energy = feats["_energy"]
    thr = np.percentile(energy, ENERGY_PCTL)
    keep = np.where(energy > thr)[0]
    rng = np.random.default_rng(seed)
    if len(keep) > n:
        keep = rng.choice(keep, size=n, replace=False)
    keep.sort()
    return {k: v[keep] for k, v in feats.items() if k != "_energy"}


def knn_sets(X: np.ndarray, k: int, device) -> torch.Tensor:
    t = torch.from_numpy(X).to(device).float()
    t = F.normalize(t, dim=1)
    sim = t @ t.t()                       # cosine similarity
    sim.fill_diagonal_(-2.0)
    return sim.topk(k, dim=1).indices     # [N, k]


def knn_overlap(Href: np.ndarray, C: np.ndarray, k: int, device) -> float:
    nn_h = knn_sets(Href, k, device)
    nn_c = knn_sets(C, k, device)
    N = nn_h.size(0)
    inter = 0
    for i in range(N):
        a = set(nn_h[i].tolist()); b = set(nn_c[i].tolist())
        inter += len(a & b)
    return inter / (N * k)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    # HSIC-based linear CKA
    xtx = X.T @ X
    yty = Y.T @ Y
    xty = X.T @ Y
    hsic = (xty ** 2).sum()
    denom = np.sqrt((xtx ** 2).sum() * (yty ** 2).sum()) + 1e-12
    return float(hsic / denom)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\nClips: {N_CLIPS}, pooled voiced frames: ~{N_FRAMES}, k={K}\n")
    feats = collect_features(device)
    feats = subsample_voiced(feats, N_FRAMES, SEED)
    Href = feats["_hubert"]
    N = Href.shape[0]
    print(f"actual frames after voicing filter/subsample: {N}\n")

    # sanity: HuBERT vs itself
    rows = [("HuBERT-L9 (self/ref)", knn_overlap(Href, Href, K, device), linear_cka(Href, Href))]
    # control: best-case 128-dim LINEAR compression of HuBERT (PCA-128). This is
    # the ceiling any 128-dim student could reach via a linear map; it bounds how
    # much neighbourhood structure is even retainable at 128 dims.
    Hc = Href - Href.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Hc, full_matrices=False)
    pca128 = Hc @ Vt[:128].T
    rows.append(("HuBERT PCA-128 (128-dim ceiling)", knn_overlap(Href, pca128, K, device), linear_cka(Href, pca128)))
    for name in PHONE_VARIANTS:
        if name in feats:
            rows.append((name, knn_overlap(Href, feats[name], K, device), linear_cka(Href, feats[name])))

    print(f"{'extractor':<24}{'knn_overlap':>14}{'lin_cka':>12}")
    print("-" * 50)
    for name, ko, cka in rows:
        print(f"{name:<24}{ko:>14.4f}{cka:>12.4f}")
    print(f"\nrandom-chance knn_overlap ~= k/N = {K / N:.4f}")
    print("Higher = more English-aligned (closer to the HuBERT English geometry).")


if __name__ == "__main__":
    main()
