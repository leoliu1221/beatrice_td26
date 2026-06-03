"""Label-based English phoneme probe (ABX + linear classification).

Non-circular counterpart to probe_english_faithfulness.py: uses REAL phoneme
labels (MFA alignments from analysis/data/ls_aligned.pkl) to measure how well
each PhoneExtractor separates English phonemes -- especially the contrasts that
are absent in Japanese and drive the "sounds Japanese" complaint.

Metrics per extractor (features pooled over the central part of each labelled
phoneme segment, stress collapsed AA1->AA, 'spn'/silence dropped):
  - abx_error    : ABX discriminability. For a triplet (a1,a2 same phone, b
                   different), error if cos_dist(a1,a2) > cos_dist(a1,b).
                   Lower = phonemes better separated. Chance = 0.5.
  - phone_acc    : top-1 accuracy of a linear softmax probe trained on frozen
                   features (utterance-disjoint train/test). Higher = features
                   linearly encode phoneme identity.
  - hard-pair ABX: per-contrast ABX error for /r/-/l/, th-/s/, /v/-/b/, etc.

HuBERT-L9 is included as the English upper bound. The en_* extractors are NOT
trained on these labels, so unlike the faithfulness probe this is not circular.
"""
from __future__ import annotations

import pickle
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from beatrice_trainer.__main__ import PhoneExtractor  # noqa: E402
from phone_extractor_trainer.train import HubertTeacher  # noqa: E402

DATA = REPO / "analysis/data/ls_aligned.pkl"
PHONE_VARIANTS = {
    "jp_122_upstream":      REPO / "assets/pretrained/phone_extractor/jp_122_3000k.pt",
    "en_v3_anchored_80k":   REPO / "outputs/phone_extractor_en_v3/checkpoint_00080000.pt",
    "en_v4_softvc_40k":     REPO / "outputs/phone_extractor_en_v4/_probe_snapshot.pt",
    "en_clean_580k":        REPO / "assets/pretrained/phone_extractor/en_clean_580k.pt",
    "en_clean_200k":        REPO / "assets/pretrained/phone_extractor/en_clean_200k.pt",
    "en_nr_targetmix_300k": REPO / "assets/pretrained/phone_extractor/en_nr_targetmix02_300k.pt",
}
HARD_PAIRS = [("R", "L"), ("TH", "S"), ("V", "B"), ("DH", "D"),
              ("AE", "EH"), ("R", "ER"), ("S", "SH"), ("IH", "IY")]
DROP = {"spn", "sil", ""}
CENTRAL = 0.6      # use only the central 60% of each phoneme segment
MAX_FR_PER_SEG = 6
N_ABX = 30000
N_ABX_PAIR = 3000
SEED = 0


def collapse(lbl: str) -> str:
    return re.sub(r"\d+$", "", lbl)


def load_phone_extractor(path: Path, device):
    m = PhoneExtractor().to(device).eval().requires_grad_(False)
    ck = torch.load(path, map_location="cpu", weights_only=True)
    sd = ck["phone_extractor"] if "phone_extractor" in ck else ck
    m.load_state_dict(sd, strict=False)
    return m


@torch.inference_mode()
def extract(name, model, utts, device, is_hubert):
    """Return (feats [N,D] float32, labels [N] int, utt_ids [N] int)."""
    feats, labels, uids = [], [], []
    for uid, u in enumerate(utts):
        wav = torch.from_numpy(u["wav"]).to(device).float()
        dur = wav.numel() / 16000.0
        if is_hubert:
            f = model(wav[None])         # [1,T,768]
        else:
            f = model.units(wav[None, None])  # [1,1,T] -> [1,T,128]
        f = f.squeeze(0)  # [T,D]
        T = f.size(0)
        fps = T / dur
        for lbl, s, e in u["phonemes"]:
            c = collapse(lbl)
            if c in DROP:
                continue
            mid = 0.5 * (s + e)
            half = 0.5 * CENTRAL * (e - s)
            i0 = max(0, int((mid - half) * fps))
            i1 = min(T, int((mid + half) * fps) + 1)
            if i1 <= i0:
                i1 = min(T, i0 + 1)
            idx = np.linspace(i0, i1 - 1, num=min(MAX_FR_PER_SEG, i1 - i0)).astype(int)
            for j in idx:
                feats.append(f[j].float().cpu().numpy())
                labels.append(c)
                uids.append(uid)
    return np.stack(feats), np.array(labels), np.array(uids)


def abx_error(F_n, lab, rng, n_triplets, pair=None):
    """F_n: L2-normalised features [N,D] (torch, gpu). lab: int array.

    Vectorised: sample all (a1,a2,b) index triplets in numpy, then one batched
    cosine computation on GPU. Error if cos_dist(a1,a2) > cos_dist(a1,b).
    """
    classes = {}
    for i, l in enumerate(lab.tolist()):
        classes.setdefault(l, []).append(i)
    classes = {k: np.array(v) for k, v in classes.items() if len(v) >= 2}
    keys = list(classes.keys())
    if pair is not None:
        if pair[0] not in classes or pair[1] not in classes:
            return float("nan"), 0
        ca = np.full(n_triplets, pair[0])
        cb = np.full(n_triplets, pair[1])
    else:
        if len(keys) < 2:
            return float("nan"), 0
        ca = rng.choice(keys, size=n_triplets)
        cb = rng.choice(keys, size=n_triplets)
        bad = ca == cb
        while bad.any():
            cb[bad] = rng.choice(keys, size=int(bad.sum()))
            bad = ca == cb

    def pick(cats, second=False):
        out = np.empty(len(cats), dtype=np.int64)
        for c in np.unique(cats):
            m = np.where(cats == c)[0]
            out[m] = rng.choice(classes[c], size=len(m))
        return out

    a1 = pick(ca)
    a2 = pick(ca)
    # ensure a1 != a2 where possible
    same = a1 == a2
    if same.any():
        a2[same] = pick(ca[same])
    b = pick(cb)

    va = F_n[torch.as_tensor(a1, device=F_n.device)]
    va2 = F_n[torch.as_tensor(a2, device=F_n.device)]
    vb = F_n[torch.as_tensor(b, device=F_n.device)]
    d_same = 1 - (va * va2).sum(1)
    d_diff = 1 - (va * vb).sum(1)
    err = (d_same > d_diff).float().mean().item()
    return err, n_triplets


def pnmi(feats, lab_int, n_clusters, device, seed, iters=25):
    """Phone-Normalized Mutual Information (HuBERT/ContentVec metric).

    k-means cluster the (L2-normalised) frames, then PNMI = I(C;Phone)/H(Phone).
    Label-free of any trained probe -> robust correctness proxy (DC-Spin 2025
    recommends PNMI over ABX, which can be a misleading proxy). Higher = better;
    range [0,1] (1 = clusters perfectly determine phone identity).
    """
    g = torch.Generator(device=device).manual_seed(seed)
    X = F.normalize(torch.from_numpy(feats).to(device).float(), dim=1)
    N = X.size(0)
    # k-means++ would be nicer; random init + Lloyd is enough for a proxy.
    cen = X[torch.randperm(N, generator=g, device=device)[:n_clusters]].clone()
    assign = torch.zeros(N, dtype=torch.long, device=device)
    for _ in range(iters):
        # cosine == euclidean on the unit sphere; assign to nearest centroid.
        assign = torch.cdist(X, cen).argmin(1)
        for k in range(n_clusters):
            m = assign == k
            if m.any():
                cen[k] = F.normalize(X[m].mean(0), dim=0)
    c = assign.cpu().numpy()
    p = lab_int
    # contingency -> mutual information, normalised by phone entropy.
    nc, npp = int(c.max()) + 1, int(p.max()) + 1
    joint = np.zeros((nc, npp), dtype=np.float64)
    np.add.at(joint, (c, p), 1.0)
    joint /= joint.sum()
    pc = joint.sum(1, keepdims=True)
    pp = joint.sum(0, keepdims=True)
    nz = joint > 0
    mi = float((joint[nz] * np.log(joint[nz] / (pc @ pp)[nz])).sum())
    hp = float(-(pp[pp > 0] * np.log(pp[pp > 0])).sum())
    return mi / hp if hp > 0 else float("nan")


def linear_probe(feats, lab_int, uids, n_classes, device, seed):
    rng = np.random.default_rng(seed)
    uniq = np.unique(uids)
    rng.shuffle(uniq)
    n_test = max(1, int(0.3 * len(uniq)))
    test_u = set(uniq[:n_test].tolist())
    te = np.array([u in test_u for u in uids])
    tr = ~te
    X = torch.from_numpy(feats).to(device).float()
    y = torch.from_numpy(lab_int).to(device).long()
    mu = X[tr].mean(0, keepdim=True)
    sd = X[tr].std(0, keepdim=True) + 1e-6
    X = (X - mu) / sd
    clf = torch.nn.Linear(X.size(1), n_classes).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-2, weight_decay=1e-4)
    Xtr, ytr = X[tr], y[tr]
    for _ in range(400):
        opt.zero_grad()
        loss = F.cross_entropy(clf(Xtr), ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = clf(X[te]).argmax(1)
        acc = (pred == y[te]).float().mean().item()
    return acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    utts = pickle.load(open(DATA, "rb"))
    print(f"device: {device}\nutterances: {len(utts)}\n")
    rng = np.random.default_rng(SEED)

    candidates = [("HuBERT-L9 (upper bound)", HubertTeacher(layer_index=9).to(device).eval(), True)]
    for n, p in PHONE_VARIANTS.items():
        if p.is_file():
            candidates.append((n, load_phone_extractor(p, device), False))

    results = []
    for name, model, is_hubert in candidates:
        feats, labels, uids = extract(name, model, utts, device, is_hubert)
        classes = sorted(set(labels.tolist()))
        cidx = {c: i for i, c in enumerate(classes)}
        lab_int = np.array([cidx[c] for c in labels])
        Fn = F.normalize(torch.from_numpy(feats).to(device).float(), dim=1)

        abx, _ = abx_error(Fn, lab_int, np.random.default_rng(SEED), N_ABX)
        acc = linear_probe(feats, lab_int, uids, len(classes), device, SEED)
        pnmi_score = pnmi(feats, lab_int, n_clusters=min(100, 2 * len(classes)), device=device, seed=SEED)
        hard = {}
        for a, b in HARD_PAIRS:
            if a in cidx and b in cidx:
                e, _ = abx_error(Fn, lab_int, np.random.default_rng(SEED),
                                 N_ABX_PAIR, pair=(cidx[a], cidx[b]))
                hard[f"{a}/{b}"] = e
        results.append((name, abx, acc, pnmi_score, hard, len(feats)))

    print(f"{'extractor':<26}{'abx_err':>9}{'phone_acc':>11}{'PNMI':>8}{'frames':>9}")
    print("-" * 64)
    for name, abx, acc, pnmi_score, hard, nf in results:
        print(f"{name:<26}{abx:>9.4f}{acc:>11.4f}{pnmi_score:>8.4f}{nf:>9}")
    print("(abx_err: lower=better, chance=0.5 | phone_acc & PNMI: higher=better)\n")

    pair_keys = [f"{a}/{b}" for a, b in HARD_PAIRS]
    print("Hard-contrast ABX error (lower = better separation):")
    print(f"{'extractor':<26}" + "".join(f"{k:>9}" for k in pair_keys))
    print("-" * (26 + 9 * len(pair_keys)))
    for name, _, _, _, hard, _ in results:
        print(f"{name:<26}" + "".join(f"{hard.get(k, float('nan')):>9.3f}" for k in pair_keys))


if __name__ == "__main__":
    main()
