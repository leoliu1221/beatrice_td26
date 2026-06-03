"""Tier-1 objective quality of CONVERTED English audio (the promotion gate).

Scores clips produced by convert_eval.py on:
  - PER   : phoneme error rate. Phoneme recognizer (wav2vec2-espeak, IPA) run on
            the SOURCE (reference) and the CONVERTED (hypothesis); edit distance
            over phoneme tokens. Directly catches accent / phonetic distortion
            introduced by conversion (R/L, TH, V ...). LOWER = better.
  - WER   : Whisper (en) on source vs converted; word edit distance. Content /
            intelligibility survival. LOWER = better.
  - UTMOS : naturalness of converted audio (same metric the trainer uses).
            HIGHER = better.
  - SPK   : speaker similarity = cosine(WavLM-SV embedding of converted,
            centroid of REAL target-speaker clips). HIGHER = better timbre match.
            We also report SPK_src = sim(converted, source) which should be LOW.

All recognizers are run identically on every system, so cross-extractor
comparison is fair even though the recognizers have their own biases.

Usage:
    uv run python analysis/score_converted.py --tags path_a_5k,jp122_5k --target sion
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
CONVERTED = REPO / "analysis/converted"
TARGET_CLIPS = REPO / "preprocessed/new_lol_data_df"

WHISPER_MODEL = "openai/whisper-base.en"
PHONEME_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"
SPK_MODEL = "microsoft/wavlm-base-plus-sv"
N_TARGET_REF = 20  # real target clips to build the speaker centroid


def edit_distance(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[n]


def load_wav_16k(path: Path) -> torch.Tensor:
    w, sr = sf.read(str(path), dtype="float32")
    if w.ndim > 1:
        w = w.mean(1)
    t = torch.from_numpy(w).float()
    if sr != 16000:
        t = AF.resample(t, sr, 16000)
    return t


class Scorers:
    def __init__(self, device):
        self.device = device
        import json as _json
        from huggingface_hub import hf_hub_download
        from transformers import (
            AutoFeatureExtractor, AutoModelForAudioXVector,
            Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC, pipeline,
        )
        print("loading UTMOS ...")
        self.utmos = torch.hub.load("tarepan/SpeechMOS:v1.0.0", "utmos22_strong",
                                    trust_repo=True).eval().to(device)
        print("loading Whisper (WER) ...")
        self.asr = pipeline("automatic-speech-recognition", model=WHISPER_MODEL,
                            device=0 if device.type == "cuda" else -1)
        print("loading wav2vec2-espeak (PER) ...")
        # Bypass the phoneme *tokenizer* (it requires the espeak/phonemizer
        # backend only for text->phoneme). For audio->phoneme we just need the
        # feature extractor + raw vocab and a manual CTC greedy decode.
        self.ph_fe = Wav2Vec2FeatureExtractor.from_pretrained(PHONEME_MODEL)
        self.ph_model = Wav2Vec2ForCTC.from_pretrained(PHONEME_MODEL).eval().to(device)
        vocab = _json.loads(Path(hf_hub_download(PHONEME_MODEL, "vocab.json")).read_text())
        self.id2tok = {i: t for t, i in vocab.items()}
        self.ph_pad_id = self.ph_model.config.pad_token_id
        print("loading WavLM-SV (speaker) ...")
        self.spk_fe = AutoFeatureExtractor.from_pretrained(SPK_MODEL)
        self.spk_model = AutoModelForAudioXVector.from_pretrained(SPK_MODEL).eval().to(device)

    @torch.inference_mode()
    def utmos_score(self, wav16k: torch.Tensor) -> float:
        return float(self.utmos(wav16k[None].to(self.device), sr=16000).item())

    @torch.inference_mode()
    def phonemes(self, wav16k: torch.Tensor) -> list[str]:
        iv = self.ph_fe(wav16k.numpy(), sampling_rate=16000,
                        return_tensors="pt").input_values.to(self.device)
        ids = self.ph_model(iv).logits.argmax(-1)[0].tolist()
        out, prev = [], None
        for i in ids:  # CTC greedy: collapse repeats, drop blank/special
            if i != prev and i != self.ph_pad_id:
                tok = self.id2tok.get(i, "")
                if tok and not (tok.startswith("<") and tok.endswith(">")):
                    out.append(tok)
            prev = i
        return out

    def words(self, wav16k: torch.Tensor) -> list[str]:
        out = self.asr(wav16k.numpy())
        import re
        return re.sub(r"[^a-z' ]", " ", out["text"].lower()).split()

    @torch.inference_mode()
    def spk_embed(self, wav16k: torch.Tensor) -> torch.Tensor:
        iv = self.spk_fe(wav16k.numpy(), sampling_rate=16000,
                         return_tensors="pt").input_values.to(self.device)
        emb = self.spk_model(iv).embeddings.squeeze(0)
        return torch.nn.functional.normalize(emb, dim=-1)


def target_centroid(sc: Scorers, target: str) -> torch.Tensor:
    clips = sorted((TARGET_CLIPS / target).glob("*.wav"))[:N_TARGET_REF]
    embs = [sc.spk_embed(load_wav_16k(p)) for p in clips]
    c = torch.stack(embs).mean(0)
    return torch.nn.functional.normalize(c, dim=-1)


def score_tag(sc: Scorers, tag: str, target: str, centroid: torch.Tensor) -> dict:
    root = CONVERTED / tag
    manifest = json.loads((root / "manifest.json").read_text())
    rows = {"per": [], "wer": [], "utmos": [], "spk": [], "spk_src": []}
    for it in manifest["items"]:
        cid = it["id"]
        src = load_wav_16k(root / "source" / f"{cid}.wav")
        conv = load_wav_16k(root / target / f"{cid}.wav")

        ref_ph, hyp_ph = sc.phonemes(src), sc.phonemes(conv)
        if ref_ph:
            rows["per"].append(edit_distance(ref_ph, hyp_ph) / len(ref_ph))
        ref_w, hyp_w = sc.words(src), sc.words(conv)
        if ref_w:
            rows["wer"].append(edit_distance(ref_w, hyp_w) / len(ref_w))
        rows["utmos"].append(sc.utmos_score(conv))
        ce = sc.spk_embed(conv)
        rows["spk"].append(float(ce @ centroid))
        rows["spk_src"].append(float(ce @ torch.nn.functional.normalize(sc.spk_embed(src), dim=-1)))
    return {k: float(np.mean(v)) for k, v in rows.items() if v} | {"n": len(manifest["items"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True, help="comma-separated convert_eval tags")
    ap.add_argument("--target", default="sion")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sc = Scorers(device)
    centroid = target_centroid(sc, args.target)

    results = {}
    for tag in [t.strip() for t in args.tags.split(",")]:
        print(f"\nscoring {tag} (target={args.target}) ...")
        results[tag] = score_tag(sc, tag, args.target, centroid)

    out = REPO / "analysis" / f"converted_quality_{args.target}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n{'tag':<18}{'PER↓':>8}{'WER↓':>8}{'UTMOS↑':>9}{'SPK↑':>8}{'SPK_src↓':>10}{'n':>5}")
    print("-" * 66)
    for tag, r in results.items():
        print(f"{tag:<18}{r.get('per',float('nan')):>8.3f}{r.get('wer',float('nan')):>8.3f}"
              f"{r.get('utmos',float('nan')):>9.3f}{r.get('spk',float('nan')):>8.3f}"
              f"{r.get('spk_src',float('nan')):>10.3f}{r.get('n',0):>5}")
    print(f"\nwrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
