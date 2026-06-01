# Pretrained model registry

Each module lives in its own subdirectory. Variants are **immutable** — never
overwrite an existing file. To promote a new variant to the default, retarget
the `current.pt` symlink and update `manifest.json`.

```
phone_extractor/
  jp_122_3000k.pt              upstream Japanese warm-start (do not delete)
  en_clean_200k.pt             v1 best target-domain (deprecated)
  en_clean_580k.pt             v1 production (deprecated)
  en_nr_targetmix02_300k.pt    v2 noise-robust + 20% target-mix (current)
  current.pt -> en_nr_targetmix02_300k.pt
  <tag>.meta.json              per-variant recipe + eval metadata

pitch_estimator/
  jp_104_3_300k.pt             upstream Japanese warm-start (do not delete)
  vctk_clean_300k.pt           clean distillation (deprecated)
  vctk_nr_300k.pt              noise-robust distillation (current)
  current.pt -> vctk_nr_300k.pt
  <tag>.meta.json

vocoder/
  libritts_r_200_2750k.pt.gz   upstream LibriTTS-R vocoder (current)
  current.pt.gz -> libritts_r_200_2750k.pt.gz

manifest.json                  registry; lists all variants + current
```

## Naming convention

`<dataset>_<recipe-tag>_<steps>.pt`

- `dataset`: `en` (LibriSpeech), `vctk`, `jp` (upstream Japanese), `libritts_r`, ...
- `recipe-tag`:
  - `clean` — no augmentation
  - `nr` — noise-robust (augmentation enabled)
  - `nr_targetmix<frac>` — adds target-domain mixing (`targetmix02` = ratio 0.2)
- `steps`: rounded, in `k`/`M` (`300k`, `3000k`).

## Config references

Configs (`assets/default_config.json`, `assets/configs/*.json`) should point at
`current.pt` for the default behavior:

```json
"phone_extractor_file": "assets/pretrained/phone_extractor/current.pt",
"pitch_estimator_file": "assets/pretrained/pitch_estimator/current.pt",
"pretrained_file":      "assets/pretrained/vocoder/current.pt.gz"
```

For an A/B run that needs to pin a specific variant (so the current symlink
moving doesn't disturb history), point at the variant file directly:

```json
"phone_extractor_file": "assets/pretrained/phone_extractor/en_nr_targetmix02_300k.pt"
```

## Promoting a new variant

1. Export new training run as `<module>/<dataset>_<tag>_<steps>k.pt` (never overwrite).
2. Write a sibling `<tag>.meta.json` with recipe + eval scores.
3. Add an entry under the right module in `manifest.json`.
4. After validation, retarget the symlink: `ln -sfn <new>.pt <module>/current.pt`.
5. Mark the previous current as `"deprecated": true` in `manifest.json`.
