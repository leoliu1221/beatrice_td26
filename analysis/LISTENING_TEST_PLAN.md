# Reusable Model-Comparison Listening-Test Site — Plan

**Status:** plan only (lower priority than the PhoneExtractor v3 retrain).
**Motivation:** the one-off `analysis/listening_test.html` (blind A/B of `en_clean_60k` vs `jp122_60k`) worked and produced the deciding verdict (lesson 2.9 / Experiment 5). We want to generalize it into a tool we can reuse for *any* future hearing test (extractor A/B, pitch-estimator A/B, denoiser A/B, checkpoint sweeps, MOS, MUSHRA) without hand-editing HTML each time.

---

## 1. Design goals
- **Config-driven, zero-code per test.** A new test = drop audio in a folder + write one small JSON manifest. No HTML edits.
- **Multiple test *modes*** from the same UI: blind **A/B preference**, **ABX** (which of A/B matches reference X), **MOS** (1–5 absolute naturalness), **MUSHRA** (multi-system slider vs hidden ref + anchor).
- **Rigorous by default:** blind labels, randomized presentation order, randomized A/B side assignment, optional repeated trials for intra-rater consistency.
- **Results you can analyze:** export JSON/CSV; show live tallies; compute win-rate + a simple significance test (binomial / sign test for A/B).
- **Reuses existing pipeline.** `analysis/convert_eval.py` already produces `analysis/converted/<tag>/{source,<target>}/*.wav` + `manifest.json`. The site consumes those directly — a "system" is just a `tag`.
- **Local-first**, no build step. Served by `python -m http.server` from `analysis/` (browsers block `file://` audio).

## 2. Architecture
- **`analysis/listen/` (static site):**
  - `index.html` — test picker (lists available test configs).
  - `app.js` — loads a test config, renders the right mode component, manages blinding/randomization, persists votes to `localStorage`, exports results.
  - `styles.css` — shared dark theme (reuse the current page's look).
  - Pure vanilla JS + ES modules (no framework, no bundler) so it stays trivially runnable. *(If it grows, migrate to Vite + React + shadcn/ui, but not before it's needed.)*
- **Test config — `analysis/listen/tests/<test_id>.json`:**
  ```jsonc
  {
    "id": "ph_v3_vs_jp122_sion",
    "mode": "ab",                       // "ab" | "abx" | "mos" | "mushra"
    "title": "PhoneExtractor v3 vs jp122 (target=sion)",
    "criteria": ["overall", "pronunciation", "naturalness", "timbre"],
    "systems": {                         // logical name -> converted tag (dir under analysis/converted/)
      "A": {"tag": "v3_60k",   "target": "sion"},
      "B": {"tag": "jp122_60k", "target": "sion"}
    },
    "reference": {"tag": "v3_60k", "track": "source"},   // for ABX/MUSHRA hidden-ref
    "items_from": "manifest",            // pull item ids from a tag's manifest.json
    "n_items": 40,
    "blind": true, "randomize_order": true, "randomize_sides": true,
    "repeats": 1
  }
  ```
- **A "results" artifact** written on export: `analysis/listen/results/<test_id>_<rater>.json` (votes per item+criterion + the *revealed* system map so it can be scored offline). Optionally a tiny `score_listen.py` to aggregate win-rates + sign-test p-values across rater files.

## 3. Modes (MVP → later)
- **MVP:** `ab` (generalize the current page) + `mos`. These cover 90% of our needs (extractor A/B + absolute naturalness).
- **Phase 2:** `abx` (true ABX with hidden reference; complements the objective ABX probe with a human one) and `mushra` (multi-checkpoint sweeps, e.g. v3 @ 20k/40k/60k/80k vs jp122 with hidden anchor).

## 4. Rigor details (carry over + add)
- Hidden, randomized A/B side per (item, rater) stored stably so reloads don't re-shuffle (current page already does this).
- **Loudness-normalize** all clips to a common LUFS at serve time (or pre-normalize in `convert_eval.py`) so level doesn't bias preference — *new vs current page*.
- Force listen-before-vote (disable vote buttons until both A and B have been played) — optional toggle.
- Repeated/duplicated trials to estimate intra-rater reliability.
- "Reveal labels" only after all items voted.

## 5. Backend (only if needed later)
- MVP is fully static (localStorage + manual export). If we want multi-rater collection without passing files around, add a ~30-line `flask`/`fastapi` endpoint that accepts `POST /results` and appends JSON to `analysis/listen/results/`. Not needed for solo evaluation.

## 6. Build order (when picked up)
1. Refactor current `listening_test.html` → `analysis/listen/{index.html,app.js,styles.css}` with the `ab` mode reading a test config (no behavior change, just config-driven).
2. Add a tiny generator: `analysis/make_listen_test.py --mode ab --systems v3_60k,jp122_60k --target sion --out analysis/listen/tests/<id>.json` (so creating a test is one command).
3. Add `mos` mode + `score_listen.py` aggregator (win-rate + sign test).
4. Loudness normalization at serve/convert time.
5. (Later) `abx`, `mushra`, optional collection backend.

## 7. Immediate reuse for v3
As soon as the v3 pilot passes the intrinsic panel, run `convert_eval.py ... --tag v3_60k` and create an `ab` test `v3_60k` vs `jp122_60k` — this is the listening half of the Tier-1 gate, and the first consumer of the reusable site.
