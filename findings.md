# Code Review Findings & Recommendations

A comprehensive analysis of `beatrice_trainer`, `phone_extractor_trainer`, and `pitch_estimator_trainer`.

---

## Executive Summary

The codebase is well-structured with clean separation between the main trainer and sub-trainers. However, there are several opportunities for improvement in code organization, performance, maintainability, and robustness.

**Priority Levels:**
- 🔴 **High** — Significant impact on reliability or performance
- 🟡 **Medium** — Improves maintainability or developer experience
- 🟢 **Low** — Nice-to-have improvements

---

## 1. Architecture & Code Organization

### 1.1 🔴 Monolithic Main File

**Issue:** `beatrice_trainer/__main__.py` is **4,538 lines** in a single file, containing:
- Model definitions (PhoneExtractor, PitchEstimator, ConverterNetwork, Discriminator)
- Training loop
- Data loading
- Augmentation
- Export logic
- Utility functions

**Recommendation:** Split into modules:
```
beatrice_trainer/
├── __init__.py
├── __main__.py          # Entry point only
├── models/
│   ├── phone_extractor.py
│   ├── pitch_estimator.py
│   ├── converter.py
│   ├── discriminator.py
│   └── layers.py        # ConvNeXtBlock, CrossAttention, etc.
├── data/
│   ├── dataset.py
│   └── augmentation.py
├── training/
│   ├── trainer.py
│   ├── losses.py
│   └── schedulers.py
└── utils/
    ├── export.py
    └── checkpoint.py
```

**Benefit:** Easier navigation, testing, and maintenance.

---

### 1.2 🟡 Code Duplication Between Sub-Trainers

**Issue:** `phone_extractor_trainer` and `pitch_estimator_trainer` share significant code:
- `discover_audio_files()` — identical in both `data.py`
- `_get_resampler()` — identical caching logic
- `cosine_warmup_lr()` — identical in both `train.py`
- Training loop structure — nearly identical

**Recommendation:** Create a shared utilities module:
```python
# beatrice_trainer/common/
├── audio.py      # discover_audio_files, _get_resampler
├── schedule.py   # cosine_warmup_lr
└── training.py   # BaseTrainer class with common loop logic
```

---

### 1.3 🟡 Import Hack for Sub-Trainers

**Issue:** Both sub-trainers use a sys.path hack to import from `beatrice_trainer`:
```python
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
```

**Recommendation:** Use proper package structure with `pyproject.toml` entry points or relative imports. The models should be importable as:
```python
from beatrice_trainer.models import PhoneExtractor, PitchEstimator
```

---

## 2. Performance Optimizations

### 2.1 🔴 CPU Bottleneck in Main Trainer

**Issue:** The main `WavDataset` performs heavy CPU augmentation (reverb convolution, noise mixing, formant shift via resampling) in `__getitem__`. This causes GPU underutilization (~50-60% on RTX 3080 Ti).

**Current flow:**
```
DataLoader workers → [CPU: load + augment] → GPU: train
                     ↑ bottleneck
```

**Recommendations:**
1. **Pre-compute augmentations** — Similar to F0 pre-extraction in pitch trainer
2. **GPU-based augmentation** — Move reverb/noise to GPU using `torchaudio.functional`
3. **Reduce augmentation probability** — `augmentation_reverb_probability: 0.2` (documented but not default)

---

### 2.2 🟡 Inefficient Resampler Caching

**Issue:** `get_resampler()` in main trainer creates new resamplers per (src, dst, device) tuple but doesn't limit cache size:
```python
_RESAMPLERS: dict[tuple[int, int, torch.device], torchaudio.transforms.Resample] = {}
```

**Risk:** Memory leak if many unique sample rates are encountered.

**Recommendation:** Use `functools.lru_cache` with maxsize or explicit cache eviction.

---

### 2.3 🟢 Unnecessary Full File Reads

**Issue:** In `phone_extractor_trainer/data.py`:
```python
wav, sr = torchaudio.load(str(path), backend="soundfile")
# Then random crop
```

For very long files, this reads the entire file before cropping.

**Recommendation:** Use `frame_offset` and `num_frames` parameters:
```python
# Get file info first
info = torchaudio.info(path)
start = random.randint(0, max(0, info.num_frames - wav_length))
wav, sr = torchaudio.load(path, frame_offset=start, num_frames=wav_length)
```

---

## 3. Robustness & Error Handling

### 3.1 🔴 Silent Failures in Data Loading

**Issue:** Both sub-trainers retry 8 times on failure but only report the last error:
```python
for _ in range(8):
    try:
        return self._load_random_crop(path)
    except Exception as e:
        last_err = e
        continue
raise RuntimeError(f"could not load any audio after 8 tries; last error: {last_err}")
```

**Problems:**
- No logging of which files failed
- No tracking of failure rate
- Could silently skip problematic files forever

**Recommendation:**
```python
import logging
logger = logging.getLogger(__name__)

def __getitem__(self, idx):
    for attempt in range(8):
        path = self._rng.choice(self.files)
        try:
            return self._load_random_crop(path)
        except Exception as e:
            logger.warning(f"Failed to load {path} (attempt {attempt+1}): {e}")
    raise RuntimeError(...)
```

---

### 3.2 🟡 No Validation Set

**Issue:** Sub-trainers have no validation split — they only report training metrics.

**Recommendation:** Add `--val-split` argument (e.g., 5%) to monitor overfitting:
```python
files = discover_audio_files(args.data_dir)
random.shuffle(files)
split = int(len(files) * 0.95)
train_files, val_files = files[:split], files[split:]
```

---

### 3.3 🟡 Missing Input Validation

**Issue:** `PitchDataset` doesn't validate that `wav_length % hop_length == 0` until runtime:
```python
if wav_length % hop_length != 0:
    raise ValueError(...)
```

**Recommendation:** Add this check in CLI argument parsing to fail fast.

---

## 4. Configuration & Defaults

### 4.1 🔴 Inconsistent Default Batch Sizes

**Issue:** Different defaults across trainers:
| Trainer | Default batch_size | Optimal (tested) |
|---------|-------------------|------------------|
| beatrice_trainer | 8 | 8 |
| phone_extractor_trainer | 32 | 32 |
| pitch_estimator_trainer | 32 | **256** |

The Makefile was updated to use 256 for pitch trainer, but the code default is still 32.

**Recommendation:** Update `pitch_estimator_trainer/train.py` default:
```python
p.add_argument("--batch-size", type=int, default=256)
```

---

### 4.2 🟡 Hardcoded Magic Numbers

**Issue:** Many magic numbers scattered throughout:
```python
# beatrice_trainer/__main__.py
wav_length = 4 * 24000  # 4s — why 4s?
segment_length = 100    # 1s — why 100 frames?
floor_noise_level = 1e-3  # why this value?
```

**Recommendation:** Add comments explaining the rationale, or move to a constants file with documentation.

---

### 4.3 🟢 Config File Validation

**Issue:** Unknown keys in config.json are silently ignored with a warning:
```python
for key in list(h.keys()):
    if key not in dict_default_hparams:
        warnings.warn(f"`{key}` specified in the config file will be ignored.")
        del h[key]
```

**Recommendation:** Option to fail on unknown keys (strict mode) to catch typos:
```python
parser.add_argument("--strict-config", action="store_true")
```

---

## 5. Testing & Quality Assurance

### 5.1 🔴 No Unit Tests

**Issue:** Zero test files in the repository.

**Recommendation:** Add tests for critical components:
```
tests/
├── test_models.py       # Model forward pass shapes
├── test_data.py         # Dataset loading
├── test_f0_extraction.py # F0 to pitch bin conversion
└── test_export.py       # Checkpoint export/load round-trip
```

Example test:
```python
def test_f0_to_pitch_bin():
    # 55 Hz (A1) should map to bin 1
    assert f0_to_pitch_bin(np.array([55.0]))[0] == 1
    # 110 Hz (A2) should map to bin 1 + 96 = 97
    assert f0_to_pitch_bin(np.array([110.0]))[0] == 97
    # Unvoiced (0 Hz) should map to bin 0
    assert f0_to_pitch_bin(np.array([0.0]))[0] == 0
```

---

### 5.2 🟡 No CI/CD Pipeline

**Issue:** No GitHub Actions or similar for automated testing.

**Recommendation:** Add `.github/workflows/test.yml`:
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v1
      - run: uv sync
      - run: uv run pytest tests/
```

---

## 6. Documentation

### 6.1 🟡 Missing Docstrings

**Issue:** Many functions lack docstrings, especially in `beatrice_trainer/__main__.py`:
```python
def dump_params(params: torch.Tensor, f: BinaryIO):
    if params is None:
        return
    # ... no docstring explaining what this does
```

**Recommendation:** Add docstrings to all public functions explaining:
- Purpose
- Parameters
- Return values
- Side effects

---

### 6.2 🟢 Japanese Comments

**Issue:** Many comments are in Japanese, which limits accessibility:
```python
# 発声部分をランダムに 1 箇所選ぶ
# スライス区間が含まれるように、ランダムに wav_length の長さを切り出す
```

**Recommendation:** Add English translations (keep Japanese for original maintainers):
```python
# 発声部分をランダムに 1 箇所選ぶ
# Select a random voiced position
```

---

## 7. Specific Bug Risks

### 7.1 🔴 Potential Race Condition in Checkpoint Saving

**Issue:** Checkpoint saving uses atomic rename but the copy to `checkpoint_latest.pt.gz` is not atomic:
```python
torch.save(ckpt, tmp)
os.replace(tmp, ckpt_latest)  # atomic
# ...
shutil.copy(checkpoint_file_save, out_dir / "checkpoint_latest.pt.gz")  # NOT atomic
```

If the process crashes during `shutil.copy`, `checkpoint_latest.pt.gz` could be corrupted.

**Recommendation:** Use atomic copy pattern:
```python
tmp_latest = out_dir / "checkpoint_latest.pt.gz.tmp"
shutil.copy(checkpoint_file_save, tmp_latest)
os.replace(tmp_latest, out_dir / "checkpoint_latest.pt.gz")
```

---

### 7.2 🟡 F0 Pre-extraction Mismatch Risk

**Issue:** If audio files are modified after F0 pre-extraction, the `.f0.npy` files become stale.

**Recommendation:** Store audio file hash or mtime in the `.f0.npy` metadata:
```python
np.savez(f0_path, f0=f0, audio_mtime=path.stat().st_mtime)
```

Or add a `--force-recompute` flag to the preprocess script.

---

### 7.3 🟢 Seed Not Propagated to Workers

**Issue:** In `PitchDataset`, the seed is used for `random.Random(seed)`, but DataLoader workers fork the RNG state, potentially causing duplicate samples across workers.

**Recommendation:** Use worker_init_fn to reseed:
```python
def worker_init_fn(worker_id):
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)

loader = DataLoader(..., worker_init_fn=worker_init_fn)
```

---

## 8. Feature Suggestions

### 8.1 🟡 Early Stopping

**Issue:** Training runs for fixed steps regardless of convergence.

**Recommendation:** Add early stopping based on validation loss plateau:
```python
p.add_argument("--early-stop-patience", type=int, default=0,
               help="Stop if val loss doesn't improve for N evals (0=disabled)")
```

---

### 8.2 🟡 Learning Rate Finder

**Issue:** LR is manually tuned.

**Recommendation:** Add LR range test utility:
```bash
uv run python -m pitch_estimator_trainer.lr_finder --data-dir ...
```

---

### 8.3 🟢 Checkpoint Pruning

**Issue:** Old checkpoints accumulate and consume disk space.

**Recommendation:** Add `--keep-last-n` argument to auto-delete old checkpoints.

---

## 9. Summary Table

| Category | High 🔴 | Medium 🟡 | Low 🟢 |
|----------|---------|-----------|--------|
| Architecture | 1 | 2 | 0 |
| Performance | 1 | 1 | 1 |
| Robustness | 1 | 2 | 0 |
| Configuration | 1 | 1 | 1 |
| Testing | 1 | 1 | 0 |
| Documentation | 0 | 1 | 1 |
| Bug Risks | 1 | 1 | 1 |
| Features | 0 | 2 | 1 |
| **Total** | **6** | **11** | **5** |

---

## 10. Recommended Action Plan

### Phase 1: Quick Wins (1-2 days)
1. ✅ Fix batch size default in pitch trainer
2. Add atomic checkpoint copy
3. Add logging for data loading failures
4. Update Japanese comments with English translations

### Phase 2: Robustness (1 week)
1. Add unit tests for critical functions
2. Add validation split to sub-trainers
3. Add F0 staleness detection
4. Add worker seed propagation

### Phase 3: Refactoring (2-3 weeks)
1. Split `__main__.py` into modules
2. Create shared utilities for sub-trainers
3. Add CI/CD pipeline
4. Add early stopping and LR finder

### Phase 4: Performance (1-2 weeks)
1. Investigate GPU-based augmentation
2. Implement pre-computed augmentation caching
3. Optimize file reading with frame_offset

---

## Appendix: Code Metrics

| File | Lines | Functions | Classes |
|------|-------|-----------|---------|
| `beatrice_trainer/__main__.py` | 4,538 | ~80 | ~25 |
| `phone_extractor_trainer/train.py` | 314 | 5 | 1 |
| `phone_extractor_trainer/data.py` | 108 | 3 | 1 |
| `pitch_estimator_trainer/train.py` | 310 | 5 | 0 |
| `pitch_estimator_trainer/data.py` | 235 | 5 | 1 |
| `pitch_estimator_trainer/preprocess.py` | 112 | 2 | 0 |

**Total: ~5,617 lines of Python**
