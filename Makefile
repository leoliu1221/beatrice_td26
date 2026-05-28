# Beatrice training pipeline.
#
# Layout:
#   inputs/<dataset>/...           raw long recordings (any audio format)
#   preprocessed/<dataset>/...     segmented clips fed to the trainer
#   outputs/<dataset>/             paraphernalia + checkpoints + config
#
# Usage:
#   make                      # preprocess + train, dataset=lol_data
#   make DATASET=other_name   # same, different dataset
#   make preprocess           # only segment audio
#   make train                # only run trainer (assumes preprocessed exists)
#   make train RESUME=1       # resume training from checkpoint_latest
#   make tensorboard          # serve TensorBoard for the dataset
#   make resume               # resume from outputs/<dataset>/checkpoint_latest
#   make clean                # wipe preprocessed/<dataset> and outputs/<dataset>
#
# PhoneExtractor sub-trainer (HuBERT distillation, English-aligned):
#   make phone-train PHONE_DATA=/path/to/english_audio
#                            # train new phone extractor from scratch
#   make phone-train PHONE_DATA=... PHONE_INIT=assets/pretrained/122_checkpoint_03000000.pt
#                            # warm-start from shipped Japanese-leaning ckpt
#   make phone-export        # export latest training ckpt -> Beatrice format
#   make phone-tensorboard   # tail phone-extractor TensorBoard
#
# PitchEstimator sub-trainer (pyworld-supervised, for better cross-gender conversion):
#   make pitch-train PITCH_DATA=/path/to/diverse_pitch_audio
#                            # train new pitch estimator from scratch
#   make pitch-train PITCH_DATA=... PITCH_INIT=assets/pretrained/104_3_checkpoint_00300000.pt
#                            # warm-start from shipped ckpt
#   make pitch-export        # export latest training ckpt -> Beatrice format
#   make pitch-tensorboard   # tail pitch-estimator TensorBoard


DATASET       ?= lol_data
RESUME        ?=
INPUTS_DIR    := inputs/$(DATASET)
PREPROC_DIR   := preprocessed/$(DATASET)
OUTPUT_DIR    := outputs/$(DATASET)
CONFIG        := $(OUTPUT_DIR)/config.json
DEFAULT_CFG   := assets/default_config.json

RESUME_FLAG   := $(if $(RESUME),-r,)

PY := uv run python

LIBRISPEECH_ROOT  := datasets/librispeech
LIBRISPEECH_SPLIT ?= train-clean-100
LIBRISPEECH_DIR   := $(LIBRISPEECH_ROOT)/LibriSpeech/$(LIBRISPEECH_SPLIT)

PHONE_DATA    ?= $(LIBRISPEECH_DIR)
PHONE_OUT     ?= outputs/phone_extractor_en
PHONE_INIT    ?= assets/pretrained/122_checkpoint_03000000.pt
PHONE_STEPS   ?= 200000
PHONE_BATCH   ?= 32
PHONE_WORKERS ?= 4
PHONE_EXPORT_OUT ?= assets/pretrained/phone_extractor_en.pt
PHONE_INIT_FLAG := $(if $(PHONE_INIT),--init-from $(PHONE_INIT),)
PHONE_RESUME_FLAG := $(if $(RESUME),--resume,)
# AUGMENT=1 enables noise-robust distillation: student sees augment_audio()-corrupted
# audio, teacher sees clean. Closes the train/test gap with Beatrice's main trainer.
PHONE_AUGMENT_FLAG := $(if $(AUGMENT),--augment,)

# PitchEstimator trainer variables
VCTK_ROOT     := datasets/vctk
VCTK_DIR      := $(VCTK_ROOT)/VCTK-Corpus-0.92/wav48_silence_trimmed
PITCH_DATA    ?= $(VCTK_DIR)
PITCH_OUT     ?= outputs/pitch_estimator_v2
PITCH_INIT    ?= assets/pretrained/104_3_checkpoint_00300000.pt
PITCH_STEPS   ?= 300000
PITCH_BATCH   ?= 256
PITCH_WORKERS ?= 8
PITCH_EXPORT_OUT ?= assets/pretrained/pitch_estimator_v2.pt
PITCH_INIT_FLAG := $(if $(PITCH_INIT),--init-from $(PITCH_INIT),)
PITCH_RESUME_FLAG := $(if $(RESUME),--resume,)
# AUGMENT=1 enables noise-robust training: student sees augmented audio,
# F0 label is still computed from clean audio.
PITCH_AUGMENT_FLAG := $(if $(AUGMENT),--augment,)

.PHONY: all preprocess config train resume tensorboard clean help \
        phone-data-download phone-train phone-train-en phone-export phone-tensorboard \
        pitch-data-download pitch-preprocess pitch-train pitch-train-vctk pitch-export pitch-tensorboard

all: train

help:
	@echo Targets: preprocess train resume tensorboard clean
	@echo Variables: DATASET=$(DATASET)

preprocess:
	$(PY) preprocess.py --dataset $(DATASET)

# Ensure $(OUTPUT_DIR)/config.json exists (copy default if missing).
config:
	$(PY) -c "import shutil, pathlib; d=pathlib.Path(r'$(OUTPUT_DIR)'); d.mkdir(parents=True, exist_ok=True); p=d/'config.json'; p.exists() or shutil.copy(r'$(DEFAULT_CFG)', p); print('config:', p)"

train: preprocess config
	$(PY) -m beatrice_trainer -d $(PREPROC_DIR) -o $(OUTPUT_DIR) -c $(CONFIG) $(RESUME_FLAG)

# `resume` deliberately skips `preprocess` (which wipes & regenerates clips)
# and `config` (already exists if a checkpoint exists).
resume:
	$(PY) -m beatrice_trainer -d $(PREPROC_DIR) -o $(OUTPUT_DIR) -c $(CONFIG) -r

tensorboard:
	$(PY) -m tensorboard.main --logdir $(OUTPUT_DIR)

clean:
	$(PY) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in [pathlib.Path(r'$(PREPROC_DIR)'), pathlib.Path(r'$(OUTPUT_DIR)')]]; print('cleaned:', r'$(PREPROC_DIR)', r'$(OUTPUT_DIR)')"

# --- PhoneExtractor sub-trainer ---------------------------------------------
#
# One-shot end-to-end:
#   make phone-train-en          # download LibriSpeech + warm-start + train + export
#
# Or step-by-step:
#   make phone-data-download     # ~6 GB, ~10-30 min depending on connection
#   make phone-train             # 200k steps, ~9 h on RTX 3080 Ti
#   make phone-train RESUME=1    # resume an interrupted run
#   make phone-export            # write Beatrice-format .pt
#   make phone-tensorboard       # live training curves

phone-data-download:
	$(PY) -c "import torchaudio, pathlib; root=pathlib.Path(r'$(LIBRISPEECH_ROOT)'); root.mkdir(parents=True, exist_ok=True); d=pathlib.Path(r'$(LIBRISPEECH_DIR)'); print('already downloaded:', d) if d.is_dir() else (torchaudio.datasets.LIBRISPEECH(str(root), url='$(LIBRISPEECH_SPLIT)', download=True), print('downloaded to', d))"

phone-train:
	@if [ -z "$(PHONE_DATA)" ]; then echo "PHONE_DATA=/path/to/english_audio is required"; exit 1; fi
	$(PY) -m phone_extractor_trainer.train \
		--data-dir $(PHONE_DATA) \
		--out-dir $(PHONE_OUT) \
		--steps $(PHONE_STEPS) \
		--batch-size $(PHONE_BATCH) \
		--num-workers $(PHONE_WORKERS) \
		$(PHONE_INIT_FLAG) \
		$(PHONE_RESUME_FLAG) \
		$(PHONE_AUGMENT_FLAG)

phone-train-en: phone-data-download
	$(MAKE) phone-train
	$(MAKE) phone-export

phone-export:
	$(PY) -m phone_extractor_trainer.export \
		$(PHONE_OUT)/checkpoint_latest.pt \
		$(PHONE_EXPORT_OUT)

phone-tensorboard:
	$(PY) -m tensorboard.main --logdir $(PHONE_OUT)

# --- PitchEstimator sub-trainer -----------------------------------------------
#
# For better female-to-male (and male-to-female) voice conversion, retrain the
# pitch estimator on diverse pitch data covering both male and female ranges.
#
# Recommended data sources:
#   - VocalSet (singing, wide pitch range)
#   - VCTK (multi-speaker, male+female)
#   - LibriTTS-R (audiobook, diverse speakers)
#   - Your own male+female speech recordings
#
# One-shot end-to-end with VCTK:
#   make pitch-train-vctk        # download VCTK + preprocess F0 + train + export
#
# Or step-by-step:
#   make pitch-data-download     # ~11 GB, ~15-30 min depending on connection
#   make pitch-preprocess        # pre-extract F0 (~10 min with 28 workers)
#   make pitch-train             # 300k steps, ~3-4 h on RTX 4070 Ti (with preprocessing)
#   make pitch-train RESUME=1    # resume an interrupted run
#   make pitch-export            # write Beatrice-format .pt
#   make pitch-tensorboard       # live training curves

pitch-data-download:
	$(PY) -c "import torchaudio, pathlib; root=pathlib.Path(r'$(VCTK_ROOT)'); root.mkdir(parents=True, exist_ok=True); d=pathlib.Path(r'$(VCTK_DIR)'); print('already downloaded:', d) if d.is_dir() else (torchaudio.datasets.VCTK_092(str(root), download=True), print('downloaded to', d))"

pitch-preprocess:
	@if [ ! -d "$(PITCH_DATA)" ]; then echo "PITCH_DATA directory not found: $(PITCH_DATA)"; exit 1; fi
	$(PY) -m pitch_estimator_trainer.preprocess \
		--data-dir $(PITCH_DATA) \
		--num-workers 28

pitch-train:
	@if [ -z "$(PITCH_DATA)" ]; then echo "PITCH_DATA=/path/to/diverse_pitch_audio is required"; exit 1; fi
	$(PY) -m pitch_estimator_trainer.train \
		--data-dir $(PITCH_DATA) \
		--out-dir $(PITCH_OUT) \
		--steps $(PITCH_STEPS) \
		--batch-size $(PITCH_BATCH) \
		--num-workers $(PITCH_WORKERS) \
		$(PITCH_INIT_FLAG) \
		$(PITCH_RESUME_FLAG) \
		$(PITCH_AUGMENT_FLAG)

pitch-train-vctk: pitch-data-download pitch-preprocess
	$(MAKE) pitch-train
	$(MAKE) pitch-export

pitch-export:
	$(PY) -m pitch_estimator_trainer.export \
		$(PITCH_OUT)/checkpoint_latest.pt \
		$(PITCH_EXPORT_OUT)

pitch-tensorboard:
	$(PY) -m tensorboard.main --logdir $(PITCH_OUT)
