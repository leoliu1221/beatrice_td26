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
#   make tensorboard          # serve TensorBoard for the dataset
#   make resume               # continue training from checkpoint_latest
#   make clean                # wipe preprocessed/<dataset> and outputs/<dataset>

DATASET       ?= lol_data
INPUTS_DIR    := inputs/$(DATASET)
PREPROC_DIR   := preprocessed/$(DATASET)
OUTPUT_DIR    := outputs/$(DATASET)
CONFIG        := $(OUTPUT_DIR)/config.json
DEFAULT_CFG   := assets/default_config.json

PY := uv run python

.PHONY: all preprocess config train resume tensorboard clean help

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
	$(PY) -m beatrice_trainer -d $(PREPROC_DIR) -o $(OUTPUT_DIR) -c $(CONFIG)

resume: config
	$(PY) -m beatrice_trainer -d $(PREPROC_DIR) -o $(OUTPUT_DIR) -c $(CONFIG) -r

tensorboard:
	$(PY) -m tensorboard.main --logdir $(OUTPUT_DIR)

clean:
	$(PY) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in [pathlib.Path(r'$(PREPROC_DIR)'), pathlib.Path(r'$(OUTPUT_DIR)')]]; print('cleaned:', r'$(PREPROC_DIR)', r'$(OUTPUT_DIR)')"
