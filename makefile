.DEFAULT_GOAL := help

PYTHON_VERSION := 3.11
VENV := .venv
PYTHON := $(VENV)/bin/python
KERNEL_NAME := kaggle-detectron2-py311
KERNEL_DISPLAY := Python 3.11 (.venv: kaggle-detectron2)

CORE_DEPS := 'setuptools<81' wheel 'ipykernel<7' nbconvert kagglehub numpy pandas opencv-python tqdm matplotlib torch torchvision pycocotools
DETECTRON2 := git+https://github.com/facebookresearch/detectron2.git

.PHONY: help setup venv deps kernel check run clean

help:
	@echo "Targets:"
	@echo "  make setup   Create .venv, install dependencies, register Jupyter kernel"
	@echo "  make venv    Create the Python $(PYTHON_VERSION) virtual environment"
	@echo "  make deps    Install Python dependencies into .venv"
	@echo "  make kernel  Register the .venv as a Jupyter kernel"
	@echo "  make check   Compile the Python script without running training"
	@echo "  make run     Run the de-poisoning workflow"
	@echo "  make clean   Remove generated outputs and caches"

setup: venv deps kernel

venv:
	uv python install $(PYTHON_VERSION)
	uv venv --python $(PYTHON_VERSION) $(VENV)

deps: $(PYTHON)
	uv pip install --python $(PYTHON) $(CORE_DEPS)
	uv pip install --python $(PYTHON) --no-build-isolation '$(DETECTRON2)'

kernel: $(PYTHON)
	$(PYTHON) -m ipykernel install --user --name $(KERNEL_NAME) --display-name "$(KERNEL_DISPLAY)"

check: $(PYTHON)
	$(PYTHON) -m py_compile de_poisoning_pruning_finetuning.py

run: $(PYTHON)
	$(PYTHON) de_poisoning_pruning_finetuning.py

clean:
	rm -rf __pycache__ .ipynb_checkpoints unlearned submission.csv

$(PYTHON):
	$(MAKE) venv
