# Kaggle Notebook Setup

This workspace is configured to run the de-poisoning workflow in VS Code with the local Python environment at `.venv`.

## VS Code

1. Open this folder in VS Code.
2. Install the Microsoft Python extension if prompted.
3. Open `de_poisoning_pruning_finetuning.py`.
4. Run the `Run de-poisoning script` launch configuration.

You can also run it from a terminal:

```bash
make run
```

## Recreate the Environment

```bash
make setup
```

Detectron2 needs `--no-build-isolation` because its setup script imports `torch` while building.
The script downloads the competition data with `kagglehub.competition_download(...)` when it runs.

Useful targets:

```bash
make help
make check
make run
make clean
```
