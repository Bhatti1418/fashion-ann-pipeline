# Fashion-MNIST ANN Pipeline

An end-to-end, reproducible Fashion-MNIST classification project using a fully connected TensorFlow/Keras ANN, Git, and DVC. The workflow is split into four command-line stages: download the dataset, preprocess it, train the model, and evaluate it.

## Requirements

- Python 3.9 or newer (Python 3.11 is recommended for TensorFlow compatibility)
- Git
- DVC with the Google Drive extra
- A Google Drive folder and a Google OAuth Desktop application for remote access

## Local setup (Windows PowerShell)

Create and activate a virtual environment, then install the project dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install "tensorflow>=2.15,<3" "dvc[gdrive]" pyyaml scikit-learn matplotlib
```

Run each stage from the repository root with `python src/prepare.py`, `python src/preprocess.py`, `python src/train.py`, and `python src/evaluate.py`. Once DVC is configured, `dvc repro` runs the complete pipeline.

## DVC Google Drive setup

Initialize DVC on the `dev` branch after the first Git commit. Create a Google Drive folder, create an OAuth Desktop client in Google Cloud Console, enable the Drive API, and add the Google account used for authorization as an OAuth test user. Configure a remote using the folder ID and set the client credentials with `dvc remote modify`; do not commit OAuth secrets or token files. See the supplied DVC integration guide for the Google Cloud steps and troubleshooting.

## Project files

- `src/prepare.py`: download and save the raw Fashion-MNIST arrays.
- `src/preprocess.py`: normalize pixels and create the training/validation split.
- `src/train.py`: train and save the ANN and its history.
- `src/evaluate.py`: write test metrics and a confusion-matrix image.
- `params.yaml`: stage configuration.
- `dvc.yaml`: four-stage reproducible pipeline.

Generated datasets and models are DVC artifacts; `metrics.json` and `dvc.lock` describe the evaluated/pipeline state.