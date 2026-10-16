# Wellness Tourism Purchase Prediction

This repository contains a reproducible pipeline for estimating `ProdTaken` from customer and sales interaction features.

## Repository layout

- `tourism_project/data/tourism.csv`: source dataset.
- `tourism_project/model_building/data_register.py`: idempotent Hugging Face dataset registration.
- `tourism_project/model_building/prep.py`: verified category cleanup, stratified 80/20 split, and Hugging Face upload.
- `tourism_project/model_building/train.py`: six model families, 5-fold F2 tuning, MLflow tracking, evaluation, and model registration.
- `tourism_project/deployment/`: Streamlit app, Dockerfile, and pinned runtime requirements.
- `tourism_project/hosting/hosting.py`: public Docker Space creation and upload.
- `.github/workflows/pipeline.yml`: ordered GitHub Actions workflow.

## Run locally

Use Python 3.12. Install the pinned dependencies with `python -m pip install -r tourism_project/requirements.txt`. Set `HF_USERNAME` and `HF_TOKEN` in the environment. The workflow registers the raw file, prepares data from the Hub, trains and evaluates the models, and publishes the winning model and app.

For a local-only run without Hub credentials, call each script with `--allow-local-only` for registration, `--local-only` for preparation, and `--local-only` for training. The app accepts `MODEL_PATH` to load a local joblib model.

## Evaluation and limitations

The selection rule uses five-fold cross-validated F2 to give recall more weight because missing a likely buyer is costly. Review test precision and outreach capacity before using scores in a campaign. The dataset is modest, buyers are a minority class, and historical interactions can reflect selection bias. Monitor drift and prospective campaign outcomes before retraining.
