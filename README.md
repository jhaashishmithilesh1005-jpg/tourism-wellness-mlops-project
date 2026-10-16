# Wellness Tourism Purchase Prediction

This repository contains a reproducible machine learning pipeline for estimating `ProdTaken` from customer and sales interaction features.

## Repository layout

- `tourism_project/data/tourism.csv`: source dataset.
- `tourism_project/model_building/data_register.py`: idempotent Hugging Face dataset registration.
- `tourism_project/model_building/prep.py`: category cleanup, stratified 80/20 split, and Hub upload.
- `tourism_project/model_building/train.py`: model comparison, five-fold F2 tuning, MLflow tracking, evaluation, and winning model registration.
- `tourism_project/deployment/`: Streamlit app, Dockerfile, and pinned runtime requirements.
- `tourism_project/hosting/hosting.py`: optional Hugging Face Docker Space publisher (requires a paid HF plan).
- `.github/workflows/pipeline.yml`: ordered GitHub Actions workflow.

## Free public app deployment

The free hosting route is Streamlit Community Cloud. The app downloads the public model from `ashishjha001/tourism-wellness-model`; it does not need an HF token or any Streamlit secrets.

1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) using GitHub and authorize access to this public repository.
2. Choose **Create app** and select `jhaashishmithilesh1005-jpg/tourism-wellness-mlops-project`, branch `main`.
3. Set the app file to `tourism_project/deployment/app.py`.
4. In **Advanced settings**, select Python `3.12`, then deploy. The dependencies are in `tourism_project/deployment/requirements.txt`.
5. Wait for the app health check to finish. Copy the public app URL and save it as `STREAMLIT_APP_URL` in the notebook only if you want the notebook to check the live URL.

Detailed click-by-click steps are in [STREAMLIT_CLOUD_DEPLOYMENT.md](STREAMLIT_CLOUD_DEPLOYMENT.md). Streamlit Community Cloud redeploys the app when the connected GitHub branch receives updates.

## GitHub Actions and Hugging Face

The workflow registers the dataset, prepares features, trains/evaluates the model, and publishes the model to Hugging Face. The optional `deploy-hosting` job is skipped by default because public Hugging Face Docker Spaces require a paid plan. The free Streamlit app is deployed by connecting the repository once in Streamlit Community Cloud.

The workflow uses the repository secret `HF_TOKEN` for dataset/model uploads. Never put that token in source files, the notebook, or Streamlit secrets. The public model itself can be downloaded without authentication.

## Run locally

Use Python 3.12. Install project dependencies with `python -m pip install -r tourism_project/requirements.txt`. For Hub operations set `HF_USERNAME=ashishjha001` and provide `HF_TOKEN` through a local environment variable. For a local-only run, use `--allow-local-only` for registration and `--local-only` for preparation/training. The app accepts `MODEL_PATH` to load a local joblib model.

## Evaluation and limitations

The selection rule uses five-fold cross-validated F2 to give recall more weight because missing a likely buyer is costly. Review test precision and outreach capacity before using scores in a campaign. The dataset is modest, buyers are a minority class, and historical interactions can reflect selection bias. Monitor drift and prospective campaign outcomes before retraining.
