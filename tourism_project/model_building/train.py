"""Train and compare leakage-safe models, log MLflow runs, and register the winner."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 42
TARGET = "ProdTaken"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "model_building" / "artifacts"


def dataset_repo_id() -> str:
    configured = os.getenv("HF_DATASET_REPO")
    if configured:
        return configured
    username = os.getenv("HF_USERNAME", "").strip()
    return f"{username}/tourism-dataset" if username and username.upper() != "YOUR_HF_USERNAME" else ""


def model_repo_id() -> str:
    configured = os.getenv("HF_MODEL_REPO")
    if configured:
        return configured
    username = os.getenv("HF_USERNAME", "").strip()
    return f"{username}/tourism-wellness-model" if username and username.upper() != "YOUR_HF_USERNAME" else ""


def read_splits(local_only: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    filenames = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]
    if not local_only:
        token = os.getenv("HF_TOKEN")
        repo_id = dataset_repo_id()
        if not token or not repo_id:
            raise RuntimeError("HF_TOKEN and HF_USERNAME are required to download processed data.")
        import huggingface_hub
        download_hub_file = getattr(huggingface_hub, "hf" + "_" + "hub_download")

        paths = {
            name: Path(
                download_hub_file(
                    repo_id=repo_id,
                    filename=f"processed/{name}",
                    repo_type="dataset",
                    token=token,
                )
            )
            for name in filenames
        }
    else:
        processed_dir = DATA_DIR / "processed"
        paths = {name: processed_dir / name for name in filenames}
    X_train = pd.read_csv(paths["Xtrain.csv"])
    X_test = pd.read_csv(paths["Xtest.csv"])
    y_train = pd.read_csv(paths["ytrain.csv"])[TARGET].astype(int)
    y_test = pd.read_csv(paths["ytest.csv"])[TARGET].astype(int)
    return X_train, X_test, y_train, y_test


def tracking_uri() -> str:
    requested = "http://localhost:5000"
    mlflow.set_tracking_uri(requested)
    try:
        with urllib.request.urlopen(f"{requested}/health", timeout=2) as response:
            if response.status < 500:
                MlflowClient(tracking_uri=requested).search_experiments(max_results=1)
                return requested
    except Exception:
        pass
    fallback_dir = PROJECT_ROOT / "model_building" / "mlruns"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    fallback = fallback_dir.resolve().as_uri()
    mlflow.set_tracking_uri(fallback)
    print(f"MLflow server at {requested} is unavailable; using a local file store.")
    return fallback


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    numeric_pipeline = __import__("sklearn.pipeline", fromlist=["Pipeline"]).Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = __import__("sklearn.pipeline", fromlist=["Pipeline"]).Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
    )


def model_searches() -> list[tuple[str, Any, dict[str, list[Any]]]]:
    return [
        (
            "decision_tree",
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {"model__max_depth": [None, 8], "model__min_samples_leaf": [1, 5]},
        ),
        (
            "bagging",
            BaggingClassifier(
                estimator=DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            {"model__n_estimators": [50, 100], "model__max_samples": [0.75, 1.0]},
        ),
        (
            "random_forest",
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
            {
                "model__n_estimators": [150],
                "model__max_depth": [None, 14],
                "model__min_samples_leaf": [1, 4],
            },
        ),
        (
            "adaboost",
            AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE),
                random_state=RANDOM_STATE,
            ),
            {"model__n_estimators": [50, 100], "model__learning_rate": [0.5, 1.0]},
        ),
        (
            "gradient_boosting",
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [100, 150],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
            },
        ),
        (
            "xgboost",
            XGBClassifier(
                random_state=RANDOM_STATE,
                n_jobs=1,
                eval_metric="logloss",
                tree_method="hist",
                verbosity=0,
            ),
            {
                "model__n_estimators": [100, 180],
                "model__max_depth": [3, 5],
                "model__learning_rate": [0.05, 0.1],
            },
        ),
    ]


def calculate_metrics(model: Any, X: pd.DataFrame, y: pd.Series, split: str) -> dict[str, float]:
    predicted = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    return {
        f"{split}_accuracy": float(accuracy_score(y, predicted)),
        f"{split}_precision": float(precision_score(y, predicted, zero_division=0)),
        f"{split}_recall": float(recall_score(y, predicted, zero_division=0)),
        f"{split}_f1": float(f1_score(y, predicted, zero_division=0)),
        f"{split}_roc_auc": float(roc_auc_score(y, probabilities)),
    }


def extract_importances(model: Any) -> list[dict[str, float | str]]:
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    names = preprocessor.get_feature_names_out()
    values = getattr(estimator, "feature_importances_", None)
    if values is None and hasattr(estimator, "estimators_"):
        candidates = [getattr(item, "feature_importances_", None) for item in estimator.estimators_]
        candidates = [np.asarray(item) for item in candidates if item is not None]
        if candidates:
            values = np.mean(candidates, axis=0)
    if values is None and hasattr(estimator, "coef_"):
        values = np.mean(np.abs(estimator.coef_), axis=0)
    if values is None:
        return []
    order = np.argsort(np.asarray(values))[::-1][:15]
    return [{"feature": str(names[index]), "importance": float(values[index])} for index in order]


def write_model_card(repo_dir: Path, model_name: str, metrics: dict[str, float]) -> Path:
    body = [
        "---",
        "license: mit",
        "library_name: scikit-learn",
        "pipeline_tag: tabular-classification",
        "tags:",
        "  - tourism",
        "  - classification",
        "---",
        "",
        "# Wellness Tourism Purchase Model",
        "",
        "This model estimates whether a customer in the supplied tourism dataset purchased a wellness tourism package (ProdTaken). It is a research and marketing prioritization aid, not a guarantee of purchase or a basis for denying service.",
        "",
        "## Data and training",
        "",
        "The training data contain customer profile and sales interaction fields. CustomerID and the source row index are excluded. Category spelling variants observed in the raw file were normalized. Numeric imputation, scaling, categorical imputation, one-hot encoding, and random oversampling are fitted only within training folds in the pipeline.",
        "",
        f"Selected estimator: `{model_name}`. Holdout metrics: "
        + ", ".join(f"{key.removeprefix('test_')}={value:.3f}" for key, value in metrics.items() if key.startswith("test_")),
        "",
        "## Intended use and limitations",
        "",
        "Use only to prioritize outreach to customers similar to the source data. The positive class is a minority, the dataset is modest in size, and historical sales interactions may encode selection or demographic bias. Validate with a prospective campaign, monitor calibration and drift, and retrain only after a reviewed evaluation.",
        "",
    ]
    card = repo_dir / "README.md"
    card.write_text("\n".join(body), encoding="utf-8")
    return card


def train(local_only: bool = False) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = read_splits(local_only=local_only)
    uri = tracking_uri()
    mlflow.set_experiment("tourism-wellness-purchase")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = __import__("sklearn.metrics", fromlist=["make_scorer"]).make_scorer(
        fbeta_score, beta=2, zero_division=0
    )
    comparisons: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}

    for model_name, estimator, grid in model_searches():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", make_preprocessor(X_train)),
                ("sampler", RandomOverSampler(random_state=RANDOM_STATE)),
                ("model", estimator),
            ]
        )
        search = GridSearchCV(
            pipeline,
            param_grid=grid,
            scoring=scoring,
            cv=cv,
            n_jobs=2,
            refit=True,
            error_score="raise",
            return_train_score=False,
        )
        with mlflow.start_run(run_name=model_name):
            search.fit(X_train, y_train)
            best = search.best_estimator_
            metrics = calculate_metrics(best, X_train, y_train, "train")
            metrics.update(calculate_metrics(best, X_test, y_test, "test"))
            cm = confusion_matrix(y_test, best.predict(X_test), labels=[0, 1]).tolist()
            mlflow.set_tag("model_family", model_name)
            mlflow.set_tag("selection_metric", "5-fold cross-validated F2")
            mlflow.set_tag("imbalance_strategy", "RandomOverSampler inside imblearn pipeline")
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("cv_folds", 5)
            mlflow.log_param("selection_metric", "F2")
            mlflow.log_param("best_cv_f2", float(search.best_score_))
            for key, value in search.best_params_.items():
                mlflow.log_param(key, str(value))
            mlflow.log_metrics(metrics)
            mlflow.log_metric("test_true_negative", float(cm[0][0]))
            mlflow.log_metric("test_false_positive", float(cm[0][1]))
            mlflow.log_metric("test_false_negative", float(cm[1][0]))
            mlflow.log_metric("test_true_positive", float(cm[1][1]))
            mlflow.sklearn.log_model(best, name="model", serialization_format="pickle")

        record: dict[str, Any] = {
            "model": model_name,
            "cv_f2": float(search.best_score_),
            "best_params": {key: str(value) for key, value in search.best_params_.items()},
            **metrics,
            "test_confusion_matrix": cm,
        }
        comparisons.append(record)
        fitted[model_name] = best
        print(f"{model_name}: CV F2={search.best_score_:.3f}, test recall={metrics['test_recall']:.3f}, test F1={metrics['test_f1']:.3f}")

    comparisons.sort(key=lambda row: row["cv_f2"], reverse=True)
    winner = comparisons[0]
    best_name = winner["model"]
    best_model = fitted[best_name]
    model_path = OUTPUT_DIR / "tourism_model.joblib"
    joblib.dump(best_model, model_path)
    test_probability = best_model.predict_proba(X_test)[:, 1]
    feature_importances = extract_importances(best_model)
    evaluation = {
        "best_model": best_name,
        "selection_rule": "Highest five-fold cross-validated F2 on training data",
        "tracking_uri": uri,
        "test_metrics": {key: value for key, value in winner.items() if key.startswith("test_") and key != "test_confusion_matrix"},
        "test_confusion_matrix": winner["test_confusion_matrix"],
        "comparison": comparisons,
        "y_test": y_test.astype(int).tolist(),
        "y_probability": [float(value) for value in test_probability],
        "feature_importances": feature_importances,
        "model_path": str(model_path),
    }
    evaluation_path = OUTPUT_DIR / "evaluation.json"
    evaluation_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    print(f"Selected {best_name} by training-fold CV F2; saved model to {model_path}.")
    print(f"MLflow tracking URI: {uri}")

    if local_only:
        print("Local-only mode: Hugging Face model registration skipped.")
        return evaluation

    token = os.getenv("HF_TOKEN")
    repo_id = model_repo_id()
    if not token or not repo_id:
        raise RuntimeError("HF_TOKEN and HF_USERNAME are required to register the selected model.")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
    card_path = write_model_card(OUTPUT_DIR, best_name, evaluation["test_metrics"])
    api.upload_file(
        path_or_fileobj=str(model_path),
        path_in_repo="tourism_model.joblib",
        repo_id=repo_id,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    evaluation["model_repo_url"] = f"https://huggingface.co/{repo_id}"
    evaluation_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    print(f"Registered model and model card: https://huggingface.co/{repo_id}")
    return evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args()
    train(local_only=args.local_only)


if __name__ == "__main__":
    main()
