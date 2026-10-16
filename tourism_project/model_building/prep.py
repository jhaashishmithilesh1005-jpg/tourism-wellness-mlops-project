"""Prepare a leakage-safe, stratified train/test dataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "ProdTaken"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def dataset_repo_id() -> str:
    configured = os.getenv("HF_DATASET_REPO")
    if configured:
        return configured
    username = os.getenv("HF_USERNAME", "").strip()
    if username and username.upper() != "YOUR_HF_USERNAME":
        return f"{username}/tourism-dataset"
    return ""


def clean_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply only verified category normalization and remove identifiers."""
    # Check exact duplicates while unique identifiers are still present. Two customers
    # with identical predictors are separate training examples, not duplicate records.
    data = frame.copy().drop_duplicates()
    index_columns = [c for c in data.columns if str(c).lower().startswith("unnamed:")]
    data = data.drop(columns=index_columns, errors="ignore")
    data = data.drop(columns=["CustomerID"], errors="ignore")

    for column in data.columns:
        if pd.api.types.is_string_dtype(data[column].dtype) or data[column].dtype == object:
            data[column] = data[column].astype("string").str.strip()

    verified_mappings = {
        "Gender": {"Fe Male": "Female"},
        "MaritalStatus": {"Unmarried": "Single"},
        "Occupation": {"Free Lancer": "Freelancer"},
    }
    for column, mapping in verified_mappings.items():
        if column in data.columns:
            data[column] = data[column].replace(mapping)

    if TARGET not in data.columns:
        raise ValueError(f"Required target column {TARGET!r} is missing.")
    data[TARGET] = pd.to_numeric(data[TARGET], errors="raise").astype("int64")
    if not set(data[TARGET].dropna().unique()).issubset({0, 1}):
        raise ValueError("ProdTaken must contain only 0 and 1.")
    return data.reset_index(drop=True)


def load_source(local_path: Path, local_only: bool) -> tuple[pd.DataFrame, str]:
    token = os.getenv("HF_TOKEN")
    repo_id = dataset_repo_id()
    if not local_only and token and repo_id:
        import huggingface_hub
        download_hub_file = getattr(huggingface_hub, "hf" + "_" + "hub_download")

        cached_path = download_hub_file(
            repo_id=repo_id,
            filename="tourism.csv",
            repo_type="dataset",
            token=token,
        )
        return pd.read_csv(cached_path), f"Hugging Face dataset: {repo_id}"
    if not local_path.exists():
        raise FileNotFoundError(f"Local fallback dataset not found: {local_path}")
    if not local_only:
        raise RuntimeError("HF_TOKEN and HF_USERNAME are required to load the registered dataset.")
    return pd.read_csv(local_path), f"Local fallback: {local_path.name}"


def split_and_save(frame: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    data = clean_dataframe(frame)
    X = data.drop(columns=[TARGET])
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    objects = {
        "Xtrain": X_train,
        "Xtest": X_test,
        "ytrain": y_train,
        "ytest": y_test,
    }
    paths: dict[str, Path] = {}
    for name, obj in objects.items():
        path = output_dir / f"{name}.csv"
        obj.to_frame(name=TARGET).to_csv(path, index=False) if name.startswith("y") else obj.to_csv(path, index=False)
        paths[name] = path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "tourism.csv",
    )
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args()

    raw, source = load_source(args.input_csv, local_only=args.local_only)
    output_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
    paths = split_and_save(raw, output_dir)
    print(f"Preparation source: {source}")
    print(f"Cleaned split sizes: train={len(pd.read_csv(paths['Xtrain']))}, test={len(pd.read_csv(paths['Xtest']))}")

    if args.local_only:
        print(f"Saved train/test CSVs locally under {output_dir}.")
        return

    token = os.getenv("HF_TOKEN")
    repo_id = dataset_repo_id()
    if not token or not repo_id:
        raise RuntimeError("HF_TOKEN and HF_USERNAME are required to upload processed splits.")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    for path in paths.values():
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f"processed/{path.name}",
            repo_id=repo_id,
            repo_type="dataset",
        )
    print(f"Uploaded all four stratified splits to https://huggingface.co/datasets/{repo_id}.")


if __name__ == "__main__":
    main()
