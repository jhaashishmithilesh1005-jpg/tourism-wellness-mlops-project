"""Register the raw tourism dataset on the Hugging Face Hub."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def dataset_repo_id() -> str:
    configured = os.getenv("HF_DATASET_REPO")
    if configured:
        return configured
    username = os.getenv("HF_USERNAME", "").strip()
    if username and username.upper() != "YOUR_HF_USERNAME":
        return f"{username}/tourism-dataset"
    return ""


def register_dataset(dataset_path: Path, allow_local_only: bool = False) -> str | None:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    token = os.getenv("HF_TOKEN")
    repo_id = dataset_repo_id()
    if not token or not repo_id:
        if allow_local_only:
            print(f"Local data is ready at {dataset_path}. Hub upload skipped because HF_TOKEN/HF_USERNAME is not configured.")
            return None
        raise RuntimeError("HF_TOKEN and HF_USERNAME (or HF_DATASET_REPO) are required for dataset registration.")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(dataset_path),
        path_in_repo="tourism.csv",
        repo_id=repo_id,
        repo_type="dataset",
    )
    url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"Registered raw dataset: {url}")
    return url


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "tourism.csv",
    )
    parser.add_argument("--allow-local-only", action="store_true")
    args = parser.parse_args()
    register_dataset(args.dataset_path, allow_local_only=args.allow_local_only)


if __name__ == "__main__":
    main()
