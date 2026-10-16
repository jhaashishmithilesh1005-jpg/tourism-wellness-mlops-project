"""Create a public Docker Space and upload the Streamlit deployment folder."""

from __future__ import annotations

import os
import time
from pathlib import Path

import requests
from huggingface_hub import HfApi


def space_repo_id() -> str:
    configured = os.getenv("HF_SPACE_REPO")
    if configured:
        return configured
    username = os.getenv("HF_USERNAME", "").strip()
    if username and username.upper() != "YOUR_HF_USERNAME":
        return f"{username}/tourism-wellness-app"
    raise RuntimeError("Set HF_SPACE_REPO or HF_USERNAME before creating the Space.")


def write_space_readme(deployment_dir: Path) -> None:
    readme = deployment_dir / "README.md"
    readme.write_text(
        "---\n"
        "title: Wellness Tourism Purchase Predictor\n"
        "emoji: ✈️\n"
        "colorFrom: blue\n"
        "colorTo: green\n"
        "sdk: docker\n"
        "app_port: 8501\n"
        "pinned: false\n"
        "---\n\n"
        "Streamlit app that estimates wellness tourism package purchase likelihood.\n",
        encoding="utf-8",
    )


def deploy() -> str:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required to create and upload the Hugging Face Space.")
    repo_id = space_repo_id()
    deployment_dir = Path(__file__).resolve().parents[1] / "deployment"
    if not (deployment_dir / "app.py").is_file() or not (deployment_dir / "Dockerfile").is_file():
        raise FileNotFoundError("The deployment folder must contain app.py and Dockerfile.")

    write_space_readme(deployment_dir)
    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        private=False,
        exist_ok=True,
    )
    try:
        api.update_repo_settings(repo_id=repo_id, repo_type="space", private=False)
    except Exception:
        # create_repo(private=False) already requests public visibility.
        pass
    api.upload_folder(
        folder_path=str(deployment_dir),
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=["__pycache__", "*.pyc"],
    )
    url = f"https://huggingface.co/spaces/{repo_id}"
    print(f"Created/updated public Docker Space: {url}")

    app_url = f"https://{repo_id.replace('/', '-')}.hf.space"
    last_error = "Space is still building."
    for _ in range(30):
        try:
            response = requests.get(app_url, timeout=15)
            if response.status_code == 200:
                print(f"Live Space returned HTTP {response.status_code}: {app_url}")
                return url
            last_error = f"HTTP {response.status_code}"
        except requests.RequestException as exc:
            last_error = type(exc).__name__
        time.sleep(10)
    print(f"Space upload completed, but the live app did not become ready: {last_error}.")
    return url


if __name__ == "__main__":
    deploy()
