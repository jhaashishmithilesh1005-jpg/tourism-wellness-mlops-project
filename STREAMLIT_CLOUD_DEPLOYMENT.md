# Deploy the tourism predictor on Streamlit Community Cloud (free)

This project uses the public model `ashishjha001/tourism-wellness-model`, so the app does not need your Hugging Face access token.

## Create the app

1. Open [Streamlit Community Cloud](https://share.streamlit.io/) and choose **Continue with GitHub** (or sign in if you already connected GitHub).
2. Authorize Streamlit to read your public GitHub repositories. If prompted for repository access, grant access to `tourism-wellness-mlops-project`.
3. On the Community Cloud home page, choose **Create app**.
4. In the repository selector, choose owner `jhaashishmithilesh1005-jpg`, repository `tourism-wellness-mlops-project`, and branch `main`.
5. For **Main file path**, enter exactly:

   `tourism_project/deployment/app.py`

6. Open **Advanced settings**, choose Python version **3.12**, and leave **Secrets** empty. The app reads the public HF model; no HF token is needed.
7. Choose **Deploy**. The first build can take several minutes while dependencies install.
8. When the app shows as running, open it and try **Estimate purchase likelihood**. Copy its public URL (usually `https://<generated-name>.streamlit.app`).

## If Streamlit asks for a requirements file

The pinned app dependencies are at `tourism_project/deployment/requirements.txt`, beside the selected app file. Community Cloud should use that subdirectory's requirements file. If the build log says packages are missing, open the app's **Manage app** page and inspect the build log before changing dependency versions.

## Connect the URL back to the notebook

In the notebook, set `STREAMLIT_APP_URL` to the public URL in the environment before running the live app check in **Output Evaluation**. Keep it as a URL only; never add an access token there. Add a screenshot of the deployed UI at the assignment's hosting evidence section.

## Workflow note

GitHub Actions continues to register the dataset, prepare it, train/evaluate the model, and publish the public model. The Hugging Face Docker Space job is disabled by default because that hosting option requires a paid Hugging Face plan. Streamlit Community Cloud is the free public app host for this repository.
