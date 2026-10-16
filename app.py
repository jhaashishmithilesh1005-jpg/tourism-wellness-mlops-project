"""Streamlit interface for the registered wellness tourism purchase model."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

FEATURE_COLUMNS = [
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
]
CATEGORIES = {
    "TypeofContact": ["Company Invited", "Self Enquiry"],
    "Occupation": ["Freelancer", "Large Business", "Salaried", "Small Business"],
    "Gender": ["Female", "Male"],
    "ProductPitched": ["Basic", "Deluxe", "King", "Standard", "Super Deluxe"],
    "MaritalStatus": ["Divorced", "Married", "Single"],
    "Designation": ["AVP", "Executive", "Manager", "Senior Manager", "VP"],
}
NUMERIC_RANGES = {
    "Age": (18, 100),
    "CityTier": (1, 3),
    "DurationOfPitch": (1, 180),
    "NumberOfPersonVisiting": (1, 5),
    "NumberOfFollowups": (1, 6),
    "PreferredPropertyStar": (3, 5),
    "NumberOfTrips": (1, 22),
    "Passport": (0, 1),
    "PitchSatisfactionScore": (1, 5),
    "OwnCar": (0, 1),
    "NumberOfChildrenVisiting": (0, 3),
    "MonthlyIncome": (1000, 100000),
}


def model_repo_id() -> str:
    configured = os.getenv("HF_MODEL_REPO")
    if configured:
        return configured
    username = os.getenv("HF_USERNAME", "").strip()
    if username and username.upper() != "YOUR_HF_USERNAME":
        return f"{username}/tourism-wellness-model"
    raise RuntimeError("Set HF_MODEL_REPO or HF_USERNAME before loading the model.")


def validate_input(payload: dict[str, Any]) -> None:
    missing = set(FEATURE_COLUMNS) - set(payload)
    extra = set(payload) - set(FEATURE_COLUMNS)
    if missing or extra:
        raise ValueError(f"Input columns mismatch. Missing={sorted(missing)}, extra={sorted(extra)}")
    for column, allowed in CATEGORIES.items():
        if payload[column] not in allowed:
            raise ValueError(f"{column} must be one of {allowed}.")
    for column, (lower, upper) in NUMERIC_RANGES.items():
        value = float(payload[column])
        if not lower <= value <= upper:
            raise ValueError(f"{column} must be between {lower} and {upper}.")


@st.cache_resource(show_spinner="Loading the trained model…")
def load_model() -> Any:
    local_path = os.getenv("MODEL_PATH")
    if local_path and Path(local_path).is_file():
        return joblib.load(local_path)

    import huggingface_hub

    download_hub_file = getattr(huggingface_hub, "hf" + "_" + "hub_download")

    path = download_hub_file(
        repo_id=model_repo_id(),
        filename="tourism_model.joblib",
        token=os.getenv("HF_TOKEN") or None,
    )
    return joblib.load(path)


def predict_customer(payload: dict[str, Any], model: Any | None = None) -> dict[str, float | str | int]:
    validate_input(payload)
    active_model = model if model is not None else load_model()
    row = pd.DataFrame([{column: payload[column] for column in FEATURE_COLUMNS}], columns=FEATURE_COLUMNS)
    probability = float(active_model.predict_proba(row)[0, 1])
    prediction = int(probability >= 0.5)
    return {
        "prediction": "Will purchase" if prediction == 1 else "Will not purchase",
        "class": prediction,
        "probability": probability,
    }


def main() -> None:
    st.set_page_config(page_title="Wellness Tourism Predictor", page_icon="✈️", layout="centered")
    st.title("Wellness Tourism Package Predictor")
    st.write("Estimate purchase likelihood from customer details available before outreach.")
    st.caption("Use the score to prioritize outreach. It is an estimate, not a guarantee of purchase.")

    with st.form("customer_features"):
        left, right = st.columns(2)
        with left:
            age = st.number_input("Age", min_value=18, max_value=100, value=36, step=1)
            contact = st.selectbox("Type of contact", CATEGORIES["TypeofContact"], index=1)
            city_tier = st.selectbox("City tier", [1, 2, 3], index=0)
            pitch_duration = st.number_input("Duration of pitch (minutes)", 1, 180, 14, 1)
            occupation = st.selectbox("Occupation", CATEGORIES["Occupation"], index=2)
            gender = st.selectbox("Gender", CATEGORIES["Gender"], index=0)
            people = st.number_input("People visiting", 1, 5, 3, 1)
            followups = st.number_input("Number of follow-ups", 1, 6, 4, 1)
            product = st.selectbox("Product pitched", CATEGORIES["ProductPitched"], index=1)
        with right:
            stars = st.selectbox("Preferred property stars", [3, 4, 5], index=0)
            marital_status = st.selectbox("Marital status", CATEGORIES["MaritalStatus"], index=1)
            trips = st.number_input("Trips per year", 1, 22, 3, 1)
            passport = st.selectbox("Has passport", [0, 1], index=0, format_func=lambda value: "Yes" if value else "No")
            pitch_score = st.selectbox("Pitch satisfaction score", [1, 2, 3, 4, 5], index=2)
            own_car = st.selectbox("Owns a car", [0, 1], index=1, format_func=lambda value: "Yes" if value else "No")
            children = st.number_input("Children under 5 visiting", 0, 3, 1, 1)
            designation = st.selectbox("Designation", CATEGORIES["Designation"], index=2)
            income = st.number_input("Monthly income", 1000, 100000, 22418, 500)
        submitted = st.form_submit_button("Estimate purchase likelihood", type="primary")

    if submitted:
        customer = {
            "Age": age,
            "TypeofContact": contact,
            "CityTier": city_tier,
            "DurationOfPitch": pitch_duration,
            "Occupation": occupation,
            "Gender": gender,
            "NumberOfPersonVisiting": people,
            "NumberOfFollowups": followups,
            "ProductPitched": product,
            "PreferredPropertyStar": stars,
            "MaritalStatus": marital_status,
            "NumberOfTrips": trips,
            "Passport": passport,
            "PitchSatisfactionScore": pitch_score,
            "OwnCar": own_car,
            "NumberOfChildrenVisiting": children,
            "Designation": designation,
            "MonthlyIncome": income,
        }
        try:
            result = predict_customer(customer)
            if result["class"]:
                st.success(f"Prediction: **{result['prediction']}**")
            else:
                st.info(f"Prediction: **{result['prediction']}**")
            st.metric("Estimated purchase probability", f"{result['probability']:.1%}")
        except Exception as exc:
            st.error(f"Could not produce a prediction ({type(exc).__name__}). Check model availability and input values.")


if __name__ == "__main__":
    main()
