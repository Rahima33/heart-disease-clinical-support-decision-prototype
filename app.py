import os
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
import shap
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
ONEHOT_BASE_FIELDS = ("sex", "cp", "fbs", "restecg", "exang", "slope", "thal")

BASE_DISPLAY_NAMES = {
    "age": "Age",
    "ca": "Number of major vessels",
    "chol": "Cholesterol level",
    "cp": "Chest pain type",
    "exang": "Exercise-induced angina",
    "fbs": "Fasting blood sugar",
    "has_missing": "Incomplete diagnostic testing",
    "oldpeak": "ST depression (oldpeak)",
    "restecg": "Resting ECG result",
    "sex": "Sex",
    "slope": "Exercise ST segment slope",
    "thal": "Thallium stress test result",
    "thalch": "Max heart rate achieved",
    "trestbps": "Resting blood pressure",
}

NUMERIC_TRAINING_RANGES = {
    "age": (28, 77),
    "trestbps": (0, 200),
    "chol": (0, 603),
    "thalch": (60, 202),
    "oldpeak": (-2.6, 6.2),
    "ca": (0, 3),
}

CATEGORY_DISPLAY_NAMES = {
    "cp_asymptomatic": "asymptomatic chest pain",
    "cp_atypical angina": "atypical angina",
    "cp_non-anginal": "non-anginal pain",
    "cp_typical angina": "typical angina",
    "restecg_lv hypertrophy": "left ventricular hypertrophy on ECG",
    "restecg_normal": "normal resting ECG",
    "restecg_st-t abnormality": "ST-T abnormality on ECG",
    "slope_downsloping": "downsloping ST segment",
    "slope_flat": "flat ST segment",
    "slope_upsloping": "upsloping ST segment",
    "thal_fixed defect": "fixed thallium defect",
    "thal_normal": "normal thallium result",
    "thal_reversable defect": "reversible thallium defect",
}


class PatientInput(BaseModel):
    age: int
    sex: Literal["Male", "Female"]
    cp: Literal["typical angina", "atypical angina", "non-anginal", "asymptomatic"]
    trestbps: float
    chol: float
    fbs: bool
    restecg: Literal["normal", "lv hypertrophy", "st-t abnormality"]
    thalch: float
    exang: bool
    oldpeak: float
    slope: Literal["upsloping", "flat", "downsloping"]
    ca: float
    thal: Literal["normal", "fixed defect", "reversable defect"]
    has_missing: bool


def load_env_file(path):
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip().lstrip("\ufeff"), value.strip().strip('"').strip("'"))


def validate_input_ranges(patient_dict):
    warnings = []
    for field, (minimum, maximum) in NUMERIC_TRAINING_RANGES.items():
        value = patient_dict.get(field)
        if value is None:
            continue
        if value < minimum or value > maximum:
            label = BASE_DISPLAY_NAMES.get(field, field)
            warnings.append(
                f"{label} ({value}) is outside the model training range ({minimum}-{maximum}); prediction and factor explanations may be unreliable."
            )
    return warnings


def clean_feature_name(feature):
    return feature.replace("num__", "", 1).replace("cat__", "", 1)


def base_feature_name(feature):
    clean_feature = clean_feature_name(feature)
    for base_field in ONEHOT_BASE_FIELDS:
        if clean_feature.startswith(f"{base_field}_"):
            return base_field
    return clean_feature


def category_from_onehot(clean_feature, base_feature):
    prefix = f"{base_feature}_"
    if clean_feature.startswith(prefix):
        return clean_feature[len(prefix):]
    return None


def boolean_display_name(base_feature, category, value):
    present = float(value) >= 0.5
    category_is_true = category == "True"
    actual_value = category_is_true if present else not category_is_true

    if base_feature == "exang":
        return "Exercise-induced angina present" if actual_value else "No exercise-induced angina"
    if base_feature == "fbs":
        return "Fasting blood sugar elevated" if actual_value else "Fasting blood sugar not elevated"

    return f"{BASE_DISPLAY_NAMES.get(base_feature, base_feature)}: {actual_value}"


def display_factor_name(feature, value):
    clean_feature = clean_feature_name(feature)
    base_feature = base_feature_name(feature)
    category = category_from_onehot(clean_feature, base_feature)

    if category is None:
        return BASE_DISPLAY_NAMES.get(clean_feature, clean_feature.replace("_", " "))

    if base_feature in {"exang", "fbs"}:
        return boolean_display_name(base_feature, category, value)

    # Sex gets special handling: the one-hot column is "sex_Male" (True/False).
    # We always want to state the actual sex plainly — "Sex: Female" or "Sex: Male" —
    # never "Sex: not Male", which was ambiguous and led the LLM to misstate direction.
    if base_feature == "sex":
        present = float(value) >= 0.5  # True if this row's "Male" column fired
        is_male = (category == "Male") == present
        return f"Sex: {'Male' if is_male else 'Female'}"

    category_key = f"{base_feature}_{category}"
    category_label = CATEGORY_DISPLAY_NAMES.get(category_key, category.replace("_", " "))
    present = float(value) >= 0.5
    base_label = BASE_DISPLAY_NAMES.get(base_feature, base_feature.replace("_", " "))
    if present:
        return f"{base_label}: {category_label}"
    return f"{base_label}: not {category_label}"


def raw_display_name(feature, value):
    label = BASE_DISPLAY_NAMES.get(feature, feature.replace("_", " "))

    if feature == "exang":
        return "Exercise-induced angina present" if value else "No exercise-induced angina"
    if feature == "fbs":
        return "Fasting blood sugar elevated" if value else "Fasting blood sugar not elevated"
    if isinstance(value, bool):
        return f"{label}: {'Yes' if value else 'No'}"
    if feature in ONEHOT_BASE_FIELDS:
        return f"{label}: {value}"

    return label


def aggregate_base_factors(explanation_df, patient_dict):
    explanation_df = explanation_df.copy()
    explanation_df["base_feature"] = explanation_df["feature"].apply(base_feature_name)

    rows = []
    for feature, group in explanation_df.groupby("base_feature", sort=False):
        rows.append(
            {
                "feature": feature,
                "value": patient_dict.get(feature, group["value"].iloc[0]),
                "shap_impact": group["shap_impact"].sum(),
            }
        )

    return pd.DataFrame(rows).sort_values("shap_impact", key=abs, ascending=False)


def add_display_fields(top_factors):
    for factor in top_factors:
        factor["display_feature"] = raw_display_name(factor["feature"], factor["value"])
    return top_factors


def describe_patient(patient_dict):
    """Builds a clean, human-readable patient summary from the original raw input —
    this is what gets fed to the LLM, instead of one-hot SHAP fragments, so it can't
    misphrase or hallucinate values it would otherwise have to guess at."""
    lines = []
    for field, label in BASE_DISPLAY_NAMES.items():
        value = patient_dict.get(field)
        if value is None:
            continue
        if isinstance(value, bool):
            value = "Yes" if value else "No"
        lines.append(f"- {label}: {value}")
    return "\n".join(lines)


load_env_file(BASE_DIR / ".env")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY")) if os.environ.get("GROQ_API_KEY") else None

# Load the saved pipeline (trained and saved in main.py)
clf_final = joblib.load(BASE_DIR / "model_pipeline.pkl")

# Rebuild the SHAP explainer from the loaded model
rf_model = clf_final.named_steps["model"]
explainer = shap.TreeExplainer(rf_model)
feature_names = clf_final.named_steps["preprocessor"].get_feature_names_out()


@app.get("/")
def root():
    return {"message": "CardioRisk API is running. Go to /docs to test the /predict endpoint."}


def generate_summary(probability, top_factors, patient_dict, warnings=None):
    patient_description = describe_patient(patient_dict)

    # Direction is now stated explicitly in words (INCREASED/DECREASED) right next to
    # each factor, rather than relying on the LLM correctly inferring sign from a small
    # decimal number — this is what caused the earlier "Sex: Not Male" direction flip.
    factors_text = "\n".join(
        [
            f"- {f.get('display_feature', f['feature'])}: this {'INCREASED' if f['shap_impact'] > 0 else 'DECREASED'} the predicted risk"
            for f in top_factors
        ]
    )

    warning_text = "\n".join([f"- {warning}" for warning in warnings or []])

    def local_summary(reason):
        factor_phrases = [
            f"{f.get('display_feature', f['feature'])} {'increased' if f['shap_impact'] > 0 else 'decreased'} the predicted risk"
            for f in top_factors
        ]
        factors_sentence = "; ".join(factor_phrases)
        warning_sentence = (
            " Input reliability warnings: " + " ".join(warnings)
            if warnings
            else " No input reliability warnings were detected."
        )
        return (
            f"The model estimated a cardiovascular disease probability of {probability:.0%}. "
            f"Main factors were: {factors_sentence}."
            f"{warning_sentence} {reason}"
        )

    if groq_client is None:
        return local_summary("Set GROQ_API_KEY in the backend .env file to generate the LLM-written clinical summary.")

    prompt = f"""You are assisting a clinician reviewing an automated cardiovascular risk prediction.

Patient information:
{patient_description}

Predicted disease probability: {probability:.0%}

Top contributing factors for this specific prediction (already collapsed from one-hot model columns back to original clinical features). Each factor's effect on risk is stated explicitly — copy this direction exactly, do not re-infer or reverse it:
{factors_text}

Input reliability warnings:
{warning_text if warning_text else "- None"}

Write a brief, clear, 2-3 sentence clinical summary explaining this prediction in plain language.
- Use the exact INCREASED/DECREASED direction given for each factor above — do not flip it.
- Mention only factors listed in the Top contributing factors section as drivers of this prediction.
- Refer to each listed factor's actual value directly, using the Patient information section as ground truth.
- Do not use double negatives like "not having X" or "not being male" — state plainly what the patient does have/is (e.g. say "female" rather than "not male").
- Do not combine increased-risk and decreased-risk factors in the same phrase.
- If there are input reliability warnings, mention that the model output may be less reliable for those specific inputs.
- Do not give medical advice or a diagnosis — only explain what drove this specific model output.
- Be neutral and factual."""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception:
        return local_summary("The LLM summary service was unavailable, so this local summary was generated instead.")


@app.post("/predict")
def predict(patient: PatientInput):
    # Convert validated input into a one-row DataFrame, same shape as training data
    patient_dict = patient.dict()
    warnings = validate_input_ranges(patient_dict)
    patient_df = pd.DataFrame([patient_dict])

    # 1. Predict probability
    probability = clf_final.predict_proba(patient_df)[0][1]

    # 2. Transform input the same way training data was transformed
    patient_transformed = clf_final.named_steps["preprocessor"].transform(patient_df)

    # 3. Run SHAP on the transformed input
    patient_shap_values = explainer.shap_values(patient_transformed)
    patient_shap_disease = patient_shap_values[:, :, 1][0]

    # 4. Build the top-factors explanation table
    explanation = pd.DataFrame(
        {
            "feature": feature_names,
            "value": patient_transformed[0],
            "shap_impact": patient_shap_disease,
        }
    ).sort_values("shap_impact", key=abs, ascending=False)

    explanation = aggregate_base_factors(explanation, patient_dict)
    top_factors = add_display_fields(explanation.head(5).to_dict(orient="records"))
    summary = generate_summary(probability, top_factors, patient_dict, warnings)

    return {
        "probability": float(probability),
        "top_factors": top_factors,
        "summary": summary,
        "warnings": warnings,
    }