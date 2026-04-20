from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import joblib
import os
import re

app = FastAPI(title="Credit Default Prediction API v2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
model_full = None
model_lite = None
df_processed = None  # indexed by SK_ID_CURR

# ------------------------------------------------
# Helpers
# ------------------------------------------------

def sanitize(col: str) -> str:
    return re.sub(r'[^A-Za-z0-9_]', '_', col)

def build_inference_row(raw_dict: dict, expected_features: list) -> np.ndarray:
    """
    Turn a flat dict of {column: value} (mix of numeric & original categorical strings)
    into a one-hot encoded numpy row matching `expected_features`.
    """
    row = {f: 0.0 for f in expected_features}

    for key, value in raw_dict.items():
        skey = sanitize(str(key))
        # Direct numeric or already-encoded column
        if skey in row:
            try:
                row[skey] = float(value)
            except (ValueError, TypeError):
                pass
        else:
            # Categorical → find the dummy column
            dummy = f"{skey}_{sanitize(str(value))}"
            if dummy in row:
                row[dummy] = 1.0

    return np.array([list(row.values())])


# ------------------------------------------------
# Startup
# ------------------------------------------------

@app.on_event("startup")
def load_assets():
    global model_full, model_lite, df_processed
    base = os.path.dirname(__file__)

    full_path = os.path.join(base, "..", "models", "lgb_model_full.pkl")
    lite_path = os.path.join(base, "..", "models", "lgb_model_lite.pkl")

    if not os.path.exists(full_path) or not os.path.exists(lite_path):
        raise RuntimeError("Models not found. Run: python src/train_model.py")

    print("Loading models...")
    model_full = joblib.load(full_path)
    model_lite = joblib.load(lite_path)
    print(f"  Model A features: {len(model_full['features'])}")
    print(f"  Model B features: {len(model_lite['features'])}")

    data_path = os.path.join(base, "..", "Data", "processed", "train_processed.csv")
    if os.path.exists(data_path):
        print("Loading applicant database (this takes ~30s)...")
        df_processed = pd.read_csv(data_path)
        df_processed.set_index('SK_ID_CURR', inplace=True)
        print(f"  Database loaded: {len(df_processed):,} applicants")
    else:
        print("WARNING: train_processed.csv not found — new-applicant mode only.")


# ------------------------------------------------
# Schema
# ------------------------------------------------

class PredictRequest(BaseModel):
    # Optional: if provided and found, Model A (full) is used
    SK_ID_CURR: Optional[int] = None

    # 15 form fields — all required so the form is complete
    NAME_CONTRACT_TYPE: str
    CODE_GENDER: str
    FLAG_OWN_CAR: str
    FLAG_OWN_REALTY: str
    CNT_CHILDREN: int
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    AMT_GOODS_PRICE: float
    NAME_INCOME_TYPE: str
    NAME_EDUCATION_TYPE: str
    NAME_FAMILY_STATUS: str
    NAME_HOUSING_TYPE: str
    DAYS_BIRTH: int
    DAYS_EMPLOYED: int


# ------------------------------------------------
# Predict
# ------------------------------------------------

@app.post("/predict")
def predict(request: PredictRequest):
    req = request.dict()
    sk_id = req.pop("SK_ID_CURR", None)

    # --- Decide which model to use ---
    has_history = (
        sk_id is not None
        and sk_id > 0
        and df_processed is not None
        and sk_id in df_processed.index
    )

    if has_history:
        # Pull full historical record and override with fresh form data
        record = df_processed.loc[sk_id]
        if isinstance(record, pd.DataFrame):
            record = record.iloc[0]

        # Start from historical values
        full_dict = record.to_dict()

        # One-hot encode the historical object columns on the fly
        # (they're stored as raw strings in df_processed)
        # We'll override with form values below
        for k, v in req.items():
            full_dict[k] = v  # form takes precedence

        inf_row = build_inference_row(full_dict, model_full['features'])
        clf = model_full['model']
        thresh = model_full['best_thresh']
        model_label = f"Full Model (200+ features, applicant #{sk_id})"
    else:
        inf_row = build_inference_row(req, model_lite['features'])
        clf = model_lite['model']
        thresh = model_lite['best_thresh']
        model_label = "Lite Model (15 form features, new applicant)"

    prob = float(clf.predict_proba(inf_row)[0, 1])
    prediction = int(prob > thresh)

    return {
        "prediction": prediction,
        "risk_level": "High Risk" if prediction == 1 else "Low Risk",
        "probability": round(prob, 4),
        "threshold": round(thresh, 2),
        "model_used": model_label,
        "has_history": has_history,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_full_loaded": model_full is not None,
        "model_lite_loaded": model_lite is not None,
        "database_loaded": df_processed is not None,
        "database_size": len(df_processed) if df_processed is not None else 0,
    }
