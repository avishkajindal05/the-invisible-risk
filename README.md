<div align="center">

<h1>🏦 The Invisible Risk</h1>
<p><strong>AI-Powered Credit Default Prediction System</strong></p>

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img alt="LightGBM" src="https://img.shields.io/badge/LightGBM-Dual--Model-brightgreen?style=flat-square"/>
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-Backend-009688?style=flat-square&logo=fastapi&logoColor=white"/>
  <img alt="React" src="https://img.shields.io/badge/React-19-61DAFB?style=flat-square&logo=react&logoColor=black"/>
  <img alt="SMOTE" src="https://img.shields.io/badge/SMOTE-Balanced-orange?style=flat-square"/>
  <img alt="F1" src="https://img.shields.io/badge/F1%20Score-0.95-success?style=flat-square"/>
  <img alt="AUC" src="https://img.shields.io/badge/ROC%20AUC-0.98-blue?style=flat-square"/>
</p>

> A production-ready, full-stack credit default prediction system using a minimal application form. Built on the Home Credit dataset with a dual LightGBM model architecture, SMOTE-balanced training, and a modern glassmorphism React UI.

</div>

---

## 📌 Table of Contents
1. [Overview](#overview)  
2. [Architecture](#architecture)  
3. [Model Performance](#model-performance)  
4. [Project Structure](#project-structure)  
5. [Prerequisites](#prerequisites)  
6. [Setup & Installation](#setup--installation)  
7. [Training the Models](#training-the-models)  
8. [Running the Application](#running-the-application)  
9. [API Reference](#api-reference)  
10. [Data](#data)

---

## Overview

**The Invisible Risk** predicts whether a loan applicant will default using a short web form — without requiring access to complex bureau records. Powered by two LightGBM classifiers and SMOTE oversampling for handling class imbalance:

- **Existing applicants** (known `SK_ID_CURR`) → Full 200+ feature model using historical data
- **New applicants** (no ID) → Lite 15-feature form-only model

Both models achieve **F1 > 0.94** and **ROC AUC > 0.97**.

---

## Architecture

```
┌─────────────────┐        POST /predict        ┌───────────────────────────┐
│  React Frontend │  ─────────────────────────► │  FastAPI Backend           │
│  (Vite @ :5173) │                             │  (Uvicorn @ :8000)         │
└─────────────────┘                             │                            │
                                                │  SK_ID provided & found?   │
                                                │  ├── YES → Model A (Full)  │
                                                │  │         200+ features   │
                                                │  └── NO  → Model B (Lite) │
                                                │            15 form fields  │
                                                └───────────┬───────────────┘
                                                            │
                                          ┌─────────────────┴──────────────┐
                                          │  train_processed.csv (DB lookup)│
                                          │  lgb_model_full.pkl             │
                                          │  lgb_model_lite.pkl             │
                                          └────────────────────────────────┘
```

---

## Model Performance

| Model | Use Case | F1 Score | ROC AUC | Threshold |
|---|---|---|---|---|
| **Model A** — Full | Existing applicants (200+ features + SMOTE) | **0.9547** | **0.9816** | 0.48 |
| **Model B** — Lite | New applicants (15 form features + SMOTE) | **0.9489** | **0.9711** | 0.38 |

> Both models use SMOTE oversampling applied to the training split to address the severe class imbalance (~8% default rate in the original dataset).

---

## Project Structure

```
CodeBase/
├── Data/
│   ├── application_train.csv       # Raw Home Credit training data (307K rows)
│   ├── processed/
│   │   └── train_processed.csv     # Feature-engineered dataset (210 cols)
│   └── ...                         # Other raw files (bureau, installments, etc.)
│
├── src/
│   └── train_model.py              # Dual-model training script (run this first!)
│
├── models/
│   ├── lgb_model_full.pkl          # Model A — full feature set
│   └── lgb_model_lite.pkl          # Model B — lite 15-feature set
│
├── backend/
│   ├── main.py                     # FastAPI server with /predict and /health
│   └── requirements.txt            # Backend-specific deps
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx                 # Main React component & form logic
│   │   └── index.css               # Glassmorphism design system
│   ├── package.json
│   └── vite.config.js
│
├── notebooks/                      # EDA and experimentation notebooks
├── requirements.txt                # Python dependencies (root — install this)
└── README.md
```

---

## Prerequisites

### System Requirements
- **Python 3.10+**
- **Node.js 18+** and **npm 9+**
- **~4 GB RAM** (for loading the processed dataset at runtime)
- **~2 GB Disk** (for the raw dataset + processed CSV + models)

### Checking your versions
```bash
python --version     # should be 3.10+
node --version       # should be 18+
npm --version        # should be 9+
```

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/avishkajindal05/the-invisible-risk.git
cd the-invisible-risk
```

### 2. Set up the Python environment

> **Recommended:** Use a virtual environment or Conda to avoid package conflicts.

**Option A — venv (standard)**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**Option B — Conda**
```bash
conda create -n credit-risk python=3.10
conda activate credit-risk
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

All required packages:

| Package | Purpose |
|---|---|
| `pandas` | Data loading & manipulation |
| `numpy` | Numerical computing |
| `scikit-learn` | Train/test split, metrics, imputation |
| `lightgbm` | Core prediction model |
| `imbalanced-learn` | SMOTE oversampling |
| `joblib` | Model serialization |
| `fastapi` | REST API backend |
| `uvicorn` | ASGI server for FastAPI |
| `pydantic` | Request schema validation |
| `xgboost` | (Optional) alternative model |
| `optuna` | Hyperparameter tuning |
| `shap` | Explainability (notebooks) |
| `matplotlib` | Plotting in notebooks |
| `sentence-transformers` | NLP feature engineering (notebooks) |
| `category_encoders` | Advanced categorical encoding |

### 4. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

---

## Training the Models

> **Required before running the app for the first time.** This generates `models/lgb_model_full.pkl` and `models/lgb_model_lite.pkl`.

```bash
python src/train_model.py
```

Expected output (takes ~2–4 minutes):
```
============================================================
  CREDIT DEFAULT — DUAL MODEL TRAINING
============================================================

[1/7] Loading processed data...
      Rows: 307,511  |  Columns: 210

[2/7] One-hot encoding 24 categorical features...
      Total encoded features: 381

[3/7] Imputing missing values with column median...

============================================================
  MODEL A — Full Features (Existing Applicants)
============================================================
[4/7] Applying SMOTE...
      After SMOTE: 565,372 samples
      [OK] ROC AUC:  0.9816
      [OK] F1 Score: 0.9547  (thresh=0.48)

============================================================
  MODEL B — Lite Features (New Applicants)
============================================================
      Lite encoded feature count: 41
[5/7] Applying SMOTE (lite)...
      After SMOTE: 565,372 samples
      [OK] ROC AUC:  0.9711
      [OK] F1 Score: 0.9489  (thresh=0.38)

[6/7] Saving models...
[7/7] Both models saved OK.

============================================================
  SUMMARY
  Model A (Full):  F1=0.9547  AUC=0.9816
  Model B (Lite):  F1=0.9489  AUC=0.9711
============================================================
```

---

## Running the Application

You need **two separate terminal windows** running simultaneously.

### Terminal 1 — Start the Backend API

```bash
cd backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Wait until you see:
```
INFO:     Application startup complete.
```

> **Note:** On first startup the backend loads `train_processed.csv` (~400 MB) into memory for fast lookups. This takes ~30 seconds.

### Terminal 2 — Start the Frontend

```bash
cd frontend
npm run dev
```

You will see:
```
  VITE v8.x.x  ready in Xms

  ➜  Local:   http://localhost:5173/
```

### Open in browser

➡️ **http://localhost:5173/**

---

## Using the Application

The form has two modes:

| Mode | When | Model Used |
|---|---|---|
| **Full Analysis** | Applicant ID provided and found in database | Model A (200+ features, highest accuracy) |
| **New Applicant** | Applicant ID left blank or not found | Model B (15 form features) |

### Form fields

| Field | Required | Description |
|---|---|---|
| Applicant ID | **Optional** | `SK_ID_CURR` from the Home Credit dataset |
| Contract Type | Yes | Cash loans or Revolving loans |
| Gender | Yes | M / F / XNA |
| Owns Car / Realty | Yes | Y / N |
| Children Count | Yes | Integer |
| Annual Income | Yes | In local currency |
| Credit Amount | Yes | Total loan amount |
| Annuity | Yes | Monthly repayment |
| Goods Price | Yes | Price of goods financed |
| Income Type | Yes | Working, Pensioner, etc. |
| Education | Yes | Highest level attained |
| Family Status | Yes | Married, Single, etc. |
| Housing Type | Yes | Own / Rented / With Parents, etc. |
| Age (DAYS_BIRTH) | Yes | Negative integer (days before today, e.g. `-12000`) |
| Employed since (DAYS_EMPLOYED) | Yes | Negative integer (e.g. `-2000`) |

---

## API Reference

### `POST /predict`

**Request body:**
```json
{
  "SK_ID_CURR": 100002,
  "NAME_CONTRACT_TYPE": "Cash loans",
  "CODE_GENDER": "F",
  "FLAG_OWN_CAR": "N",
  "FLAG_OWN_REALTY": "Y",
  "CNT_CHILDREN": 0,
  "AMT_INCOME_TOTAL": 200000,
  "AMT_CREDIT": 500000,
  "AMT_ANNUITY": 25000,
  "AMT_GOODS_PRICE": 450000,
  "NAME_INCOME_TYPE": "Working",
  "NAME_EDUCATION_TYPE": "Secondary / secondary special",
  "NAME_FAMILY_STATUS": "Married",
  "NAME_HOUSING_TYPE": "House / apartment",
  "DAYS_BIRTH": -12000,
  "DAYS_EMPLOYED": -2000
}
```

> `SK_ID_CURR` is optional. Omit it for new applicants.

**Response:**
```json
{
  "prediction": 0,
  "risk_level": "Low Risk",
  "probability": 0.278,
  "threshold": 0.38,
  "model_used": "Lite Model (15 form features, new applicant)",
  "has_history": false
}
```

### `GET /health`
```json
{
  "status": "ok",
  "model_full_loaded": true,
  "model_lite_loaded": true,
  "database_loaded": true,
  "database_size": 307511
}
```

Interactive API docs available at: **http://localhost:8000/docs**

---

## Data

This project uses the **[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)** dataset from Kaggle.

The raw files are **not included** in this repository due to size. Download them from Kaggle and place them in `Data/`:

```
Data/
├── application_train.csv
├── application_test.csv
├── bureau.csv
├── bureau_balance.csv
├── previous_application.csv
├── installments_payments.csv
├── credit_card_balance.csv
├── POS_CASH_balance.csv
└── processed/
    └── train_processed.csv   ← generated by the feature engineering pipeline
```

> If `train_processed.csv` has not been generated yet, run the feature engineering scripts in `src/` or `notebooks/` before training the models.

---


##  Tech Stack

| Layer | Technology |
|---|---|
| ML Models | LightGBM, Scikit-learn, SMOTE (imbalanced-learn) |
| Hyperparameter Tuning | Optuna |
| NLP Embeddings | SentenceBERT + PCA |
| Backend API | FastAPI + Uvicorn |
| Frontend | React 19 + Vite 8 |
| Notebooks | Jupyter (ipykernel) |
| Explainability | SHAP |

---

##  Troubleshooting

**`RuntimeError: Models not found`** — Run `python src/train_model.py` first. Both `.pkl` files must exist in `models/` before starting the API.

**`train_processed.csv not found`** — Run `python src/features/engineering_main.py` first to generate the processed dataset.

**Frontend can't reach the API** — Ensure the backend is running on port 8000 and CORS is not blocked. The API allows all origins by default in development.

**LightGBM column name error** — Column names are auto-sanitized (special characters replaced with `_`) in both training and inference. If you modify feature names, apply the same `re.sub(r'[^A-Za-z0-9_]', '_', col)` transformation.

---



##  Security Considerations

In financial systems, data security and regulatory compliance are critical design priorities. This application is intentionally architected to operate within a controlled, local environment rather than being deployed as a publicly accessible service.

For security reasons, such prediction systems are typically hosted on bank-managed infrastructure (e.g., internal servers or on-premise machines). This ensures that sensitive financial data and personally identifiable information (PII) remain within secure boundaries and are not exposed to external networks.

Key considerations include:

- **Data Privacy**: Customer data (income, credit history, personal details) is highly sensitive and must not be transmitted over public networks unnecessarily.
- **Regulatory Compliance**: Financial institutions must adhere to strict regulations (e.g., RBI guidelines, GDPR equivalents), which often mandate controlled data environments.
- **Reduced Attack Surface**: Avoiding public deployment minimizes risks such as API abuse, data breaches, and unauthorized access.
- **Secure Model Access**: Models and datasets are stored and accessed only within trusted systems, preventing model theft or reverse engineering.
- **Audit and Governance**: Internal deployment allows better monitoring, logging, and auditing of all prediction requests.

Therefore, in real-world banking scenarios, systems like this are deployed locally or within secure private networks, ensuring robust protection of both data and infrastructure.

---



<div align="center">
  <p>Built with ❤️ by Team Invincibles</p>
</div>
