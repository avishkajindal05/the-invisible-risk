# High-Level Design (HLD)
## The Invisible Risk — Financial Default Intelligence

> **Version:** 1.0 | **Date:** April 2026 | **Project:** Futurense AI Clinic, IIT Gandhinagar

---

## 1. Problem Statement

Traditional credit scoring systems (e.g., CIBIL) systematically exclude a large segment of the population — individuals with no credit card, no mortgage, and no prior loan history. This project addresses that gap by building a **behavioral risk model** using alternative signals from relational financial data to predict the probability of loan default.

The core business tension:

- **Approve too liberally** → increased non-performing loans and financial losses
- **Approve too conservatively** → fails the underserved population the system is meant to serve

The goal is a model that maximizes recall on defaults (catching genuine risk) while minimizing false positives (avoiding wrongful rejections) — calibrated via threshold tuning rather than raw accuracy.

---

## 2. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                   │
│                                                                     │
│  7 Raw CSV Tables (Home Credit Kaggle Dataset)                      │
│  ┌──────────────┐ ┌──────────┐ ┌──────────────────┐                │
│  │ application_ │ │ bureau   │ │ previous_        │                │
│  │ train/test   │ │ .csv     │ │ application.csv  │                │
│  └──────────────┘ └──────────┘ └──────────────────┘                │
│  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐    │
│  │ bureau_balance   │ │ installments_    │ │ POS_CASH_balance │    │
│  │ .csv             │ │ payments.csv     │ │ .csv             │    │
│  └──────────────────┘ └──────────────────┘ └──────────────────┘    │
│  ┌──────────────────┐                                               │
│  │ credit_card_     │                                               │
│  │ balance.csv      │                                               │
│  └──────────────────┘                                               │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING LAYER                        │
│                                                                     │
│  engineering_main.py (Master Orchestrator)                          │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐               │
│  │ engineering_ │ │ engineering_ │ │ engineering_ │               │
│  │ app.py       │ │ bureau.py    │ │ prev.py      │               │
│  │ (ratios,     │ │ (active/     │ │ (approval/   │               │
│  │  EXT sources)│ │  closed,DPD) │ │  refusal)    │               │
│  └──────────────┘ └──────────────┘ └──────────────┘               │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐               │
│  │ engineering_ │ │ engineering_ │ │ nlp_          │               │
│  │ installments │ │ pos_cc.py    │ │ embeddings.py│               │
│  │ .py (DPD/DBD)│ │ (utilization)│ │ (BERT→PCA32) │               │
│  └──────────────┘ └──────────────┘ └──────────────┘               │
│                                                                     │
│  categorical_encoding.py (OHE + Target/Freq Encoding)              │
│                                                                     │
│  Output: 210-feature train_processed.csv / test_processed.csv      │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        MODEL LAYER                                  │
│                                                                     │
│  ┌──────────────────────────────┐                                   │
│  │  LightGBM (Primary Model)    │                                   │
│  │  • Optuna hyperparameter     │                                   │
│  │    tuning                    │                                   │
│  │  • 5-Fold Stratified CV      │                                   │
│  │  • F1-optimal threshold      │                                   │
│  │    tuning                    │                                   │
│  └──────────────────────────────┘                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│  │ Logistic        │  │ Random Forest   │  │ SVM             │    │
│  │ Regression      │  │ (Baseline)      │  │ (Baseline)      │    │
│  │ (Baseline)      │  │                 │  │                 │    │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      INFERENCE LAYER (WIP)                          │
│                                                                     │
│  Flask API — app/app.py                                             │
│  • Accepts applicant feature payload                                │
│  • Returns default probability score                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Sources

The project uses the **Home Credit Default Risk** dataset (Kaggle), consisting of 7 relational tables:

| Table | Description | Key Signal |
|---|---|---|
| `application_train/test` | Primary applicant record | Income, credit amount, employment, EXT sources |
| `bureau.csv` | Historical bureau loans from Credit Bureau | Active/closed loan counts, overdue amounts |
| `bureau_balance.csv` | Monthly status of bureau loans | Payment status time series |
| `previous_application.csv` | Prior Home Credit loan applications | Approval/refusal patterns, capital ratios |
| `installments_payments.csv` | Installment payment history | Days Past Due (DPD), Days Before Due (DBD) |
| `POS_CASH_balance.csv` | Monthly POS and cash loan status | Active months, overdue DPD |
| `credit_card_balance.csv` | Monthly credit card balance | Utilization, draw-down reliance |

> **Design note:** Raw data is intentionally excluded from version control (`.gitignore`). Only processed feature files (`data/processed/`) are retained.

---

## 4. Feature Engineering Design

### 4.1 Philosophy

> *"In credit risk, missingness is not random. A missing bureau record often means zero prior credit history — itself a strong default signal."*

Missing values are **not imputed or dropped blindly**. Instead:
- Missing bureau records → encoded as "no credit history" signal
- Sparse columns are retained as informative absence indicators
- All feature engineering is modular, table-scoped, and orchestrated centrally

### 4.2 Feature Groups

| Module | Feature Category | Example Features |
|---|---|---|
| `engineering_app.py` | Application ratios & scores | `CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`, `EMPLOYED_RATIO`, `EXT_SOURCE_MEAN/PROD/STD` |
| `engineering_bureau.py` | Bureau history aggregates | Active vs. closed loan counts, current debt sums, overdue loan counts |
| `engineering_prev.py` | Previous application behavior | Approval/refusal counts, requested vs. approved capital ratios |
| `engineering_installments.py` | Repayment behavior | DPD (Days Past Due), DBD (Days Before Due), payment consistency metrics |
| `engineering_pos_cc.py` | POS & credit card usage | Utilization flags, max draw reliance, active month counts |
| `nlp_embeddings.py` | Text-based financial signals | SentenceBERT embeddings → PCA reduced to 32 dimensions |
| `categorical_encoding.py` | Categorical feature handling | One-Hot (low cardinality), Target/Frequency encoding (high cardinality) |

**Total features engineered: 210**

### 4.3 NLP Pipeline

Financial narrative fields are transformed into dense vector representations:
1. Raw text fields processed through **SentenceBERT**
2. High-dimensional embeddings reduced via **PCA to 32 components**
3. Resulting 32 features appended to the main feature matrix

---

## 5. Modeling Architecture

### 5.1 Primary Model — LightGBM

LightGBM is chosen as the primary model due to:
- Efficient handling of large, sparse tabular datasets
- Native support for categorical features
- Strong performance on imbalanced classification tasks

**Training pipeline:**
- **Hyperparameter tuning:** Optuna (Bayesian optimization)
- **Validation strategy:** 5-Fold Stratified Cross-Validation (preserves class imbalance across folds)
- **Threshold selection:** F1-optimal threshold tuning post-training (not default 0.5 cutoff)

### 5.2 Baselines

| Model | Purpose |
|---|---|
| Logistic Regression | Linear baseline; interpretability benchmark |
| Random Forest | Non-linear ensemble baseline |
| SVM | Margin-based baseline |

### 5.3 Evaluation Metrics

| Metric | Rationale |
|---|---|
| **ROC-AUC** | Measures overall discriminative ability across all thresholds |
| **Recall (Defaults)** | Primary business metric — catching actual defaulters is critical |
| **F1-Score** | Balances precision and recall at the chosen operating threshold |

> **Why not accuracy?** The dataset is class-imbalanced (defaults are minority class). A model predicting "no default" for all applicants would achieve high accuracy but zero utility.

### 5.4 Model Results

| Model | ROC-AUC | Recall (Defaults) | F1-Score |
|---|---|---|---|
| Logistic Regression | 0.771 | 0.61 | 0.58 |
| Random Forest | 0.758 | 0.61 | 0.57 |
| **LightGBM** | **0.786** | 0.487 | 0.340 |
| SVM | 0.750 | 0.374 | 0.61 |

> **Note:** LightGBM achieves the highest AUC (best discrimination), but its recall on defaults is lower — indicating scope for threshold adjustment or class-weight tuning in future iterations.

---

## 6. Inference API (WIP)

A lightweight **Flask API** (`app/app.py`) is under development to expose the trained model as an HTTP endpoint:

- **Input:** JSON payload with applicant feature values
- **Output:** Predicted default probability score (0–1)
- **Status:** Work in progress; not yet production-ready

---

## 7. Key Design Decisions

| Decision | Rationale |
|---|---|
| Retain sparse/missing columns | Missingness carries signal (e.g., no bureau = no credit history) |
| Modular feature engineering | Each table processed independently; easy to swap or extend |
| Stratified K-Fold CV | Preserves default/non-default ratio in every validation fold |
| F1-optimal threshold | Business-appropriate cutoff vs. arbitrary 0.5 threshold |
| SentenceBERT + PCA | Captures semantic signal from narrative fields without dimensionality explosion |
| Target encoding for high-cardinality | Avoids one-hot explosion for fields with many unique categories |

---

## 8. Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.9+ |
| ML Framework | LightGBM, scikit-learn |
| Hyperparameter Tuning | Optuna |
| NLP | SentenceBERT (sentence-transformers) |
| Dimensionality Reduction | scikit-learn PCA |
| Inference API | Flask |
| Notebooks | Jupyter |
| Dependency Management | pip / requirements.txt |

---

## 9. Repository Layout (Summary)

```
the-invisible-risk/
├── data/
│   ├── raw/            # Source CSVs (not committed)
│   └── processed/      # Engineered feature files (210 features)
├── src/
│   ├── data/           # Data loader
│   └── features/       # Feature engineering modules + encoding + NLP
├── src/models/         # Training, evaluation scripts
├── Notebooks/          # EDA, baseline model notebooks
├── app/                # Flask inference API
├── docs/               # HLD.md, LLD.md
└── tests/              # Preprocessing unit tests
```

---

## 10. Future Scope

- **Model improvement:** Class-weight tuning and SMOTE oversampling to improve default recall in LightGBM
- **Feature selection:** SHAP-based feature importance analysis to reduce 210 features to a compact, interpretable set
- **API completion:** Full Flask API with input validation, model versioning, and response schema
- **Monitoring:** Data drift detection for production deployment
- **Explainability:** SHAP waterfall plots per applicant decision for regulatory compliance

---

