# 🏦 The Invisible Risk — Financial Default Intelligence

> Moving beyond credit scores to behavioral risk modeling for underserved borrowers.

**Futurense AI Clinic | IIT Gandhinagar | April 2026**

---

## 📌 Problem Statement

Traditional credit scoring excludes millions of borrowers with no CIBIL score, no credit card, and no mortgage history. This project builds a behavioral risk model using alternative signals — application patterns, employment history, bureau data, installment behavior, and credit card usage — to predict loan default probability.

**Business tension:** Approve too many → losses spike. Reject too many → you fail the population you exist to serve.

---

## 🗂️ Repository Structure

```
the-invisible-risk/
│
├── data/
│   ├── raw/                    # Original Home Credit CSVs (not committed — see below)
│   └── processed/              # train_processed.csv, test_processed.csv (210 features)
│
├── src/
│   ├── data/
│   │   └── loader.py           # Loads all 7 raw source tables
│   └── features/
│       ├── engineering_main.py      # Master orchestrator — runs full pipeline
│       ├── engineering_app.py       # Application table features (ratios, age, ext sources)
│       ├── engineering_bureau.py    # Bureau history (active/closed loans, debt sums)
│       ├── engineering_prev.py      # Previous applications (approval/refusal patterns)
│       ├── engineering_installments.py  # Payment behavior (DPD, DBD metrics)
│       ├── engineering_pos_cc.py    # POS cash & credit card utilization
│       ├── categorical_encoding.py  # One-hot + target/frequency encoding
│       └── nlp_embeddings.py        # SentenceBERT → PCA(32) financial narratives
│
├── src/models/
│   ├── train_lgbm.py           # Optuna-tuned LightGBM with 5-Fold Stratified CV
│   ├── evaluate_lgbm.py        # Evaluation with threshold tuning (F1-optimal)
│   ├── evaluate_lgbm_pca.py    # LightGBM + PCA variant
│   └── evaluate_svm.py         # SVM baseline
│
├── Notebooks/
│   ├── preprocess/
│   │   └── EDA_train_processed.ipynb   # Full EDA: missing values, correlations, distributions
│   └── models/
│       ├── baseline_logistic_regression.ipynb
│       └── random_forest.ipynb
│
├── app/
│   └── app.py                  # Flask inference API (WIP)
│
├── docs/
│   ├── HLD.md                  # High-Level Design: architecture + metric rationale
│   └── LLD.md                  # Low-Level Design: feature specs + missing value strategy
│
├── tests/
│   └── test_preprocessing.py
│
├── PROMPTS.md                  # AI tool usage log
├── requirements.txt
└── .gitignore
```

---

## ⚙️ Setup & Installation

**Prerequisites:** Python 3.9+

```bash
# 1. Clone the repo
git clone https://github.com/avishkajindal05/the-invisible-risk.git
cd the-invisible-risk

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 📦 Data Setup

Download the raw data from the [Home Credit Default Risk Kaggle competition](https://www.kaggle.com/competitions/home-credit-default-risk/data) and place all CSV files inside `data/raw/`.

Required files:
```
data/raw/
├── application_train.csv
├── application_test.csv
├── bureau.csv
├── bureau_balance.csv
├── previous_application.csv
├── installments_payments.csv
├── POS_CASH_balance.csv
└── credit_card_balance.csv
```

> ⚠️ Raw data files are **not committed** to this repository (see `.gitignore`). They must be downloaded separately.

---

## 📊 Model Results

| Model | ROC-AUC | Recall (Defaults) | F1-Score |
|---|---|---|---|
| Logistic Regression | 0.771 | 0.61 | 0.58 |
| Random Forest | 0.758 | 0.61 | 0.57 |
| LightGBM | 0.786 | 0.487 | 0.340 |
| SVM | 0.750 | 0.374 | 0.61 |

---

## 🧠 Feature Engineering Summary

210 features engineered from 7 relational tables:

- **Application ratios** — `CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`, `EMPLOYED_RATIO`, `EXT_SOURCE_MEAN/PROD/STD`
- **Bureau aggregates** — active vs closed loans, current debt sums, overdue counts
- **Previous application patterns** — approval/refusal counts, requested vs approved capital ratios
- **Installment behavior** — Days Past Due (DPD), Days Before Due (DBD), payment consistency
- **POS & Credit Card** — utilization flags, max draw reliance, active month counts
- **NLP embeddings** — SentenceBERT on financial narratives → PCA reduced to 32 dimensions
- **Categorical encoding** — One-Hot for low-cardinality, Target/Frequency encoding for high-cardinality fields

---

## 🔑 Key Design Decisions

**Why not drop sparse columns?** In credit risk, missingness is not random. A missing bureau record often means zero prior credit history — itself a strong default signal.
