#  **THE INVISIBLE RISK**

Financial Default Intelligence

#  **LOW-LEVEL DESIGN DOCUMENT**

Version 1.0 | April 2026

Futurense AI Clinic | IIT Gandhinagar

# **1\. Document Overview**

This Low-Level Design (LLD) document specifies the internal logic, data contracts, function signatures, and design decisions for every module in The Invisible Risk project - a behavioral credit default prediction system built on the Home Credit Default Risk dataset.

It is intended as a technical reference for developers, reviewers, and contributors who need to understand how each component works at the implementation level.

## **1.1 Scope**

- Data loading and schema normalization (src/data/loader.py)
- Six feature engineering modules (src/features/)
- Dual-model training pipeline (src/train_model.py)
- Optuna-tuned evaluation harness (src/models/)
- FastAPI inference backend (backend/main.py)
- React + Vite frontend (frontend/src/)

## **1.2 System Context Summary**

The system ingests 7 relational CSV tables from Kaggle's Home Credit competition. A feature engineering pipeline joins and transforms these into a 210-feature flat matrix. Two LightGBM models are trained: Model A (200+ features, for applicants with prior credit history) and Model B (15 form fields, for new applicants). The FastAPI backend loads both at startup and routes each inference request to the appropriate model. A React frontend provides a form-based UI.

# **2\. Data Layer - src/data/loader.py**

## **2.1 Module Purpose**

Centralises all raw CSV file reads into typed loader functions. Each function returns a fully pre-processed DataFrame ready for the corresponding feature engineering module.

## **2.2 Configuration**

**DATA_DIR** is defined at module level as the root directory of the raw CSVs. Change this to Data/raw/ relative to the project root for portable usage.

## **2.3 Function Specifications**

### **load_application_data(is_train: bool = True) → pd.DataFrame**

Loads application_train.csv or application_test.csv. Applies the following transformations in-place before returning:

- EXT_SOURCE_1/2/3 - Creates binary missing-flag columns (e.g. EXT_SOURCE_1_MISSING = 1), then imputes with per-column median.
- OCCUPATION_TYPE, ORGANIZATION_TYPE - Missing values filled with string literal 'Unknown'.
- DAYS_EMPLOYED - Positive values indicate retirees (Kaggle artifact). IS_RETIRED flag is created (1 if DAYS_EMPLOYED > 0), then positive values are clamped to 0.
- TARGET column is retained when is_train=True; absent in test mode.

| **Column**        | **Type** | **Missing Strategy** | **Note**              |
| ----------------- | -------- | -------------------- | --------------------- |
| EXT_SOURCE_1/2/3  | float64  | Median imputation    | Missing flag created  |
| OCCUPATION_TYPE   | object   | 'Unknown' fill       | High cardinality      |
| ORGANIZATION_TYPE | object   | 'Unknown' fill       | High cardinality      |
| DAYS_EMPLOYED     | int64    | Clamp > 0 to 0       | IS_RETIRED flag added |
| TARGET            | int64    | N/A - label only     | Train set only        |

### **load_bureau_data() → Tuple\[pd.DataFrame, pd.DataFrame\]**

Returns (bureau, bureau_balance) as a tuple. No pre-processing; raw join keys SK_ID_BUREAU and SK_ID_CURR are preserved for downstream aggregation.

### **load_previous_applications() / load_installments_payments() / load_pos_cash_balance() / load_credit_card_balance()**

Each returns the corresponding raw DataFrame. No transformations applied - all feature logic is in the corresponding engineering module.

# **3\. Feature Engineering - src/features/**

## **3.1 Orchestrator - engineering_main.py**

### **build_features(is_train: bool = True) → pd.DataFrame**

Master function that calls all loaders and engineering modules in dependency order, performs left-merges on SK_ID_CURR, and writes the result to Data/processed/train_processed.csv or test_processed.csv.

Merge order and join semantics:

| **Step** | **Source**                                          | **Join Key**     | **Join Type** | **Resulting Columns Added** |
| -------- | --------------------------------------------------- | ---------------- | ------------- | --------------------------- |
| 1        | application_train/test → engineering_app.py         | N/A (base table) | -             | ~40 engineered ratios       |
| 2        | bureau + bureau_balance → engineering_bureau.py     | SK_ID_CURR       | LEFT          | ~22 bureau aggregates       |
| 3        | previous_application → engineering_prev.py          | SK_ID_CURR       | LEFT          | ~15 prev-app features       |
| 4        | installments_payments → engineering_installments.py | SK_ID_CURR       | LEFT          | ~6 payment features         |
| 5        | POS_CASH_balance → engineering_pos_cc.py            | SK_ID_CURR       | LEFT          | ~5 POS features             |
| 6        | credit_card_balance → engineering_pos_cc.py         | SK_ID_CURR       | LEFT          | ~5 CC features              |

All left-merges result in NaN for applicants with no history in that table. Downstream training handles these with median imputation. This is intentional - missing bureau data is a strong default signal.

## **3.2 Application Features - engineering_app.py**

### **process_application_features(df: pd.DataFrame) → pd.DataFrame**

Constructs ~40 numeric features directly from the application table. All divisions use safe_div(a, b) which replaces 0 denominators with NaN to avoid inf values.

| **Feature Name**      | **Formula**                                | **Rationale**                    |
| --------------------- | ------------------------------------------ | -------------------------------- |
| CREDIT_INCOME_RATIO   | AMT_CREDIT / AMT_INCOME_TOTAL              | Debt burden relative to income   |
| ANNUITY_INCOME_RATIO  | AMT_ANNUITY / AMT_INCOME_TOTAL             | Monthly repayment feasibility    |
| CREDIT_TERM           | AMT_ANNUITY / AMT_CREDIT                   | Loan duration proxy              |
| GOODS_CREDIT_RATIO    | AMT_GOODS_PRICE / AMT_CREDIT               | Collateral coverage ratio        |
| AGE_YEARS             | \|DAYS_BIRTH\| / 365.25                    | Applicant age in years           |
| EMPLOYMENT_YEARS      | \|DAYS_EMPLOYED\| / 365.25                 | Employment tenure                |
| EMPLOYED_RATIO        | EMPLOYMENT_YEARS / AGE_YEARS               | Career stability indicator       |
| INCOME_PER_PERSON     | AMT_INCOME_TOTAL / CNT_FAM_MEMBERS         | Household income per head        |
| CHILDREN_RATIO        | CNT_CHILDREN / CNT_FAM_MEMBERS             | Dependency ratio                 |
| EXT_SOURCE_MEAN       | mean(EXT_SOURCE_1,2,3)                     | Composite external credit score  |
| EXT_SOURCE_PROD       | EXT_SOURCE_1 × 2 × 3                       | Non-linear score interaction     |
| EXT_SOURCE_STD        | std(EXT_SOURCE_1,2,3)                      | Score consistency across bureaus |
| EXT1_EXT2_INTERACTION | EXT_SOURCE_1 × EXT_SOURCE_2                | Bureau cross-signal              |
| DOCUMENT_COUNT        | sum(FLAG_DOCUMENT_X)                       | Number of documents submitted    |
| TOTAL_ENQUIRIES       | sum(AMT*REQ_CREDIT_BUREAU*\*)              | Credit enquiry intensity         |
| RECENT_ENQUIRY_RATIO  | WEEK_ENQUIRIES / TOTAL_ENQUIRIES           | Recent vs total enquiry ratio    |
| HAS_CAR_REALTY        | FLAG_OWN_CAR == Y AND FLAG_OWN_REALTY == Y | Asset ownership indicator        |
| CREDIT_DOWNPAYMENT    | AMT_GOODS_PRICE − AMT_CREDIT               | Down payment amount              |

## **3.3 Bureau Features - engineering_bureau.py**

### **process_bureau_features(bureau: pd.DataFrame, bureau_balance: pd.DataFrame) → pd.DataFrame**

Two-stage aggregation: bureau_balance is first aggregated per SK_ID_BUREAU (loan level), then bureau is aggregated per SK_ID_CURR (applicant level). Returns ~22 features.

### **Stage 1 - bureau_balance aggregation (per SK_ID_BUREAU)**

STATUS column is mapped: 'C' → 0 (closed, safe), 'X' → 0 (unknown), integers 1-5 are delinquency levels and kept as-is. Two features produced:

- BUREAU_STATUS_WORST_PER_BUREAU - max STATUS per loan (worst delinquency ever seen)
- BUREAU_STATUS_MEAN_PER_BUREAU - mean STATUS per loan

### **Stage 2 - bureau aggregation (per SK_ID_CURR)**

| **Output Feature**              | **Source Columns**                   | **Aggregation**                 |
| ------------------------------- | ------------------------------------ | ------------------------------- |
| BUREAU_COUNT                    | SK_ID_BUREAU                         | count - total credit lines      |
| BUREAU_ACTIVE_COUNT             | CREDIT_ACTIVE_BINARY                 | sum - count of active loans     |
| BUREAU_DAYS_CREDIT_MEAN/MIN/MAX | DAYS_CREDIT                          | mean, min, max                  |
| BUREAU_AMT_CREDIT_SUM_SUM       | AMT_CREDIT_SUM                       | sum - total credit exposure     |
| BUREAU_DEBT_CREDIT_RATIO_MEAN   | AMT_CREDIT_SUM_DEBT / AMT_CREDIT_SUM | mean debt burden ratio          |
| BUREAU_STATUS_WORST_MAX         | BUREAU_STATUS_WORST_PER_BUREAU       | max - worst-ever delinquency    |
| LAST_ACTIVE_DAYS_CREDIT         | DAYS_CREDIT (active only)            | max - most recent active credit |

## **3.4 Previous Application Features - engineering_prev.py**

### **process_previous_application_features(prev: pd.DataFrame) → pd.DataFrame**

Sorted by DAYS_DECISION descending per applicant. A LAG_IDX column (1 = most recent) enables sliced aggregates.

| **Output Feature**         | **Formula / Source**                  | **Interpretation**          |
| -------------------------- | ------------------------------------- | --------------------------- |
| PREV_COUNT                 | count SK_ID_PREV                      | Total prior applications    |
| PREV_APPROVED_RATE         | mean(APPROVED flag)                   | Historical approval rate    |
| PREV_REFUSED_RATE          | mean(REFUSED flag)                    | Historical refusal rate     |
| PREV_APP_CREDIT_RATIO_MEAN | mean(AMT_APPLICATION / AMT_CREDIT)    | Requested vs granted ratio  |
| PREV_CREDIT_MEAN_LAST3     | mean AMT_CREDIT, LAG ≤ 3              | Recent credit amounts       |
| PREV_CREDIT_MEAN_LAST5     | mean AMT_CREDIT, LAG ≤ 5              | Medium-term credit amounts  |
| PREV_STATUS_LAG_1..5       | pivot NAME_CONTRACT_STATUS by LAG_IDX | Last 5 application outcomes |

## **3.5 Installment Features - engineering_installments.py**

### **process_installments_features(inst: pd.DataFrame) → pd.DataFrame**

| **Derived Column** | **Formula**                          | **Feature Produced**                                  |
| ------------------ | ------------------------------------ | ----------------------------------------------------- |
| PAYMENT_DIFF       | AMT_INSTALMENT − AMT_PAYMENT         | INST_PAYMENT_DIFF_MEAN/MAX - underpayment magnitude   |
| DAYS_ENTRY_DIFF    | DAYS_INSTALMENT − DAYS_ENTRY_PAYMENT | Negative = late payment                               |
| LATE_PAYMENT       | DAYS_ENTRY_DIFF < 0                  | INST_LATE_PAYMENT_RATE - fraction of late payments    |
| SHORT_PAYMENT      | PAYMENT_DIFF > 0                     | INST_SHORT_PAYMENT_RATE - fraction of under-payments  |
| 365-day window     | DAYS_INSTALMENT >= −365              | INST_PAYMENT_DIFF_MEAN_365 - recent payment behaviour |

## **3.6 POS Cash & Credit Card Features - engineering_pos_cc.py**

### **process_pos_cash_features(pos: pd.DataFrame) → pd.DataFrame**

| **Feature**             | **Formula**         | **Rationale**                   |
| ----------------------- | ------------------- | ------------------------------- |
| POS_MONTHS_COUNT        | count SK_ID_PREV    | Depth of POS credit history     |
| POS_SK_DPD_MEAN/MAX     | mean/max SK_DPD     | POS delinquency level           |
| POS_DPD_RATE            | mean HAS_DPD flag   | Fraction of months with DPD > 0 |
| POS_CNT_INSTALMENT_MEAN | mean CNT_INSTALMENT | Average loan term in months     |

### **process_credit_card_features(cc: pd.DataFrame) → pd.DataFrame**

| **Feature**           | **Formula**                                    | **Rationale**               |
| --------------------- | ---------------------------------------------- | --------------------------- |
| CC_COUNT              | count SK_ID_PREV                               | CC history depth            |
| CC_UTIL_RATE_MEAN/MAX | AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL          | Credit utilisation ratio    |
| CC_DRAWING_RATE_MEAN  | AMT_DRAWINGS_CURRENT / AMT_CREDIT_LIMIT_ACTUAL | Cash draw reliance          |
| CC_AMT_BALANCE_MEAN   | mean AMT_BALANCE                               | Average outstanding balance |
| CC_DPD_MEAN           | mean SK_DPD                                    | CC delinquency level        |

## **3.7 Categorical Encoding - categorical_encoding.py**

### **apply_categorical_encodings(train_df, test_df) → Tuple\[pd.DataFrame, pd.DataFrame\]**

Applied after the merge pipeline. Three distinct encoding strategies are used based on cardinality and leakage risk:

| **Column**                | **Strategy**                                 | **Output**                             | **Reason**                                                    |
| ------------------------- | -------------------------------------------- | -------------------------------------- | ------------------------------------------------------------- |
| OCCUPATION_TYPE           | K-Means (k=6) on {default_rate, income_mean} | OCCUPATION_CLUSTER (int)               | Groups occupations by risk behaviour, not label               |
| ORGANIZATION_TYPE         | Target encoding + frequency encoding         | ORG_TYPE_TARGET_ENC, ORG_TYPE_FREQ_ENC | High cardinality - label encoding would impose spurious order |
| All remaining object cols | LabelEncoder (fit on train+test union)       | int-encoded in-place                   | Low cardinality; consistent mapping across splits             |

Target encoding is fit exclusively on training data to prevent leakage. The fitted encoder is then transform-only on test data.

## **3.8 NLP Embeddings - nlp_embeddings.py**

### **process_nlp_embeddings(df: pd.DataFrame) → pd.DataFrame**

Converts structured applicant fields into natural-language narratives, then encodes them with a pre-trained sentence embedding model.

### **generate_financial_narratives(df) → List\[str\]**

Iterates over rows and constructs a fixed-template sentence for each applicant using: DAYS_BIRTH, AMT_INCOME_TOTAL, EXT_SOURCE_MEAN, DAYS_EMPLOYED, CNT_CHILDREN, FLAG_OWN_REALTY.

Example output narrative:

**Example Narrative**

The applicant is 34 years old with an annual income of 202500. They have 0 children. Employment history is 7 years. External credit assessment is 0.621. Real estate ownership: Y.

Pipeline:

- Encode all narratives with SentenceTransformer('all-MiniLM-L6-v2') in batch_size=128 - outputs 384-dimensional float vectors
- Apply PCA(n_components=32) to reduce to 32 dimensions
- Append columns NLP_EMB_0 through NLP_EMB_31 to the dataframe

Note: batch encoding is GPU-friendly. On CPU, 307K rows takes ~15-20 minutes. The NLP module is called optionally and its columns are included in the full feature set.

# **4\. Training Pipeline - src/train_model.py**

## **4.1 Entry Point - train()**

The primary training script. Produces two serialised model artifacts. Execution steps:

| **Step** | **Action**         | **Detail**                                                                          |
| -------- | ------------------ | ----------------------------------------------------------------------------------- |
| 1/7      | Load data          | Read Data/processed/train_processed.csv                                             |
| 2/7      | One-hot encode     | pd.get_dummies on all object columns; column names sanitized (special chars → '\_') |
| 3/7      | Median imputation  | Fill NaN with per-column median; zero-fill any remaining NaN                        |
| 4/7      | SMOTE - Full model | Oversample minority class to balance; fit on full encoded X                         |
| 5/7      | SMOTE - Lite model | Subset to 15 form-field columns and their one-hot expansions; SMOTE again           |
| 6/7      | Train both models  | LGBMClassifier with fixed hyperparameters; 80/20 random split                       |
| 7/7      | Save artifacts     | joblib.dump to models/lgb_model_full.pkl and models/lgb_model_lite.pkl              |

## **4.2 Column Sanitization**

LightGBM rejects column names with special characters. The sanitize() function applies:

re.sub(r'\[^A-Za-z0-9\_\]', '\_', col)

This same regex is applied in both train_model.py and backend/main.py - the inference row builder must produce identical column names to what the model was trained on.

## **4.3 LightGBM Hyperparameters (Fixed in train_model.py)**

| **Parameter**     | **Value** | **Effect**                             |
| ----------------- | --------- | -------------------------------------- |
| n_estimators      | 400       | Number of boosting rounds              |
| num_leaves        | 96        | Tree complexity - controls overfitting |
| learning_rate     | 0.05      | Step size for gradient descent         |
| min_child_samples | 20        | Minimum samples per leaf               |
| subsample         | 0.8       | Row sampling ratio per tree            |
| colsample_bytree  | 0.8       | Feature sampling ratio per tree        |
| random_state      | 42        | Reproducibility seed                   |

## **4.4 Threshold Tuning**

After training, the F1-optimal threshold is computed by sweeping thresholds from 0.30 to 0.70 in steps of 0.02, evaluating F1 at each step on the held-out test set. The best threshold is stored inside the model artifact:

joblib.dump({'model': clf_full, 'features': feature_names_full, 'best_thresh': thresh_f}, ...)

This means the inference server uses the same threshold that was validated during training - no hardcoding in the API.

## **4.5 Lite Model Feature Selection**

The 15 raw lite features are:

- NAME_CONTRACT_TYPE, CODE_GENDER, FLAG_OWN_CAR, FLAG_OWN_REALTY (categorical - one-hot expanded)
- CNT_CHILDREN, AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY, AMT_GOODS_PRICE (numeric)
- NAME_INCOME_TYPE, NAME_EDUCATION_TYPE, NAME_FAMILY_STATUS, NAME_HOUSING_TYPE (categorical - one-hot expanded)
- DAYS_BIRTH, DAYS_EMPLOYED (numeric)

All one-hot columns whose prefix matches a sanitized lite feature name are included, giving ~30-40 encoded columns for Model B.

## **4.6 Optuna Tuner - src/models/train_lgbm.py**

An alternative training script using Optuna for hyperparameter search. Uses 3-fold StratifiedKFold CV inside each Optuna trial. Search space:

| **Parameter**          | **Type**    | **Range**          |
| ---------------------- | ----------- | ------------------ |
| boosting_type          | categorical | \['gbdt', 'dart'\] |
| num_leaves             | int         | 31 - 128           |
| learning_rate          | float (log) | 0.01 - 0.1         |
| feature_fraction       | float       | 0.6 - 1.0          |
| bagging_fraction       | float       | 0.6 - 1.0          |
| min_child_samples      | int         | 5 - 100            |
| reg_alpha / reg_lambda | float (log) | 1e-4 - 10.0        |
| max_depth              | int         | 4 - 10             |

# **5\. Evaluation Harness - src/models/**

## **5.1 evaluate_lgbm.py - evaluate_model()**

80/20 stratified split (not SMOTE - evaluates on natural class distribution). LightGBM trained with scale_pos_weight to handle imbalance. Uses early stopping (150 rounds patience, 3000 max boosting rounds).

Evaluation metrics reported:

- ROC-AUC - primary metric
- Optimal threshold sweep from 0.10 to 0.90 in steps of 0.05
- Accuracy, Precision, Recall, F1 at best threshold
- Confusion matrix and full classification report

## **5.2 evaluate_lgbm_pca.py**

Identical to evaluate_lgbm.py but applies PCA before training to study the trade-off between dimensionality reduction and predictive power. Useful for benchmarking against the NLP embedding approach.

## **5.3 evaluate_svm.py - SVM Baseline**

Provides a linear SVM baseline for comparison. Trained on the same stratified split. Results are logged to src/models/model_res.txt and model_res_optimized.txt.

## **5.4 Model Results**

| **Model**           | **ROC-AUC** | **Recall (Defaults)** | **F1-Score** | **Notes**              |
| ------------------- | ----------- | --------------------- | ------------ | ---------------------- |
| Logistic Regression | 0.771       | 0.61                  | 0.58         | Baseline - good recall |
| Random Forest       | 0.758       | 0.61                  | 0.57         | Similar to LR          |
| LightGBM (Full)     | 0.786       | 0.487                 | 0.340        | Best AUC, lower recall |
| SVM                 | 0.750       | 0.374                 | 0.61         | Best F1, poor recall   |

# **6\. Backend API - backend/main.py**

## **6.1 Framework & Startup**

FastAPI application with CORS middleware (all origins permitted for development). On startup, the app loads both model artifacts and optionally the full processed dataset for applicant lookup.

| **Asset**           | **Path**                           | **Loaded Into**                      |
| ------------------- | ---------------------------------- | ------------------------------------ |
| lgb_model_full.pkl  | models/lgb_model_full.pkl          | model_full (global)                  |
| lgb_model_lite.pkl  | models/lgb_model_lite.pkl          | model_lite (global)                  |
| train_processed.csv | Data/processed/train_processed.csv | df_processed (indexed by SK_ID_CURR) |

If the models do not exist, startup raises RuntimeError with an explicit message directing the user to run the training script. If train_processed.csv is absent, a warning is logged and the server continues in new-applicant mode.

## **6.2 Data Schemas**

### **PredictRequest (Pydantic BaseModel)**

| **Field**                                                                       | **Type**        | **Required** | **Notes**                                      |
| ------------------------------------------------------------------------------- | --------------- | ------------ | ---------------------------------------------- |
| SK_ID_CURR                                                                      | Optional\[int\] | No           | If provided and found in DB, routes to Model A |
| NAME_CONTRACT_TYPE                                                              | str             | Yes          | e.g. 'Cash loans'                              |
| CODE_GENDER                                                                     | str             | Yes          | 'M' or 'F'                                     |
| FLAG_OWN_CAR / FLAG_OWN_REALTY                                                  | str             | Yes          | 'Y' or 'N'                                     |
| CNT_CHILDREN                                                                    | int             | Yes          | Number of children                             |
| AMT_INCOME_TOTAL / AMT_CREDIT / AMT_ANNUITY / AMT_GOODS_PRICE                   | float           | Yes          | Currency amounts                               |
| NAME_INCOME_TYPE / NAME_EDUCATION_TYPE / NAME_FAMILY_STATUS / NAME_HOUSING_TYPE | str             | Yes          | Category strings                               |
| DAYS_BIRTH / DAYS_EMPLOYED                                                      | int             | Yes          | Negative = days before application             |

## **6.3 Inference Routing Logic - POST /predict**

The routing decision is made by a single boolean has_history:

has_history = (SK_ID_CURR is not None AND SK_ID_CURR > 0 AND df_processed is not None AND SK_ID_CURR in df_processed.index)

| **Condition**       | **Model Used**                | **Feature Source**                                        |
| ------------------- | ----------------------------- | --------------------------------------------------------- |
| has_history = True  | Model A (full, 200+ features) | Historical record from df_processed, form values override |
| has_history = False | Model B (lite, 15 fields)     | Form values only                                          |

## **6.4 Feature Row Construction - build_inference_row()**

### **build_inference_row(raw_dict: dict, expected_features: list) → np.ndarray**

Converts a flat dictionary of raw field values into a one-hot encoded numpy row aligned to the model's expected feature list. Algorithm:

- Initialise a zero-filled dict keyed by all expected feature names
- For each input key-value pair: sanitize the key (same regex as training)
  - If the sanitized key exists directly → row\[skey\] = float(value)
  - Else → try one-hot lookup: row\['{skey}\_{sanitized(value)}'\] = 1.0
- Return np.array of values in expected_features order

## **6.5 Response Schema**

| **Field**   | **Type**     | **Description**                                                          |
| ----------- | ------------ | ------------------------------------------------------------------------ |
| prediction  | int (0 or 1) | 0 = Low Risk, 1 = High Risk                                              |
| risk_level  | str          | 'Low Risk' or 'High Risk'                                                |
| probability | float        | Raw model probability (0-1), rounded to 4 d.p.                           |
| threshold   | float        | Decision threshold loaded from model artifact                            |
| model_used  | str          | Human-readable label including model type and applicant ID if applicable |
| has_history | bool         | Whether historical data was found and used                               |

## **6.6 Health Endpoint - GET /health**

Returns JSON with four boolean flags (model_full_loaded, model_lite_loaded, database_loaded) and database_size (int). Useful for startup verification and monitoring.

# **7\. Frontend - frontend/src/**

## **7.1 Stack**

| **Technology**   | **Version** | **Role**                     |
| ---------------- | ----------- | ---------------------------- |
| React            | 19.2.4      | UI component framework       |
| Vite             | 8.0.4       | Dev server and bundler       |
| JavaScript (JSX) | -           | Component language           |
| CSS              | -           | Styling (App.css, index.css) |

## **7.2 Component Structure**

### **App.jsx - Main Component**

Single-page application. Manages form state for all 15 PredictRequest fields plus the optional SK_ID_CURR lookup field. On submit, constructs the POST /predict request body and renders the response.

Key UI states:

- Idle - form visible, no result panel
- Loading - spinner shown, form disabled
- Result - risk level, probability gauge, and model_used label displayed
- Error - API error message displayed inline

## **7.3 API Integration**

The frontend calls the backend at <http://localhost:8000/predict>. In production, this base URL should be configured via an environment variable (VITE_API_URL).

## **7.4 Build & Serve**

| **Command**     | **Purpose**                                         |
| --------------- | --------------------------------------------------- |
| npm run dev     | Vite dev server on <http://localhost:5173> with HMR |
| npm run build   | Production bundle to frontend/dist/                 |
| npm run preview | Preview production build locally                    |
| npm run lint    | ESLint check                                        |

# **8\. Error Handling & Edge Cases**

| **Scenario**                                    | **Where Handled**                        | **Behaviour**                                     |
| ----------------------------------------------- | ---------------------------------------- | ------------------------------------------------- |
| Division by zero in feature engineering         | safe_div() in all engineering modules    | Returns NaN; imputed downstream                   |
| Missing bureau / prev-app / installment records | Left merge in engineering_main.py        | NaN propagated; training imputes with median      |
| DAYS_EMPLOYED > 0 (retiree artifact)            | loader.py                                | IS_RETIRED = 1; DAYS_EMPLOYED clamped to 0        |
| LightGBM column name with special chars         | sanitize() in train_model.py and backend | All non-alphanumeric chars replaced with '\_'     |
| Model artifacts missing at startup              | backend/main.py @startup event           | RuntimeError raised with actionable message       |
| SK_ID_CURR not in database                      | predict() routing logic                  | Falls back to Model B silently                    |
| Inference row with unknown category value       | build_inference_row()                    | One-hot column not found → stays 0 (safe default) |
| train_processed.csv absent                      | backend/main.py @startup event           | Warning logged; server starts in lite-only mode   |

# **9\. Dependency Map**

## **9.1 Python Dependencies (requirements.txt)**

| **Package**           | **Version Constraint** | **Used In**                                       |
| --------------------- | ---------------------- | ------------------------------------------------- |
| lightgbm              | -                      | train_model.py, evaluate_lgbm.py, backend/main.py |
| scikit-learn          | -                      | PCA, SMOTE, metrics, LabelEncoder, KMeans         |
| imbalanced-learn      | -                      | SMOTE in train_model.py                           |
| optuna                | -                      | src/models/train_lgbm.py hyperparameter search    |
| sentence-transformers | -                      | src/features/nlp_embeddings.py                    |
| category_encoders     | -                      | src/features/categorical_encoding.py              |
| fastapi               | -                      | backend/main.py                                   |
| uvicorn               | -                      | ASGI server for FastAPI                           |
| pydantic              | -                      | PredictRequest schema in backend                  |
| joblib                | -                      | Model serialization/deserialization               |
| pandas                | -                      | All data processing                               |
| numpy                 | -                      | Numeric operations                                |
| shap                  | -                      | Model explainability (optional)                   |
| xgboost               | -                      | Available for baseline experiments                |
| streamlit             | -                      | Available for rapid dashboard prototyping         |

## **9.2 Node Dependencies (frontend/package.json)**

| **Package**          | **Version** | **Role**                   |
| -------------------- | ----------- | -------------------------- |
| react                | ^19.2.4     | UI library                 |
| react-dom            | ^19.2.4     | DOM renderer               |
| vite                 | ^8.0.4      | Dev server and bundler     |
| @vitejs/plugin-react | ^6.0.1      | JSX + Fast Refresh support |
| eslint               | ^9.39.4     | Code linting               |

# **10\. Known Limitations & Future Work**

| **Item**                     | **Type**    | **Description**                                                                                                  |
| ---------------------------- | ----------- | ---------------------------------------------------------------------------------------------------------------- |
| DATA_DIR hardcoded           | Limitation  | loader.py and engineering_main.py use absolute Windows paths - must be changed to relative paths for portability |
| No authentication on API     | Limitation  | CORS is allow-all; no API key or JWT - not production-safe                                                       |
| SMOTE on full dataset in RAM | Limitation  | 307K rows × 200+ features after SMOTE can require 8-12 GB RAM                                                    |
| NLP embeddings slow on CPU   | Limitation  | ~15-20 min for 307K rows with all-MiniLM-L6-v2 on CPU; GPU recommended                                           |
| No model versioning          | Limitation  | Overwriting .pkl files on re-train; no experiment tracking (MLflow, W&B)                                         |
| Target encoding leakage risk | Limitation  | If CV is used, target encoding should be inside the fold to prevent data leakage                                 |
| Configurable API base URL    | Future Work | Frontend should read VITE_API_URL from .env instead of hardcoded localhost                                       |
| SHAP integration             | Future Work | shap is in requirements; local explainability per-prediction endpoint not yet implemented                        |
| Model monitoring             | Future Work | No drift detection or prediction logging in production                                                           |
