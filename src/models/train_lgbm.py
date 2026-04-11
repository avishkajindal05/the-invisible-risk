import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import os
import joblib

def load_data(data_dir=r"d:\IITGN\Project\2 - Invincibles\CodeBase\Data\processed"):
    train = pd.read_csv(os.path.join(data_dir, "train_processed.csv"))
    # Assume categorical_encoding.py applied and targets exist
    y = train['TARGET']
    features = [c for c in train.columns if c not in ['TARGET', 'SK_ID_CURR']]
    X = train[features].copy()
    
    # LightGBM requires categorical types instead of object strings
    cat_cols = X.select_dtypes(include=['object']).columns
    X[cat_cols] = X[cat_cols].astype('category')
        
    return X, y, train['SK_ID_CURR']

def objective(trial, X, y, scale_pos_weight):
    param = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
        'num_leaves': trial.suggest_int('num_leaves', 31, 128),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 0.5),
        'scale_pos_weight': scale_pos_weight,
        'verbose': -1,
        'random_state': 42,
        'device': 'cpu'
    }
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    auc_scores = []
    
    for train_idx, valid_idx in cv.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
        
        dtrain = lgb.Dataset(X_train, label=y_train)
        dvalid = lgb.Dataset(X_valid, label=y_valid)
        
        model = lgb.train(
            param, 
            dtrain, 
            valid_sets=[dvalid], 
            num_boost_round=500,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False)
            ]
        )
        
        preds = model.predict(X_valid, num_iteration=model.best_iteration)
        auc = roc_auc_score(y_valid, preds)
        auc_scores.append(auc)
        
    return np.mean(auc_scores)

def train_final_model(X, y, best_params, scale_pos_weight, ids):
    """
    Trains on 5-Fold Stratified CV, saves OOF preds for stacker.
    """
    best_params['objective'] = 'binary'
    best_params['metric'] = 'auc'
    best_params['scale_pos_weight'] = scale_pos_weight
    best_params['verbose'] = -1
    best_params['device'] = 'cpu'
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    
    importances = np.zeros(X.shape[1])
    fold_models = []
    
    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
        
        dtrain = lgb.Dataset(X_train, label=y_train)
        dvalid = lgb.Dataset(X_valid, label=y_valid)
        
        model = lgb.train(
            best_params, 
            dtrain, 
            valid_sets=[dvalid], 
            num_boost_round=2000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=True),
                lgb.log_evaluation(period=100)
            ]
        )
        
        fold_preds = model.predict(X_valid, num_iteration=model.best_iteration)
        oof_preds[valid_idx] = fold_preds
        
        # Collect importance
        importances += model.feature_importance(importance_type='gain') / 5
        fold_models.append(model)
        
    final_auc = roc_auc_score(y, oof_preds)
    print(f"Final 5-Fold OOF AUC: {final_auc:.5f}")
    
    # Save OOF
    oof_df = pd.DataFrame({
        'SK_ID_CURR': ids,
        'LGBM_OOF': oof_preds
    })
    
    out_dir = r"d:\IITGN\Project\2 - Invincibles\CodeBase\models"
    os.makedirs(out_dir, exist_ok=True)
    oof_df.to_csv(os.path.join(out_dir, "oof_lgbm.csv"), index=False)
    print("OOF predictions saved.")
    
    return fold_models, importances

if __name__ == "__main__":
    X, y, ids = load_data()
    scale_pos_weight = (len(y) - y.sum()) / y.sum()
    print(f"Computed scale_pos_weight: {scale_pos_weight:.2f}")
    
    # Optuna study - commented out to avoid blind run, user runs it
    # study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42), pruner=MedianPruner())
    # study.optimize(lambda trial: objective(trial, X, y, scale_pos_weight), n_trials=80)
    # print('Best trial:', study.best_trial.params)
    
    # train_final_model(X, y, study.best_trial.params, scale_pos_weight, ids)
