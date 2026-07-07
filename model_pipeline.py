"""
model_pipeline.py
------------------
Backend model logic: training, hyperparameter tuning, evaluation, and SHAP
explainability. Kept separate from data_pipeline.py and the Streamlit UI so
it can be reused or unit-tested independently.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

DEFAULT_PARAMS = dict(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

PARAM_DIST = {
    "n_estimators": [100, 200, 300],
    "max_depth": [4, 6, 7, 9],
    "learning_rate": [0.03, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
}


def train_model(X_train: pd.DataFrame, y_train_log: pd.Series,
                 tune: bool = False, n_iter: int = 15, cv_splits: int = 3,
                 progress_callback=None):
    """
    Trains an XGBRegressor on log1p(sales). If tune=True, runs
    RandomizedSearchCV with a TimeSeriesSplit CV (never shuffled, so no
    future data leaks into training folds during the search).
    Returns (fitted_model, params_used).
    """
    if tune:
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        search = RandomizedSearchCV(
            estimator=XGBRegressor(random_state=42),
            param_distributions=PARAM_DIST,
            n_iter=n_iter,
            scoring="neg_root_mean_squared_error",
            cv=tscv,
            n_jobs=-1,
            random_state=42,
        )
        search.fit(X_train, y_train_log)
        model = search.best_estimator_
        params_used = search.best_params_
    else:
        model = XGBRegressor(**DEFAULT_PARAMS)
        model.fit(X_train, y_train_log)
        params_used = DEFAULT_PARAMS

    return model, params_used


def evaluate_model(model, X_test: pd.DataFrame, y_test_actual: pd.Series) -> dict:
    """y_test_actual must be the RAW (non-log) sales values."""
    y_pred_log = model.predict(X_test)
    y_pred = np.clip(np.expm1(y_pred_log), 0, None)

    mae = mean_absolute_error(y_test_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
    try:
        rmsle = np.sqrt(mean_squared_log_error(y_test_actual, y_pred))
    except ValueError:
        rmsle = float("nan")  # negative predictions edge case, shouldn't happen post-clip

    return {"MAE": mae, "RMSE": rmse, "RMSLE": rmsle, "y_pred": y_pred}


def get_shap_values(model, X_sample: pd.DataFrame):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    return explainer, shap_values


def save_model(model, path: str):
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)
