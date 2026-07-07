# Store Sales Time Series Forecasting

XGBoost + SHAP forecasting pipeline for the Kaggle "Store Sales - Time Series
Forecasting" competition, split into a backend (`data_pipeline.py`,
`model_pipeline.py`) and a Streamlit frontend (`app.py`).

## Structure
- `data_pipeline.py` — loading, merging, null-handling, lag/rolling feature
  engineering, label/one-hot encoding. `SalesFeatureEngineer` is fit once on
  train data and reused (`.transform`) on the real test set, so both go
  through identical preprocessing (no train/test skew).
- `model_pipeline.py` — XGBoost training (with optional `RandomizedSearchCV`
  + `TimeSeriesSplit` tuning), evaluation (MAE / RMSE / RMSLE), SHAP values.
- `app.py` — Streamlit UI: upload CSVs → EDA → feature engineering → train →
  SHAP explainability → error analysis → generate `submission.csv`.

## Setup
```bash
pip install -r requirements.txt
```

## Run
```bash
streamlit run app.py
```

Download `train.csv`, `test.csv`, `stores.csv`, `oil.csv`,
`transactions.csv`, `holidays_events.csv` from the Kaggle competition page
and upload them in the sidebar (test.csv is only needed for the final
submission step).

## Notes
- Target is trained on `log1p(sales)`; predictions are `expm1`-transformed
  and clipped at 0.
- The 80/20 split is chronological (no shuffling), matching the time-series
  nature of the data.
- Fill values (oil price, transactions median) are computed from train only
  and reused on test, to avoid leakage.
