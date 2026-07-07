"""
data_pipeline.py
----------------
Backend data-processing logic for the Store Sales Time Series Forecasting project.

This module is UI-agnostic. It takes raw CSVs (as pandas DataFrames) and turns
them into model-ready features. The same `SalesFeatureEngineer` instance is
fit once on the training data and then reused (via `transform`) on the Kaggle
test set, so both go through identical merges, fills, encodings, and lag
calculations -- no train/test skew.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

LAG_DAYS = [1, 7, 14, 30, 365]
ROLLING_WINDOWS = [7, 30]
LABEL_ENCODE_COLS = ["family", "city", "state"]
ONE_HOT_COLS = ["store_type", "holiday_type"]


def load_raw_csvs(files: dict) -> dict:
    """
    files: dict mapping logical name -> file path or file-like object, with keys:
        'train', 'test', 'stores', 'oil', 'transactions', 'holidays'
    Returns a dict of DataFrames with dates already parsed.
    """
    date_cols = {"train": "date", "test": "date", "oil": "date", "transactions": "date",
                 "holidays": "date"}
    data = {}
    for name, f in files.items():
        if f is None:
            continue
        parse_dates = [date_cols[name]] if name in date_cols else None
        data[name] = pd.read_csv(f, parse_dates=parse_dates)
    return data


def _merge_base(df: pd.DataFrame, stores: pd.DataFrame, oil: pd.DataFrame,
                 transactions: pd.DataFrame, holidays: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(stores, on="store_nbr", how="left")
    df = df.merge(oil, on="date", how="left")
    df = df.merge(transactions, on=["date", "store_nbr"], how="left")

    hol = holidays[["date", "type"]].rename(columns={"type": "holiday_type"})
    hol = hol.drop_duplicates(subset="date")
    df = df.merge(hol, on="date", how="left")

    df = df.drop_duplicates().reset_index(drop=True)
    return df


def _add_date_parts(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    return df


def _add_lag_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    grp = df.groupby(["store_nbr", "family"])["sales"]

    for lag in LAG_DAYS:
        df[f"lag_{lag}"] = grp.shift(lag)

    # rolling stats computed on the shifted (lag_1) series so the current row's
    # own sales value never leaks into its own rolling window
    shifted = grp.shift(1)
    df["_shifted_sales"] = shifted
    for window in ROLLING_WINDOWS:
        df[f"rolling_mean_{window}"] = (
            df.groupby(["store_nbr", "family"])["_shifted_sales"]
            .transform(lambda s: s.rolling(window, min_periods=1).mean())
        )
        df[f"rolling_std_{window}"] = (
            df.groupby(["store_nbr", "family"])["_shifted_sales"]
            .transform(lambda s: s.rolling(window, min_periods=1).std())
        )
    df = df.drop(columns=["_shifted_sales"])
    return df


class SalesFeatureEngineer:
    """
    Fit on the training data once, then call `transform` on any new data
    (validation split or the real Kaggle test.csv) to get identical processing.
    """

    def __init__(self):
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.oil_fill_series: pd.Series | None = None       # date -> oil_price, ffilled
        self.transactions_median: float | None = None
        self.dummy_columns: list[str] | None = None          # final one-hot columns seen in train
        self.feature_columns: list[str] | None = None        # full X column order used for training
        self.is_fitted = False

    def _base_merge_and_dates(self, raw_df, stores, oil, transactions, holidays):
        df = _merge_base(raw_df, stores, oil, transactions, holidays)
        df = _add_date_parts(df)
        df = df.rename(columns={"type": "store_type", "dcoilwtico": "oil_price"})
        return df

    def fit_transform(self, train_raw, stores, oil, transactions, holidays) -> pd.DataFrame:
        df = self._base_merge_and_dates(train_raw, stores, oil, transactions, holidays)

        # save fill statistics from TRAIN ONLY
        oil_series = df[["date", "oil_price"]].drop_duplicates("date").sort_values("date")
        oil_series["oil_price"] = oil_series["oil_price"].ffill().bfill()
        self.oil_fill_series = oil_series.set_index("date")["oil_price"]
        self.transactions_median = df["transactions"].median()

        df["oil_price"] = df["date"].map(self.oil_fill_series)
        df["transactions"] = df["transactions"].fillna(self.transactions_median)
        df["holiday_type"] = df["holiday_type"].fillna("None")

        df = _add_lag_rolling_features(df)

        for col in LABEL_ENCODE_COLS:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le

        df = pd.get_dummies(df, columns=ONE_HOT_COLS, drop_first=True)
        self.dummy_columns = [c for c in df.columns
                               if any(c.startswith(p + "_") for p in ONE_HOT_COLS)]

        drop_cols = [c for c in ["date", "id"] if c in df.columns]
        self.feature_columns = [c for c in df.columns if c not in drop_cols + ["sales"]]
        self.is_fitted = True
        return df

    def transform(self, raw_df, stores, oil, transactions, holidays, is_test=False) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Call fit_transform on training data before transform().")

        df = self._base_merge_and_dates(raw_df, stores, oil, transactions, holidays)

        if is_test:
            df["sales"] = np.nan

        df["oil_price"] = df["date"].map(self.oil_fill_series)
        df["oil_price"] = df["oil_price"].ffill().bfill()  # covers dates outside train's oil range
        df["transactions"] = df["transactions"].fillna(self.transactions_median)
        df["holiday_type"] = df["holiday_type"].fillna("None")

        df = _add_lag_rolling_features(df)

        for col in LABEL_ENCODE_COLS:
            le = self.label_encoders[col]
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(lambda v: v if v in known else le.classes_[0])
            df[col] = le.transform(df[col])

        df = pd.get_dummies(df, columns=ONE_HOT_COLS, drop_first=True)
        for c in self.dummy_columns:
            if c not in df.columns:
                df[c] = 0

        return df

    def align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reindex any engineered frame to the exact training feature column order."""
        return df.reindex(columns=self.feature_columns, fill_value=0)

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "SalesFeatureEngineer":
        return joblib.load(path)


def time_split(df: pd.DataFrame, test_fraction: float = 0.2):
    """80/20 chronological split (no shuffling) based on the date column's range."""
    date_range_days = (df["date"].max() - df["date"].min()).days
    split_date = df["date"].min() + pd.Timedelta(days=int(date_range_days * (1 - test_fraction)))
    train_df = df[df["date"] < split_date].copy()
    test_df = df[df["date"] >= split_date].copy()
    return train_df, test_df, split_date
