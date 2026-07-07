"""
app.py
------
Streamlit frontend for the Store Sales Time Series Forecasting project.
Run with:  streamlit run app.py

This is a thin UI layer only -- all real logic lives in data_pipeline.py and
model_pipeline.py so the backend can be reused/tested without Streamlit.
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from data_pipeline import SalesFeatureEngineer, load_raw_csvs, time_split
from model_pipeline import train_model, evaluate_model, get_shap_values, save_model

st.set_page_config(page_title="Store Sales Forecasting", layout="wide")
st.title("🛒 Store Sales Time Series Forecasting")
st.caption("XGBoost + SHAP pipeline for the Kaggle Store Sales competition")

# ---------------------------------------------------------------- session state
for key in ["raw", "fe", "engineered_df", "train_df", "test_df",
            "X_train", "y_train_log", "X_test", "y_test", "model",
            "eval_results", "params_used"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ---------------------------------------------------------------- sidebar: data upload
st.sidebar.header("1. Upload data")
st.sidebar.caption("Download the CSVs from the Kaggle competition page first.")

train_file = st.sidebar.file_uploader("train.csv", type="csv")
stores_file = st.sidebar.file_uploader("stores.csv", type="csv")
oil_file = st.sidebar.file_uploader("oil.csv", type="csv")
transactions_file = st.sidebar.file_uploader("transactions.csv", type="csv")
holidays_file = st.sidebar.file_uploader("holidays_events.csv", type="csv")
test_file = st.sidebar.file_uploader("test.csv (optional, for final submission)", type="csv")

run_pipeline = st.sidebar.button("Load & merge data", type="primary")

if run_pipeline:
    required = {"train": train_file, "stores": stores_file, "oil": oil_file,
                "transactions": transactions_file, "holidays": holidays_file}
    if any(v is None for v in required.values()):
        st.sidebar.error("Please upload train, stores, oil, transactions, and holidays_events.")
    else:
        files = dict(required)
        if test_file is not None:
            files["test"] = test_file
        st.session_state.raw = load_raw_csvs(files)
        st.sidebar.success("Files loaded.")

tabs = st.tabs(["📊 EDA", "🛠️ Feature Engineering", "🤖 Train Model",
                "🔍 Explainability (SHAP)", "📉 Error Analysis", "📤 Predict & Submit"])

# ---------------------------------------------------------------- Tab 1: EDA
with tabs[0]:
    st.subheader("Exploratory Data Analysis")
    if st.session_state.raw is None:
        st.info("Upload and load data from the sidebar first.")
    else:
        raw = st.session_state.raw
        df = raw["train"].merge(raw["stores"], on="store_nbr", how="left")
        df = df.merge(raw["oil"], on="date", how="left")
        df = df.merge(raw["transactions"], on=["date", "store_nbr"], how="left")
        hol = raw["holidays"][["date", "type"]].rename(columns={"type": "holiday_type"})
        df = df.merge(hol.drop_duplicates("date"), on="date", how="left")
        df = df.rename(columns={"type": "store_type", "dcoilwtico": "oil_price"})
        df["date"] = pd.to_datetime(df["date"])
        df["day_of_week"] = df["date"].dt.dayofweek
        df["month"] = df["date"].dt.month

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            ax.hist(df["sales"], bins=100, color="steelblue")
            ax.set_title("Sales Distribution (raw)")
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots()
            ax.hist(np.log1p(df["sales"]), bins=100, color="darkorange")
            ax.set_title("Sales Distribution (log1p)")
            st.pyplot(fig)

        st.markdown("**Total daily sales over time**")
        fig, ax = plt.subplots(figsize=(12, 4))
        df.groupby("date")["sales"].sum().plot(ax=ax)
        st.pyplot(fig)

        col3, col4 = st.columns(2)
        with col3:
            fig, ax = plt.subplots()
            df.groupby("day_of_week")["sales"].mean().plot(kind="bar", ax=ax, color="seagreen")
            ax.set_title("Avg Sales by Day of Week (0=Mon)")
            st.pyplot(fig)
        with col4:
            fig, ax = plt.subplots()
            df.groupby("month")["sales"].mean().plot(kind="bar", ax=ax, color="indianred")
            ax.set_title("Avg Sales by Month")
            st.pyplot(fig)

        st.markdown("**Avg sales by product family**")
        fig, ax = plt.subplots(figsize=(8, 8))
        df.groupby("family")["sales"].mean().sort_values().plot(kind="barh", ax=ax, color="mediumpurple")
        st.pyplot(fig)

        st.markdown("**Correlation heatmap**")
        num_cols = [c for c in ["sales", "onpromotion", "oil_price", "transactions"] if c in df.columns]
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        st.pyplot(fig)

# ---------------------------------------------------------------- Tab 2: Feature Engineering
with tabs[1]:
    st.subheader("Feature Engineering")
    if st.session_state.raw is None:
        st.info("Upload and load data from the sidebar first.")
    else:
        test_fraction = st.slider("Holdout test fraction (chronological)", 0.1, 0.4, 0.2, 0.05)
        if st.button("Run feature engineering"):
            with st.spinner("Merging, encoding, building lag/rolling features..."):
                raw = st.session_state.raw
                fe = SalesFeatureEngineer()
                engineered = fe.fit_transform(
                    raw["train"], raw["stores"], raw["oil"], raw["transactions"], raw["holidays"]
                )
                st.session_state.fe = fe
                st.session_state.engineered_df = engineered

                train_df, test_df, split_date = time_split(engineered, test_fraction)
                st.session_state.train_df = train_df
                st.session_state.test_df = test_df

                feat_cols = fe.feature_columns
                st.session_state.X_train = train_df[feat_cols]
                st.session_state.y_train_log = np.log1p(train_df["sales"])
                st.session_state.X_test = test_df[feat_cols]
                st.session_state.y_test = test_df["sales"]

                st.success(f"Done. Chronological split at {split_date.date()} "
                           f"({len(train_df)} train rows / {len(test_df)} test rows).")

        if st.session_state.engineered_df is not None:
            st.markdown("**Engineered data preview**")
            st.dataframe(st.session_state.engineered_df.head(20))
            st.markdown("**Null counts after fill**")
            st.dataframe(st.session_state.engineered_df.isnull().sum().rename("nulls"))

# ---------------------------------------------------------------- Tab 3: Train Model
with tabs[2]:
    st.subheader("Train Model")
    if st.session_state.X_train is None:
        st.info("Run feature engineering first.")
    else:
        tune = st.checkbox("Tune hyperparameters with RandomizedSearchCV (slower)", value=False)
        n_iter = st.slider("Search iterations", 5, 30, 15) if tune else 15
        if st.button("Train"):
            with st.spinner("Training XGBoost..."):
                model, params_used = train_model(
                    st.session_state.X_train, st.session_state.y_train_log,
                    tune=tune, n_iter=n_iter,
                )
                st.session_state.model = model
                st.session_state.params_used = params_used

                results = evaluate_model(model, st.session_state.X_test, st.session_state.y_test)
                st.session_state.eval_results = results

        if st.session_state.model is not None:
            st.success("Model trained.")
            st.json(st.session_state.params_used)
            r = st.session_state.eval_results
            c1, c2, c3 = st.columns(3)
            c1.metric("MAE", f"{r['MAE']:.2f}")
            c2.metric("RMSE", f"{r['RMSE']:.2f}")
            c3.metric("RMSLE", f"{r['RMSLE']:.4f}")

            if st.button("Save model to model.joblib"):
                save_model(st.session_state.model, "model.joblib")
                st.session_state.fe.save("feature_engineer.joblib")
                st.success("Saved model.joblib and feature_engineer.joblib")

# ---------------------------------------------------------------- Tab 4: SHAP
with tabs[3]:
    st.subheader("Model Explainability (SHAP)")
    if st.session_state.model is None:
        st.info("Train a model first.")
    else:
        sample_size = st.slider("Sample size for SHAP (larger = slower)", 200, 5000, 1000, 100)
        if st.button("Compute SHAP values"):
            X_sample = st.session_state.X_test.sample(
                min(sample_size, len(st.session_state.X_test)), random_state=42
            )
            with st.spinner("Computing SHAP values..."):
                explainer, shap_values = get_shap_values(st.session_state.model, X_sample)
                st.session_state["shap_values"] = shap_values
                st.session_state["shap_X"] = X_sample

        if "shap_values" in st.session_state and st.session_state.get("shap_values") is not None:
            st.markdown("**Global feature importance**")
            fig = plt.figure()
            shap.summary_plot(st.session_state["shap_values"], st.session_state["shap_X"],
                               plot_type="bar", show=False)
            st.pyplot(fig)
            plt.clf()

            st.markdown("**Impact direction (beeswarm)**")
            fig = plt.figure()
            shap.summary_plot(st.session_state["shap_values"], st.session_state["shap_X"], show=False)
            st.pyplot(fig)
            plt.clf()

# ---------------------------------------------------------------- Tab 5: Error Analysis
with tabs[4]:
    st.subheader("Error Analysis (holdout test set)")
    if st.session_state.model is None:
        st.info("Train a model first.")
    else:
        test_df_eval = st.session_state.test_df.copy()
        test_df_eval["y_pred"] = st.session_state.eval_results["y_pred"]
        test_df_eval["abs_error"] = (test_df_eval["sales"] - test_df_eval["y_pred"]).abs()

        # map encoded family back to names if possible
        fe = st.session_state.fe
        family_names = dict(enumerate(fe.label_encoders["family"].classes_))
        test_df_eval["family_name"] = test_df_eval["family"].map(family_names)

        st.markdown("**Avg absolute error by product family**")
        fig, ax = plt.subplots(figsize=(8, 8))
        test_df_eval.groupby("family_name")["abs_error"].mean().sort_values().plot(
            kind="barh", ax=ax, color="crimson"
        )
        st.pyplot(fig)

        st.markdown("**Top 15 stores by avg absolute error**")
        fig, ax = plt.subplots(figsize=(8, 6))
        test_df_eval.groupby("store_nbr")["abs_error"].mean().sort_values(ascending=False).head(15).plot(
            kind="barh", ax=ax, color="darkorange"
        )
        ax.invert_yaxis()
        st.pyplot(fig)

        st.markdown("**Actual vs predicted total daily sales**")
        fig, ax = plt.subplots(figsize=(12, 4))
        test_df_eval.groupby("date")["sales"].sum().plot(ax=ax, label="Actual", color="steelblue")
        test_df_eval.groupby("date")["y_pred"].sum().plot(ax=ax, label="Predicted", color="darkorange")
        ax.legend()
        st.pyplot(fig)

# ---------------------------------------------------------------- Tab 6: Predict & Submit
with tabs[5]:
    st.subheader("Kaggle Submission Pipeline")
    if st.session_state.model is None:
        st.info("Train a model first.")
    elif st.session_state.raw is None or "test" not in st.session_state.raw:
        st.info("Upload test.csv in the sidebar and click 'Load & merge data' again.")
    else:
        if st.button("Generate submission.csv"):
            with st.spinner("Running full pipeline on test.csv..."):
                raw = st.session_state.raw
                fe = st.session_state.fe
                test_engineered = fe.transform(
                    raw["test"], raw["stores"], raw["oil"], raw["transactions"], raw["holidays"],
                    is_test=True,
                )
                X_submit = fe.align_features(test_engineered)
                y_submit_log = st.session_state.model.predict(X_submit)
                y_submit = np.clip(np.expm1(y_submit_log), 0, None)

                submission = pd.DataFrame({
                    "id": raw["test"]["id"].values,
                    "sales": y_submit,
                })
                st.session_state["submission"] = submission

        if "submission" in st.session_state:
            st.dataframe(st.session_state["submission"].head(20))
            csv_bytes = st.session_state["submission"].to_csv(index=False).encode("utf-8")
            st.download_button("Download submission.csv", csv_bytes, "submission.csv", "text/csv")
