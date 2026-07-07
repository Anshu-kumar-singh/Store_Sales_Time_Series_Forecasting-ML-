# 🛒 Store Sales Time Series Forecasting

A Machine Learning project that predicts future store sales using historical sales data. This project is built using **XGBoost**, advanced **time series feature engineering**, and **SHAP Explainable AI**. It also includes an interactive **Streamlit** web application for data exploration, model training, evaluation, and prediction.

---

## 📌 Project Overview

This project forecasts store sales by combining historical sales records with additional business information such as store details, oil prices, holidays, and transactions.

The complete pipeline includes:

- Data preprocessing and merging
- Time series feature engineering
- XGBoost regression model
- Hyperparameter tuning
- Model evaluation
- SHAP explainability
- Interactive Streamlit dashboard

---

## 🚀 Features

- Upload Kaggle Store Sales dataset
- Automatic data preprocessing
- Time-based train/test split
- Lag feature generation
- Rolling mean and rolling standard deviation features
- Date-based feature extraction
- Label Encoding and One-Hot Encoding
- XGBoost model training
- Hyperparameter tuning using RandomizedSearchCV
- SHAP feature importance visualization
- Error analysis
- Sales prediction interface
- Save trained model using Joblib

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- SHAP
- Streamlit
- Matplotlib
- Seaborn
- Joblib

---

## 📂 Project Structure

```
Store_Sales_Time_Series_Forecasting/
│
├── app.py                     # Streamlit Application
├── data_pipeline.py           # Data preprocessing & feature engineering
├── model_pipeline.py          # Model training and evaluation
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

This project uses the **Store Sales - Time Series Forecasting** dataset from Kaggle.

Dataset includes:

- train.csv
- test.csv
- stores.csv
- oil.csv
- transactions.csv
- holidays_events.csv

Download the dataset from Kaggle and upload the CSV files through the Streamlit interface.

---

## ⚙️ Machine Learning Pipeline

### 1. Data Loading

- Load all CSV files
- Parse date columns
- Merge datasets

### 2. Feature Engineering

The following features are generated:

- Year
- Month
- Day
- Day of Week

### Lag Features

- Lag 1
- Lag 7
- Lag 14
- Lag 30
- Lag 365

### Rolling Statistics

- Rolling Mean (7 days)
- Rolling Mean (30 days)
- Rolling Standard Deviation (7 days)
- Rolling Standard Deviation (30 days)

### Encoding

- Label Encoding
- One-Hot Encoding

---

## 🤖 Model

The project uses **XGBoost Regressor** for forecasting.

Default parameters:

- n_estimators = 300
- max_depth = 7
- learning_rate = 0.1
- subsample = 0.8
- colsample_bytree = 0.8

Optional hyperparameter tuning is performed using:

- RandomizedSearchCV
- TimeSeriesSplit Cross Validation

---

## 📈 Model Evaluation

The model is evaluated using:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- RMSLE (Root Mean Squared Log Error)

---

## 🔍 Explainable AI

The project uses **SHAP (SHapley Additive exPlanations)** to explain model predictions.

SHAP visualizations help identify:

- Most important features
- Positive and negative feature contributions
- Global feature importance

---

## 🖥️ Streamlit Dashboard

The application includes the following sections:

- 📊 Exploratory Data Analysis (EDA)
- 🛠️ Feature Engineering
- 🤖 Model Training
- 🔍 SHAP Explainability
- 📉 Error Analysis
- 📤 Prediction & Submission

---

## ▶️ Installation

Clone the repository

```bash
git clone https://github.com/your-username/Store_Sales_Time_Series_Forecasting.git
```

Move into the project folder

```bash
cd Store_Sales_Time_Series_Forecasting
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run app.py
```

---

## 📷 Workflow

```
CSV Files
     │
     ▼
Data Loading
     │
     ▼
Feature Engineering
     │
     ▼
Train/Test Split
     │
     ▼
XGBoost Training
     │
     ▼
Model Evaluation
     │
     ▼
SHAP Explainability
     │
     ▼
Prediction
```

---

## 🎯 Future Improvements

- LSTM and GRU based forecasting
- Prophet model comparison
- Ensemble learning
- Automated feature selection
- Model deployment using Docker
- Cloud deployment (Render/AWS)

---

## 📌 Requirements

```
streamlit
pandas
numpy
scikit-learn
xgboost
shap
matplotlib
seaborn
joblib
```

Install using:

```bash
pip install -r requirements.txt
```

---

## 👨‍💻 Author

**Anshu Kumar Singh**

Machine Learning | Data Science | Python | Deep Learning | NLP

---

## ⭐ If you found this project useful, consider giving it a Star!
