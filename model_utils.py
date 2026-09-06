from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_COLUMNS = [
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges",
]


def load_data():
    data = pd.read_csv(DATA_FILE)
    data["TotalCharges"] = data["TotalCharges"].replace(r"^\s*$", pd.NA, regex=True)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce").fillna(0)
    data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})
    return data


def make_preprocessor():
    numeric_columns = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_columns = [column for column in MODEL_COLUMNS if column not in numeric_columns]
    return ColumnTransformer([
        ("numbers", Pipeline([
            ("fill_missing", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric_columns),
        ("categories", Pipeline([
            ("fill_missing", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_columns),
    ])


def make_models():
    return {
        "logistic": Pipeline([
            ("preprocess", make_preprocessor()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "forest": Pipeline([
            ("preprocess", make_preprocessor()),
            ("model", RandomForestClassifier(
                n_estimators=250, random_state=42, class_weight="balanced", n_jobs=-1
            )),
        ]),
    }


def save_models():
    data = load_data()
    features = data[MODEL_COLUMNS]
    target = data["Churn"]
    for name, model in make_models().items():
        model.fit(features, target)
        joblib.dump(model, BASE_DIR / f"{name}_model.joblib")
    joblib.dump(MODEL_COLUMNS, BASE_DIR / "model_columns.joblib")
    return data