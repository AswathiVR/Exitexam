from model_utils import save_models


if __name__ == "__main__":
    data = save_models()
    print(f"Saved models using {len(data):,} customer records.")
    print("Created logistic_model.joblib, forest_model.joblib, and model_columns.joblib")