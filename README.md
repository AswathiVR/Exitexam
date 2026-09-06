# Telco Customer Churn Streamlit App

## Run locally

```powershell
python -m pip install -r requirements.txt
python save_models.py
streamlit run app.py
```

The app loads `forest_model.joblib` and does not retrain on every page load.

## Deploy publicly

1. Push all project files to a public GitHub repository.
2. Open https://share.streamlit.io and sign in with GitHub.
3. Select **New app**.
4. Choose the repository and the `main` branch.
5. Set the main file to `app.py`.
6. Click **Deploy**.

Commit these saved model files because Streamlit Cloud needs them:

- `forest_model.joblib`
- `logistic_model.joblib`
- `model_columns.joblib`

Only load model files created by you. Never upload secrets or API keys.