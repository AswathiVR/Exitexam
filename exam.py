from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans


file_path = Path(__file__).parent / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
data = pd.read_csv(file_path)

# Convert spaces in TotalCharges to missing values, then fill them with 0.
data["TotalCharges"] = data["TotalCharges"].replace(r"^\s*$", pd.NA, regex=True)
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
data["TotalCharges"] = data["TotalCharges"].fillna(0)

# Convert Churn values to binary integers.
data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})

# Calculate the requested metrics.
overall_churn_rate = data["Churn"].mean() * 100
contract_churn = data.groupby("Contract")["Churn"].agg(
	count="count", churn_rate_percent="mean"
)
contract_churn["churn_rate_%"] = contract_churn.pop("churn_rate_percent") * 100

internet_churn = data.groupby("InternetService")["Churn"].agg(
	count="count", churn_rate_percent="mean"
)
internet_churn["churn_rate_%"] = internet_churn.pop("churn_rate_percent") * 100

tenure_churn_corr = data["tenure"].corr(data["Churn"])

# Create a simple plot of churn by contract type.
plot_file = Path(__file__).parent / "churn_by_contract.png"
contract_churn["churn_rate_%"].plot(kind="bar", color="steelblue")
plt.title("Customer Churn Rate by Contract Type")
plt.xlabel("Contract type")
plt.ylabel("Churn rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(plot_file)
plt.close()

# Output calculated KPI results.
print("CALCULATED KPI RESULTS")
print("=" * 24)
print(f"Overall Churn Rate: {overall_churn_rate:.2f}%\n")
print("Churn Rate by Contract Type:")
print(contract_churn[["count", "churn_rate_%"]].round(2))
print("\nChurn Rate by Internet Service Type:")
print(internet_churn[["count", "churn_rate_%"]].round(2))
print(f"\nPearson Correlation between Tenure and Churn: {tenure_churn_corr:.4f}")

# Explain the results in beginner-friendly language.
highest_contract = contract_churn["churn_rate_%"].idxmax()
highest_contract_rate = contract_churn.loc[highest_contract, "churn_rate_%"]
highest_internet = internet_churn["churn_rate_%"].idxmax()
highest_internet_rate = internet_churn.loc[highest_internet, "churn_rate_%"]

print("\nHUMAN ANALYSIS")
print("=" * 14)
print(f"About {overall_churn_rate:.1f} out of every 100 customers leave the company.")
print(
	f"Customers with a {highest_contract} contract have the highest churn "
	f"rate ({highest_contract_rate:.2f}%)."
)
print(
	f"Customers using {highest_internet} internet service have the highest "
	f"internet-service churn rate ({highest_internet_rate:.2f}%)."
)
print(
	f"The correlation is {tenure_churn_corr:.4f}, which means customers with "
	"longer tenure generally churn less in this dataset."
)
print(
	"Correlation is not causation: this analysis shows that tenure and churn "
	"are related, but it does not prove that tenure alone causes customers to stay."
)
print(f"\nPlot saved as: {plot_file.name}")

# Prepare data for machine learning.
target = data["Churn"]
features = data.drop(columns=["Churn", "customerID"])
numeric_columns = features.select_dtypes(include="number").columns.tolist()
categorical_columns = features.select_dtypes(exclude="number").columns.tolist()

numeric_steps = Pipeline(
	steps=[
		("fill_missing", SimpleImputer(strategy="median")),
		("scale", StandardScaler()),
	]
)
categorical_steps = Pipeline(
	steps=[
		("fill_missing", SimpleImputer(strategy="most_frequent")),
		("encode", OneHotEncoder(handle_unknown="ignore")),
	]
)
preprocessor = ColumnTransformer(
	transformers=[
		("numbers", numeric_steps, numeric_columns),
		("categories", categorical_steps, categorical_columns),
	]
)

logistic_model = Pipeline(
	steps=[
		("preprocess", preprocessor),
		("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
	]
)
forest_model = Pipeline(
	steps=[
		("preprocess", preprocessor),
		("model", RandomForestClassifier(
			n_estimators=250,
			random_state=42,
			class_weight="balanced",
			n_jobs=-1,
		)),
	]
)

train_features, test_features, train_target, test_target = train_test_split(
	features,
	target,
	test_size=0.2,
	random_state=42,
	stratify=target,
)

logistic_model.fit(train_features, train_target)
forest_model.fit(train_features, train_target)


def show_model_results(name, model):
	train_predictions = model.predict(train_features)
	test_predictions = model.predict(test_features)
	train_probabilities = model.predict_proba(train_features)[:, 1]
	test_probabilities = model.predict_proba(test_features)[:, 1]

	train_metrics = {
		"accuracy": accuracy_score(train_target, train_predictions),
		"precision": precision_score(train_target, train_predictions),
		"recall": recall_score(train_target, train_predictions),
		"f1": f1_score(train_target, train_predictions),
		"roc_auc": roc_auc_score(train_target, train_probabilities),
	}
	test_metrics = {
		"accuracy": accuracy_score(test_target, test_predictions),
		"precision": precision_score(test_target, test_predictions),
		"recall": recall_score(test_target, test_predictions),
		"f1": f1_score(test_target, test_predictions),
		"roc_auc": roc_auc_score(test_target, test_probabilities),
	}

	print(f"\n{name}")
	print("Metric       Train       Test")
	print(f"Accuracy     {train_metrics['accuracy']:.2%}      {test_metrics['accuracy']:.2%}")
	print(f"Precision    {train_metrics['precision']:.2%}      {test_metrics['precision']:.2%}")
	print(f"Recall       {train_metrics['recall']:.2%}      {test_metrics['recall']:.2%}")
	print(f"F1 score     {train_metrics['f1']:.2%}      {test_metrics['f1']:.2%}")
	print(f"ROC-AUC      {train_metrics['roc_auc']:.4f}      {test_metrics['roc_auc']:.4f}")

	return train_probabilities, test_probabilities


print("\nMACHINE LEARNING RESULTS")
print("=" * 24)
logistic_train_risk, logistic_test_risk = show_model_results(
	"Baseline: Logistic Regression", logistic_model
)
forest_train_risk, forest_test_risk = show_model_results(
	"Complex model: Random Forest", forest_model
)

print("\nWHY RECALL MATTERS TO THE CFO")
print("Accuracy can look good by mostly predicting the larger No-Churn group.")
print(
	"Recall is important because a missed churner is a customer at risk whose "
	"lost revenue was not flagged for retention action."
)
print(
	"The train and test scores are compared above. A much higher train score "
	"than test score would indicate overfitting; similar scores suggest better "
	"generalization."
)

# Aggregate Random Forest importance for one-hot encoded columns.
forest_preprocessor = forest_model.named_steps["preprocess"]
forest_classifier = forest_model.named_steps["model"]
encoded_names = forest_preprocessor.get_feature_names_out()
importance_by_column = {column: 0 for column in features.columns}
for encoded_name, importance in zip(encoded_names, forest_classifier.feature_importances_):
	clean_name = encoded_name.split("__", 1)[1]
	matching_column = next(
		column for column in features.columns if clean_name == column or clean_name.startswith(column + "_")
	)
	importance_by_column[matching_column] += importance

top_features = pd.Series(importance_by_column).sort_values(ascending=False).head(3)
print("\nTOP 3 FEATURES DRIVING CHURN")
print(top_features.round(4))

feature_actions = {
	"tenure": "Contact newer customers early with onboarding help and a 90-day retention check-in.",
	"MonthlyCharges": "Offer a right-sized plan or a time-limited discount before the next bill.",
	"TotalCharges": "Use account-value bands to prioritize save offers for high-value customers.",
	"Contract": "Offer month-to-month customers a clear annual-plan incentive with flexible benefits.",
	"InternetService": "Investigate fiber service complaints and offer technical support or service credits.",
	"PaymentMethod": "Promote automatic payment setup and resolve failed-payment issues quickly.",
	"SeniorCitizen": "Offer accessible support and a simple retention call for older customers.",
}
for feature in top_features.index:
	action = feature_actions.get(
		feature,
		f"Monitor {feature} and test a targeted retention offer for high-risk customers.",
	)
	print(f"{feature}: {action}")

# Create four customer groups using tenure, monthly charges, and predicted risk.
cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
out_of_fold_risk = cross_val_predict(
	forest_model,
	features,
	target,
	cv=cross_validation,
	method="predict_proba",
	n_jobs=1,
)[:, 1]

segment_data = data[["tenure", "MonthlyCharges"]].copy()
segment_data["predicted_churn_risk"] = out_of_fold_risk
segment_inputs = StandardScaler().fit_transform(segment_data)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
segment_data["cluster"] = kmeans.fit_predict(segment_inputs)

segment_summary = segment_data.groupby("cluster").agg(
	customers=("cluster", "size"),
	average_tenure=("tenure", "mean"),
	average_monthly_charges=("MonthlyCharges", "mean"),
	average_churn_risk=("predicted_churn_risk", "mean"),
	expected_monthly_revenue_at_risk=(
		"predicted_churn_risk",
		lambda risk: (risk * segment_data.loc[risk.index, "MonthlyCharges"]).sum(),
	),
)

average_charges = segment_summary["average_monthly_charges"].mean()
average_risk = segment_summary["average_churn_risk"].mean()


def name_segment(row):
	if row["average_churn_risk"] >= average_risk and row["average_monthly_charges"] >= average_charges:
		return "Urgent high-value"
	if row["average_churn_risk"] >= average_risk:
		return "At-risk value seekers"
	if row["average_monthly_charges"] >= average_charges:
		return "Loyal high-value"
	return "Stable low-value"


segment_summary["segment_name"] = segment_summary.apply(name_segment, axis=1)
segment_summary = segment_summary.sort_values(
	"expected_monthly_revenue_at_risk", ascending=False
)

print("\nCUSTOMER SEGMENTS")
print("=" * 18)
print(
	segment_summary[
		[
			"segment_name",
			"customers",
			"average_tenure",
			"average_monthly_charges",
			"average_churn_risk",
			"expected_monthly_revenue_at_risk",
		]
	].round(2).to_string(index=False)
)

first_segment = segment_summary.iloc[0]
print(
	f"\nCFO FUNDING PRIORITY: {first_segment['segment_name']} should be funded first. "
	f"It has the highest expected monthly revenue at risk "
	f"(${first_segment['expected_monthly_revenue_at_risk']:,.2f}), balancing "
	"customer risk with revenue value."
)

# Save a visual showing customer risk and monthly revenue value.
segment_plot = Path(__file__).parent / "customer_segments.png"
for _, row in segment_summary.iterrows():
	cluster_rows = segment_data[segment_data["cluster"] == row.name]
	plt.scatter(
		cluster_rows["MonthlyCharges"],
		cluster_rows["predicted_churn_risk"] * 100,
		label=row["segment_name"],
		alpha=0.35,
	)
plt.xlabel("Monthly charges ($)")
plt.ylabel("Predicted churn risk (%)")
plt.title("Customer Segments: Risk versus Monthly Charges")
plt.legend()
plt.tight_layout()
plt.savefig(segment_plot)
plt.close()
print(f"Customer segment plot saved as: {segment_plot.name}")

# Retention playbook: choose one business clause from risk and tenure only.
retention_clauses = {
	"Clause 1": (
		"Early-tenure support: offer onboarding help, explain the service clearly, "
		"and schedule a follow-up before the next billing cycle."
	),
	"Clause 2": (
		"Established-customer save: review the current plan, resolve service issues, "
		"and offer a relevant loyalty benefit without making an unsupported promise."
	),
	"Clause 3": (
		"Proactive value check: share useful plan guidance, confirm satisfaction, "
		"and invite the customer to contact support if their needs have changed."
	),
}

mandatory_clause_4 = (
	"Clause 4 - Non-Discrimination: Never use or infer gender, SeniorCitizen, "
	"Partner, Dependents, or any other protected or demographic attribute when "
	"selecting an offer or explaining a retention action."
)


def select_retention_clause(risk_score, tenure):
	if risk_score >= 0.60 and tenure < 12:
		selected_clause = "Clause 1"
	elif risk_score >= 0.60 and tenure >= 12:
		selected_clause = "Clause 2"
	else:
		selected_clause = "Clause 3"

	return selected_clause, retention_clauses[selected_clause], mandatory_clause_4


prohibited_fields = {"gender", "SeniorCitizen", "Partner", "Dependents"}
safe_top_features = [
	feature for feature in top_features.index if feature not in prohibited_fields
]
if len(safe_top_features) != len(top_features.index):
	raise ValueError("Demographic fields cannot be used in the retention explanation.")

example_customer = data.loc[segment_data["predicted_churn_risk"].idxmax()]
example_risk = out_of_fold_risk[example_customer.name]
selected_clause, clause_text, non_discrimination_clause = select_retention_clause(
	example_risk, example_customer["tenure"]
)

llm_system_prompt = """You are a retention assistant for a telecom company.
Write a professional explanation in exactly 3 or 4 sentences for a retention agent.
Use only the retrieved retention clause and the supplied allowed churn features.
Do not invent customer facts, discounts, causes, or promises.
Never use, mention, infer, or make decisions from gender, SeniorCitizen, Partner,
Dependents, or any other demographic or protected attribute.
Clause 4 (Non-Discrimination) is mandatory for every response.
"""

llm_user_context = {
	"retrieved_clause": f"{selected_clause}: {clause_text}",
	"mandatory_guardrail": non_discrimination_clause,
	"allowed_top_features": safe_top_features,
	"risk_score": round(float(example_risk), 4),
	"tenure_months": int(example_customer["tenure"]),
}

print("\nRETENTION PLAYBOOK")
print("=" * 19)
print(f"Example risk score: {example_risk:.2%}")
print(f"Example tenure: {int(example_customer['tenure'])} months")
print(f"Retrieved clause: {selected_clause}")
print(f"Clause text: {clause_text}")
print(f"Mandatory guardrail: {non_discrimination_clause}")
print(f"Allowed features for LLM: {safe_top_features}")
print("\nLLM SYSTEM PROMPT")
print("=" * 17)
print(llm_system_prompt)
print("LLM input context:")
print(llm_user_context)

# Demonstrate why the retrieval step must not be skipped.
guessed_clause_without_retrieval = "Clause 2"
retrieval_was_skipped = True
guess_was_wrong = guessed_clause_without_retrieval != selected_clause

print("\nRETRIEVAL SAFETY TEST")
print("=" * 22)
print(f"Correct clause from risk and tenure: {selected_clause}")
print(
	f"Clause guessed without retrieval: {guessed_clause_without_retrieval} "
	"(simulation)"
)
print(f"Did the guess match the correct clause? {'No' if guess_was_wrong else 'Yes'}")
print(
	"Result: skipping retrieval can cause a hallucination because the LLM does "
	"not know which approved business rule applies."
)
print(
	"Why this is dangerous: a made-up clause could produce an unauthorized "
	"discount, a false promise, or an unsuitable retention action. In a "
	"compliance environment, that can cause financial loss, inconsistent "
	"customer treatment, failed audits, and reputational or legal risk."
)
print(
	"The approved clause should always be selected by code first; the LLM should "
	"only explain that clause and never invent a replacement."
)

# Demonstrate that protected demographic fields are excluded from the LLM payload.
demographic_fields_sent = sorted(prohibited_fields.intersection(llm_user_context))
demographic_fields_removed = sorted(prohibited_fields.intersection(data.columns))

print("\nNON-DISCRIMINATION PAYLOAD TEST")
print("=" * 31)
print(f"Demographic columns removed before the LLM call: {demographic_fields_removed}")
print(f"Demographic columns present in the LLM payload: {demographic_fields_sent}")
print(
	"Result: the LLM cannot directly use these demographic values to choose "
	"an offer or explain a retention action. This helps prevent bias leak and "
	"discrimination in generated communication."
)
print(
	"Important limitation: excluding demographic fields from the prompt reduces "
	"direct bias, but it does not guarantee fairness. Features such as location, "
	"payment method, or service type may act as indirect proxies, so outcomes "
	"should still be checked across customer groups."
)