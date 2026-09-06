# Board Memo: Customer Churn Retention Program

## Headline Finding

Customer churn is **26.54%**, and the highest-value immediate opportunity is the **Urgent high-value** segment, where predicted churn risk and monthly revenue exposure are both high.

## Recommendation and Trade-offs

Fund the **Urgent high-value** segment first. It represents approximately **$93,081.80 in expected monthly revenue at risk**. Prioritize early-tenure customers with high predicted risk using onboarding support, service issue resolution, and carefully approved retention offers.

The trade-off is cost: targeted discounts and additional support reduce short-term margin. This is preferable to broad discounts because the program focuses spending where the expected revenue protected is highest. Logistic Regression is currently the safer operational model than Random Forest because its test recall is **78.34%**, while Random Forest recall is **47.86%** and its training accuracy of **99.80%** indicates overfitting.

## Monitoring and Retraining Signal

Create a weekly monitoring alert for **competitor price drops or major competitor promotions** in the same service areas. Start a model review when a verified competitor offer is at least **5% cheaper** than the comparable company plan, or when the model's test recall falls below **70%** for two consecutive weeks. Also monitor monthly data drift in contract type, monthly charges, tenure, and churn rate. Retrain only after confirming the signal with new labeled churn outcomes, then compare the candidate model against the current model on a fixed holdout set.

## Audit and Governance

For every retention recommendation, write an immutable audit record containing:

- Customer pseudonymous ID and timestamp
- Model version, data version, risk score, and selected clause ID
- The exact retrieved clause text and source/version citation
- The non-demographic feature names used in the explanation
- Confirmation that `gender`, `SeniorCitizen`, `Partner`, and `Dependents` were absent from the LLM payload
- The offer shown, agent decision, and final customer outcome

Run a monthly fairness audit comparing risk flags, offers, approval rates, and outcomes across protected groups in a restricted audit environment. Protected fields must be excluded from the decision and LLM payload but retained separately, with access controls, only for independent fairness testing. Block deployment if unexplained material disparities appear.

## Weekly LLM Cost Control

Use a weekly batch process with deterministic clause retrieval first. Send the LLM only the clause ID, exact clause text, approved citation, risk summary, and allowed top features. Cache identical clause-and-feature combinations, use a smaller model for routine 3-4 sentence explanations, and reserve a larger model for low-confidence or exception cases. Validate every response automatically for sentence count, prohibited demographic terms, and exact clause citation; retry or route failures to a human rather than generating an uncited explanation. Store the prompt, retrieved clause, model version, response, and validation result for auditability.

**Decision requested:** Approve a targeted pilot for the Urgent high-value segment with weekly drift monitoring, mandatory clause citations, and a documented fairness audit before expansion.
