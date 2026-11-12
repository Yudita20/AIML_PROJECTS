# Interview notes: Customer Churn Prediction using ANN

- **Problem:** Binary classification; 0 = no churn, 1 = churn.
- **Inputs:** Tenure, monthly/total charges, contract, payment method, internet/service options, and customer attributes.
- **Why ANN?** It can learn nonlinear feature relationships. It is a candidate model, not automatically the best model for tabular data.
- **Why one-hot encoding?** Converts categorical inputs into numeric indicator features without imposing an artificial ranking.
- **Why StandardScaler?** Puts numerical features on comparable scales, which can help neural-network optimization.
- **Output:** One logit; sigmoid converts it to a probability from 0 to 1.
- **Loss:** BCEWithLogitsLoss for binary classification.
- **Optimizer:** Adam updates parameters using gradients calculated by backpropagation.
- **Overfitting controls:** Dropout and early stopping based on validation loss.
- **Metrics:** Accuracy, precision, recall, F1, ROC-AUC, and confusion matrix. For churn, recall may matter because missing at-risk customers can be costly; business costs determine the trade-off.
- **Data leakage prevention:** Split before fitting preprocessing; fit the preprocessor only on training data.
- **Results:** Quote only metrics from your own `artifacts/metrics.json` after running the project. Never invent a score.
