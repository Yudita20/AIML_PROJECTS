# Customer Churn Prediction using ANN

A binary-classification project that estimates whether a telecom customer may discontinue service, using a feed-forward Artificial Neural Network (ANN), scikit-learn preprocessing, PyTorch training, and a Streamlit interface.

## Stack
Python, Pandas, NumPy, Scikit-learn, PyTorch, Streamlit.

## Dataset
Obtain the IBM Telco Customer Churn CSV from a legitimate source and place it at:
`data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

Expected target: `Churn` with values `Yes` and `No`. Retain dataset attribution/license details.

## Run
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train.py
streamlit run app.py
```

Training creates `artifacts/churn_ann.pt`, `artifacts/preprocessor.joblib`, and `artifacts/metrics.json`.

## Workflow
1. Drop customer ID and convert TotalCharges to numeric.
2. Split into train, validation, and test sets before fitting preprocessing.
3. Impute and scale numeric columns; impute and one-hot encode categorical columns.
4. Train an ANN with 64- and 32-neuron hidden layers, ReLU, and dropout.
5. Use BCEWithLogitsLoss and Adam; early-stop based on validation loss.
6. Evaluate on the held-out test set with accuracy, precision, recall, F1, ROC-AUC, and confusion matrix.
7. Serve predictions through Streamlit.

## Important points
- ANN is suitable for learning nonlinear relationships among customer features, but is not guaranteed to outperform tree-based models on tabular data.
- Sigmoid converts the output logit into a churn probability.
- BCEWithLogitsLoss combines sigmoid and binary cross-entropy stably.
- Backpropagation calculates gradients; Adam uses them to update weights.
- Dropout and early stopping can help reduce overfitting.
- Preprocessing is fitted only on training data to prevent data leakage.

## Live Demo

**Try the application:** [ChurnSense – Customer Churn Prediction](YOUR_DEPLOYED_STREAMLIT_URL)

Explore the application by entering customer details and getting ANN-based churn probability and risk predictions.

