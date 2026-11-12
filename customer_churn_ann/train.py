from pathlib import Path
import copy, json
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUT = ROOT / "artifacts"
OUT.mkdir(exist_ok=True)
SEED, BATCH, EPOCHS, LR, PATIENCE, THRESHOLD = 42, 64, 100, 0.001, 10, 0.5
torch.manual_seed(SEED)
np.random.seed(SEED)

class ChurnANN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.30),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.network(x)

def main():
    if not DATA.exists():
        raise FileNotFoundError(
            f"Dataset missing: {DATA}\\nAdd the IBM Telco Customer Churn CSV to the data folder."
        )
    df = pd.read_csv(DATA).drop(columns=["customerID"], errors="ignore")
    if "Churn" not in df.columns:
        raise ValueError("Expected target column: Churn")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    y = df.pop("Churn").map({"No": 0, "Yes": 1})
    if y.isna().any():
        raise ValueError("Churn must contain only Yes/No values.")
    X = df
    nums = X.select_dtypes(include=["number"]).columns.tolist()
    cats = X.select_dtypes(exclude=["number"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), nums),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cats)
    ])

    # Split first; fit preprocessing on training data only to prevent leakage.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.20, random_state=SEED, stratify=y_trainval)

    Xt = preprocessor.fit_transform(X_train).astype(np.float32)
    Xv = preprocessor.transform(X_val).astype(np.float32)
    Xtest = preprocessor.transform(X_test).astype(np.float32)
    yt = y_train.to_numpy(dtype=np.float32)
    yv = y_val.to_numpy(dtype=np.float32)
    ytest = y_test.to_numpy(dtype=np.int64)

    def loader(a, b, shuffle):
        return DataLoader(TensorDataset(torch.tensor(a), torch.tensor(b).view(-1, 1)),
                          batch_size=BATCH, shuffle=shuffle)
    train_loader, val_loader = loader(Xt, yt, True), loader(Xv, yv, False)

    model = ChurnANN(Xt.shape[1])
    positives = max(float(yt.sum()), 1.0)
    pos_weight = torch.tensor([(len(yt) - yt.sum()) / positives], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_loss, best_state, wait = float("inf"), copy.deepcopy(model.state_dict()), 0
    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_total += loss_fn(model(xb), yb).item() * len(xb)
        tr_loss = total / len(train_loader.dataset)
        va_loss = val_total / len(val_loader.dataset)
        print(f"Epoch {epoch+1:03d}: train_loss={tr_loss:.4f}, val_loss={va_loss:.4f}")

        if va_loss < best_loss:
            best_loss, best_state, wait = va_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print("Early stopping.")
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.tensor(Xtest))).squeeze(1).numpy()
    preds = (probs >= THRESHOLD).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(ytest, preds)),
        "precision": float(precision_score(ytest, preds, zero_division=0)),
        "recall": float(recall_score(ytest, preds, zero_division=0)),
        "f1_score": float(f1_score(ytest, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(ytest, probs)),
        "threshold": THRESHOLD,
        "confusion_matrix": confusion_matrix(ytest, preds).tolist()
    }
    torch.save({"model_state_dict": model.state_dict(),
                "input_dim": int(Xt.shape[1]), "threshold": THRESHOLD}, OUT / "churn_ann.pt")
    joblib.dump(preprocessor, OUT / "preprocessor.joblib")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\\nTest metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
