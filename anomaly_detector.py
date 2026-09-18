import sys
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

# Windows consoles default stdout to cp1252, which can't encode the
# characters this script prints — force UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

df = pd.read_csv("data/ethereum_fraud.csv")
df.columns = df.columns.str.strip()

drop_cols = [
    "FLAG", "Unnamed: 0", "Index", "Address",
    "ERC20 most sent token type", "ERC20_most_rec_token_type"
]
feature_cols = [c for c in df.columns if c not in drop_cols]
X = df[feature_cols].fillna(0)
y = df["FLAG"]

# Fit on the same train split used by train_model.py (same test_size,
# random_state and stratify) so the anomaly model never sees the test
# addresses during fitting — avoids the train/test leakage that comes
# from fitting on the full dataset, while still letting every address
# (train and test) receive an anomaly score for production use below.
X_train, X_test = train_test_split(
    X, test_size=0.25, random_state=42, stratify=y
)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_scaled = scaler.transform(X)

model = IsolationForest(n_estimators=200, contamination=0.1, random_state=42)
model.fit(scaler.transform(X_train))

predictions = model.predict(X_scaled)
scores      = model.decision_function(X_scaled)

df["Anomaly"]       = predictions
df["Anomaly_Score"] = scores

joblib.dump(model,  "isolation_forest.pkl")
joblib.dump(scaler, "if_scaler.pkl")
df.to_csv("anomaly_results.csv", index=False)

print("\nIsolation Forest trained successfully.")
print(f"  n_estimators  : 200  |  contamination : 0.1  |  random_state : 42")
print(f"  Fit on {len(X_train)} training-split rows only (no test leakage);")
print(f"  scored all {len(df)} rows for production anomaly labels.")
print(f"Normal Samples  : {(predictions ==  1).sum()}")
print(f"Anomaly Samples : {(predictions == -1).sum()}")
print("\nGenerated: isolation_forest.pkl  if_scaler.pkl  anomaly_results.csv")
