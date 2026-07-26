import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("data/ethereum_fraud.csv")
df.columns = df.columns.str.strip()

drop_cols = [
    "FLAG", "Unnamed: 0", "Index", "Address",
    "ERC20 most sent token type", "ERC20_most_rec_token_type"
]
feature_cols = [c for c in df.columns if c not in drop_cols]
X = df[feature_cols].fillna(0)

scaler   = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

model = IsolationForest(n_estimators=200, contamination=0.1, random_state=42)
model.fit(X_scaled)

predictions = model.predict(X_scaled)
scores      = model.decision_function(X_scaled)

df["Anomaly"]       = predictions
df["Anomaly_Score"] = scores

joblib.dump(model,  "isolation_forest.pkl")
joblib.dump(scaler, "if_scaler.pkl")
df.to_csv("anomaly_results.csv", index=False)

print("\nIsolation Forest trained successfully.")
print(f"  n_estimators  : 200  |  contamination : 0.1  |  random_state : 42")
print(f"Normal Samples  : {(predictions ==  1).sum()}")
print(f"Anomaly Samples : {(predictions == -1).sum()}")
print("\nGenerated: isolation_forest.pkl  if_scaler.pkl  anomaly_results.csv")
