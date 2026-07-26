import pandas as pd, joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
model        = joblib.load("fraud_model.pkl")
feature_cols = joblib.load("feature_cols.pkl")
df = pd.read_csv("data/ethereum_fraud.csv"); df.columns = df.columns.str.strip()
X  = df.reindex(columns=feature_cols, fill_value=0); y = df["FLAG"]
y_pred = model.predict(X)
print("Accuracy:", accuracy_score(y, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y, y_pred))
print("\nClassification Report:\n", classification_report(y, y_pred))
