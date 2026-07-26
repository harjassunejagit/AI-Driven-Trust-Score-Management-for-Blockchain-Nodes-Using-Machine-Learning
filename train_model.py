import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

df = pd.read_csv("data/ethereum_fraud.csv")
df.columns = df.columns.str.strip()

drop_columns = [
    "FLAG", "Unnamed: 0", "Index", "Address",
    "ERC20 most sent token type", "ERC20_most_rec_token_type"
]
feature_cols = [col for col in df.columns if col not in drop_columns]
X = df[feature_cols].fillna(0)
y = df["FLAG"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

logistic      = LogisticRegression(max_iter=1000, random_state=42)
svm           = SVC(kernel="rbf", probability=True, random_state=42)
decision_tree = DecisionTreeClassifier(random_state=42)
random_forest = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
xgboost       = XGBClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=8,
    subsample=0.9, colsample_bytree=0.9,
    random_state=42, eval_metric="logloss"
)

print("\nTraining Logistic Regression...")
logistic.fit(X_train_scaled, y_train)
print("Training SVM...")
svm.fit(X_train_scaled, y_train)
print("Training Decision Tree...")
decision_tree.fit(X_train, y_train)
print("Training Random Forest...")
random_forest.fit(X_train, y_train)
print("Training XGBoost...")
xgboost.fit(X_train, y_train)

def evaluate(name, y_true, pred, prob):
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    print("Accuracy  :", round(accuracy_score(y_true, pred), 4))
    print("Precision :", round(precision_score(y_true, pred), 4))
    print("Recall    :", round(recall_score(y_true, pred), 4))
    print("F1 Score  :", round(f1_score(y_true, pred), 4))
    print("ROC AUC   :", round(roc_auc_score(y_true, prob), 4))
    print("\nConfusion Matrix\n", confusion_matrix(y_true, pred))
    print("\nClassification Report\n", classification_report(y_true, pred))

logistic_pred = logistic.predict(X_test_scaled)
logistic_prob = logistic.predict_proba(X_test_scaled)[:, 1]
svm_pred      = svm.predict(X_test_scaled)
svm_prob      = svm.predict_proba(X_test_scaled)[:, 1]
dt_pred       = decision_tree.predict(X_test)
dt_prob       = decision_tree.predict_proba(X_test)[:, 1]
rf_pred       = random_forest.predict(X_test)
rf_prob       = random_forest.predict_proba(X_test)[:, 1]
xgb_pred      = xgboost.predict(X_test)
xgb_prob      = xgboost.predict_proba(X_test)[:, 1]

evaluate("Logistic Regression", y_test, logistic_pred, logistic_prob)
evaluate("SVM",                 y_test, svm_pred,      svm_prob)
evaluate("Decision Tree",       y_test, dt_pred,       dt_prob)
evaluate("Random Forest",       y_test, rf_pred,       rf_prob)
evaluate("XGBoost",             y_test, xgb_pred,      xgb_prob)

comparison = pd.DataFrame({
    "Model":     ["Logistic Regression","SVM","Decision Tree","Random Forest","XGBoost"],
    "Accuracy":  [accuracy_score(y_test,p) for p in [logistic_pred,svm_pred,dt_pred,rf_pred,xgb_pred]],
    "Precision": [precision_score(y_test,p) for p in [logistic_pred,svm_pred,dt_pred,rf_pred,xgb_pred]],
    "Recall":    [recall_score(y_test,p) for p in [logistic_pred,svm_pred,dt_pred,rf_pred,xgb_pred]],
    "F1 Score":  [f1_score(y_test,p) for p in [logistic_pred,svm_pred,dt_pred,rf_pred,xgb_pred]],
    "ROC AUC":   [roc_auc_score(y_test,p) for p in [logistic_prob,svm_prob,dt_prob,rf_prob,xgb_prob]],
})
comparison = comparison.sort_values("Accuracy", ascending=False)
comparison.to_csv("model_comparison.csv", index=False)
print("\n\nModel Comparison:\n", comparison)

joblib.dump(logistic,      "logistic_model.pkl")
joblib.dump(svm,           "svm_model.pkl")
joblib.dump(decision_tree, "decision_tree_model.pkl")
joblib.dump(random_forest, "fraud_model.pkl")
joblib.dump(xgboost,       "xgboost_model.pkl")
joblib.dump(scaler,        "rf_scaler.pkl")
joblib.dump(feature_cols,  "feature_cols.pkl")

print("\n" + "="*60)
print("Training Completed Successfully")
print("="*60)
print("Generated: logistic_model.pkl, svm_model.pkl, decision_tree_model.pkl")
print("           fraud_model.pkl, xgboost_model.pkl, rf_scaler.pkl, feature_cols.pkl")
