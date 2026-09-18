# cross_validation.py
# 5-fold stratified cross-validation for all five classifiers (mean ± std),
# plus PR-AUC (average precision) alongside ROC-AUC, and a McNemar's test
# comparing XGBoost vs Random Forest on the held-out test split.

import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

df = pd.read_csv("data/ethereum_fraud.csv")
df.columns = df.columns.str.strip()
drop_columns = ["FLAG", "Unnamed: 0", "Index", "Address",
                "ERC20 most sent token type", "ERC20_most_rec_token_type"]
feature_cols = [c for c in df.columns if c not in drop_columns]
X = df[feature_cols].fillna(0)
y = df["FLAG"]

def make_models():
    return {
        "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=42), True),
        "SVM":                 (SVC(kernel="rbf", probability=True, random_state=42), True),
        "Decision Tree":       (DecisionTreeClassifier(random_state=42), False),
        "Random Forest":       (RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1), False),
        "XGBoost":             (XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=8,
                                               subsample=0.9, colsample_bytree=0.9,
                                               random_state=42, eval_metric="logloss"), False),
    }

print("="*70 + "\n5-FOLD STRATIFIED CROSS-VALIDATION\n" + "="*70)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_rows = []
for name, (_, needs_scaling) in make_models().items():
    print(f"\n{name}...")
    fold_metrics = {"Accuracy": [], "Precision": [], "Recall": [], "F1": [], "ROC_AUC": [], "PR_AUC": []}
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        model, needs_scaling_f = make_models()[name]
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        if needs_scaling_f:
            scaler = MinMaxScaler().fit(X_train)
            X_train_f, X_test_f = scaler.transform(X_train), scaler.transform(X_test)
        else:
            X_train_f, X_test_f = X_train, X_test
        model.fit(X_train_f, y_train)
        pred = model.predict(X_test_f)
        prob = model.predict_proba(X_test_f)[:, 1]
        fold_metrics["Accuracy"].append(accuracy_score(y_test, pred))
        fold_metrics["Precision"].append(precision_score(y_test, pred, zero_division=0))
        fold_metrics["Recall"].append(recall_score(y_test, pred, zero_division=0))
        fold_metrics["F1"].append(f1_score(y_test, pred, zero_division=0))
        fold_metrics["ROC_AUC"].append(roc_auc_score(y_test, prob))
        fold_metrics["PR_AUC"].append(average_precision_score(y_test, prob))
        print(f"  fold {fold}: acc={fold_metrics['Accuracy'][-1]:.4f}  "
              f"auc={fold_metrics['ROC_AUC'][-1]:.4f}  pr_auc={fold_metrics['PR_AUC'][-1]:.4f}")
    row = {"Model": name}
    for k, vals in fold_metrics.items():
        row[f"{k}_mean"] = round(float(np.mean(vals)), 4)
        row[f"{k}_std"]  = round(float(np.std(vals)), 4)
    cv_rows.append(row)

cv_df = pd.DataFrame(cv_rows)
cv_df.to_csv("results/cross_validation.csv", index=False)
print("\n" + cv_df.to_string(index=False))
print("\nSaved: results/cross_validation.csv")

# ── PR-AUC on the original single 75/25 split (Table 5/6 companion) ────
print("\n" + "="*70 + "\nPR-AUC ON THE ORIGINAL 75/25 SPLIT (Table 5/6 companion)\n" + "="*70)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)
scaler = MinMaxScaler().fit(X_train)
X_train_scaled, X_test_scaled = scaler.transform(X_train), scaler.transform(X_test)

split_models = {
    "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=42), X_train_scaled, X_test_scaled),
    "SVM":                 (SVC(kernel="rbf", probability=True, random_state=42), X_train_scaled, X_test_scaled),
    "Decision Tree":       (DecisionTreeClassifier(random_state=42), X_train, X_test),
    "Random Forest":       (RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1), X_train, X_test),
    "XGBoost":             (XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=8,
                                          subsample=0.9, colsample_bytree=0.9,
                                          random_state=42, eval_metric="logloss"), X_train, X_test),
}
pr_auc_rows = []
preds_by_model = {}
for name, (model, Xtr, Xte) in split_models.items():
    model.fit(Xtr, y_train)
    prob = model.predict_proba(Xte)[:, 1]
    pred = model.predict(Xte)
    preds_by_model[name] = pred
    pr_auc = average_precision_score(y_test, prob)
    roc_auc = roc_auc_score(y_test, prob)
    pr_auc_rows.append({"Model": name, "ROC_AUC": round(roc_auc, 4), "PR_AUC": round(pr_auc, 4)})
    print(f"{name:22s}  ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}")
pd.DataFrame(pr_auc_rows).to_csv("results/pr_auc.csv", index=False)

# ── McNemar's test: XGBoost vs Random Forest on the same test set ──────
print("\n" + "="*70 + "\nMCNEMAR'S TEST: XGBoost vs Random Forest\n" + "="*70)
xgb_correct = (preds_by_model["XGBoost"] == y_test.values)
rf_correct  = (preds_by_model["Random Forest"] == y_test.values)
# contingency: both correct / xgb only / rf only / both wrong
both_correct = int((xgb_correct & rf_correct).sum())
xgb_only     = int((xgb_correct & ~rf_correct).sum())
rf_only      = int((~xgb_correct & rf_correct).sum())
both_wrong   = int((~xgb_correct & ~rf_correct).sum())
print(f"Both correct: {both_correct}  XGB-only-correct: {xgb_only}  "
      f"RF-only-correct: {rf_only}  Both wrong: {both_wrong}")

from scipy.stats import binomtest
n = xgb_only + rf_only
if n > 0:
    result = binomtest(min(xgb_only, rf_only), n, 0.5)
    p_value = result.pvalue
else:
    p_value = 1.0
print(f"McNemar exact (binomial) test on discordant pairs (n={n}): p={p_value:.6f}")
print("Significant (p<0.05):", "YES" if p_value < 0.05 else "NO")

pd.DataFrame([{
    "both_correct": both_correct, "xgb_only_correct": xgb_only,
    "rf_only_correct": rf_only, "both_wrong": both_wrong,
    "discordant_n": n, "p_value": round(p_value, 6),
    "significant_at_0.05": p_value < 0.05,
}]).to_csv("results/mcnemar_xgb_vs_rf.csv", index=False)
print("\nSaved: results/pr_auc.csv, results/mcnemar_xgb_vs_rf.csv")
