# ablation_study.py
# Produces exact (not "~approximate") numbers for every row of the
# ablation table: RF only, XGB only, 5-model ensemble, ensemble + IF
# penalty, ensemble + graph trust, and the full system — all evaluated
# as binary classifiers on the SAME held-out test split used everywhere
# else in the project (test_size=0.25, random_state=42, stratify=y),
# so every row is directly comparable.

import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from graph_trust import GraphTrust
import trust_fusion

DATASET_PATH     = "data/ethereum_fraud.csv"
TRANSACTION_PATH = "transactions.csv"
THRESHOLD        = 0.5   # matches trust-tier midpoint (trust=50 <=> p=0.5)

df = pd.read_csv(DATASET_PATH)
df.columns = df.columns.str.strip()

drop_cols = ["FLAG", "Unnamed: 0", "Index", "Address",
             "ERC20 most sent token type", "ERC20_most_rec_token_type"]
feature_cols = [c for c in df.columns if c not in drop_cols]
X = df[feature_cols].fillna(0)
y = df["FLAG"]

X_train, X_test, y_train, y_test, addr_train, addr_test = train_test_split(
    X, y, df["Address"], test_size=0.25, random_state=42, stratify=y
)

print("Loading models...")
rf_scaler           = joblib.load("rf_scaler.pkl")
rf_model            = joblib.load("fraud_model.pkl")
xgb_model           = joblib.load("xgboost_model.pkl")
logistic_model      = joblib.load("logistic_model.pkl")
svm_model           = joblib.load("svm_model.pkl")
decision_tree_model = joblib.load("decision_tree_model.pkl")
if_model            = joblib.load("isolation_forest.pkl")
if_scaler           = joblib.load("if_scaler.pkl")

X_test_scaled = rf_scaler.transform(X_test)
X_full_scaled = rf_scaler.transform(X)

def ensemble_prob(X_raw, X_scaled):
    rf_p  = rf_model.predict_proba(X_raw)[:, 1]
    xgb_p = xgb_model.predict_proba(X_raw)[:, 1]
    lr_p  = logistic_model.predict_proba(X_scaled)[:, 1]
    sv_p  = svm_model.predict_proba(X_scaled)[:, 1]
    dt_p  = decision_tree_model.predict_proba(X_raw)[:, 1]
    return rf_p*0.30 + xgb_p*0.30 + lr_p*0.15 + sv_p*0.15 + dt_p*0.10

def metrics(name, pred, score):
    return {
        "Configuration": name,
        "Accuracy":  round(accuracy_score(y_test, pred), 4),
        "Precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, pred, zero_division=0), 4),
        "F1":        round(f1_score(y_test, pred, zero_division=0), 4),
        "AUC":       round(roc_auc_score(y_test, score), 4),
    }

results = []

# ── Rows 1-2: single models (RF only, XGB only) ────────────────────
rf_prob_test  = rf_model.predict_proba(X_test)[:, 1]
xgb_prob_test = xgb_model.predict_proba(X_test)[:, 1]
# sklearn's .predict() breaks ties at exactly 0.5 toward the lower class
# index (benign) via argmax — use a strict ">" here so rows 1-2 match
# Table 5/6 exactly instead of drifting by the handful of exact-tie rows.
results.append(metrics("RF only (w=1.0)",  (rf_prob_test  > THRESHOLD).astype(int), rf_prob_test))
results.append(metrics("XGB only (w=1.0)", (xgb_prob_test > THRESHOLD).astype(int), xgb_prob_test))

# ── Row 3: 5-model weighted ensemble, no IF, no graph ──────────────
p_ens_test = ensemble_prob(X_test, X_test_scaled)
results.append(metrics("5-model ensemble (no IF, no graph)",
                        (p_ens_test > THRESHOLD).astype(int), p_ens_test))

# ── Anomaly labels + feature trust, computed for the FULL dataset so
#    graph neighbours outside the test split still have a trust value.
p_ens_full = ensemble_prob(X, X_full_scaled)
feature_trust_full = (1 - p_ens_full) * 100

anomaly_full = if_model.predict(if_scaler.transform(X))
feature_trust_full_penalized = np.where(
    anomaly_full == -1,
    np.clip(feature_trust_full - trust_fusion.ANOMALY_PENALTY, 0, None),
    feature_trust_full,
)

feature_trust_test           = pd.Series(feature_trust_full, index=X.index).loc[addr_test.index].values
feature_trust_test_penalized = pd.Series(feature_trust_full_penalized, index=X.index).loc[addr_test.index].values

# ── Row 4: Ensemble + IF penalty, no graph ─────────────────────────
pred_if = (feature_trust_test_penalized < 50).astype(int)
score_if = 100 - feature_trust_test_penalized
results.append(metrics("Ensemble + IF penalty (no graph)", pred_if, score_if))

# ── Graph trust over the REAL behavioural-similarity graph ─────────
print("Building transaction graph...")
gt_plain = GraphTrust()
gt_plain.build_graph(pd.read_csv(TRANSACTION_PATH))
gt_plain.set_feature_trust(dict(zip(df["Address"], feature_trust_full)))
graph_trust_test = np.array([gt_plain.graph_trust(a) for a in addr_test])

gt_penalized = GraphTrust()
gt_penalized.build_graph(pd.read_csv(TRANSACTION_PATH))
gt_penalized.set_feature_trust(dict(zip(df["Address"], feature_trust_full_penalized)))
graph_trust_test_penalized = np.array([gt_penalized.graph_trust(a) for a in addr_test])

# ── Row 5: Ensemble + graph trust, no IF ───────────────────────────
final_no_if = np.clip(trust_fusion.LAMBDA * feature_trust_test
                       + (1 - trust_fusion.LAMBDA) * graph_trust_test, 0, 100)
pred_graph = (final_no_if < 50).astype(int)
score_graph = 100 - final_no_if
results.append(metrics("Ensemble + graph trust (no IF)", pred_graph, score_graph))

# ── Row 6: Full system (ensemble + IF + graph) ─────────────────────
final_full = np.clip(trust_fusion.LAMBDA * feature_trust_test_penalized
                      + (1 - trust_fusion.LAMBDA) * graph_trust_test_penalized, 0, 100)
pred_full = (final_full < 50).astype(int)
score_full = 100 - final_full
results.append(metrics("Full system (ensemble + IF + graph)", pred_full, score_full))

out = pd.DataFrame(results)
out.to_csv("ablation_study.csv", index=False)
print("\n" + "="*70)
print(out.to_string(index=False))
print("="*70)
print("\nSaved to ablation_study.csv")
