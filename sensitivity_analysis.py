
import sys
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# Windows consoles default stdout to cp1252, which can't encode the
# characters this script prints — force UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

TRUST_LOG       = "trust_log.csv"
ANOMALY_RESULTS = "anomaly_results.csv"
DATASET_PATH    = "data/ethereum_fraud.csv"
LOW_TIER        = 40

# ── Load the ML fraud probability + graph score already computed by
#    auto_trust_engine.py (one row per node per iteration — take the
#    latest), the anomaly flag from anomaly_detector.py, and ground truth.
trust_log = pd.read_csv(TRUST_LOG)
latest = (trust_log.groupby("node").tail(1)
          [["node", "fraud_probability", "graph_score"]]
          .rename(columns={"node": "Address"}))

anomaly = pd.read_csv(ANOMALY_RESULTS)[["Address", "Anomaly"]].drop_duplicates(subset="Address")

gt = pd.read_csv(DATASET_PATH)
gt.columns = gt.columns.str.strip()
gt = gt[["Address", "FLAG"]].drop_duplicates(subset="Address")

df = latest.merge(anomaly, on="Address").merge(gt, on="Address")
print(f"Evaluating sensitivity over {len(df)} addresses")

feature_trust = (1 - df["fraud_probability"]) * 100
graph_trust   = df["graph_score"]
is_anomaly    = (df["Anomaly"] == -1)
flag          = df["FLAG"]

# Test different values of lambda (feature/graph blend) and delta
# (anomaly penalty subtracted from feature_trust before blending).
lambdas = [0.30, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
deltas  = [0, 5, 10, 15, 20, 25]

results = []
for lam in lambdas:
    for delta in deltas:
        penalized = feature_trust.where(~is_anomaly, (feature_trust - delta).clip(lower=0))
        final = (lam * penalized + (1 - lam) * graph_trust).clip(0, 100)

        corr, _ = spearmanr(final, 1 - flag)
        low_tier = final < LOW_TIER
        containment_rate = low_tier[flag == 1].mean() * 100
        false_alarm_rate = low_tier[flag == 0].mean() * 100

        results.append({
            "lambda": lam,
            "delta": delta,
            "spearman": round(corr, 4),
            "containment": round(containment_rate, 2),
            "false_alarm": round(false_alarm_rate, 2),
        })

df_results = pd.DataFrame(results)
df_results.to_csv("sensitivity_analysis.csv", index=False)
print("\nTop 10 (lambda, delta) combinations by Spearman correlation:\n")
print(df_results.sort_values("spearman", ascending=False).head(10).to_string(index=False))
print("\nSaved full grid to sensitivity_analysis.csv")
