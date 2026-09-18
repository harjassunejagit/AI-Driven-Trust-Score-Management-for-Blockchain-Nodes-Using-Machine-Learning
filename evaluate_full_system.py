# evaluate_full_system.py
# ─────────────────────────────────────────────────────────────
# Evaluates the COMPLETE proposed trust score management system
# Run AFTER auto_trust_engine.py has generated trust scores
# ─────────────────────────────────────────────────────────────

import sys
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score
)
from scipy.stats import spearmanr

# Windows consoles default stdout to cp1252, which can't encode the
# box-drawing/checkmark characters this script prints — force UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Load computed trust scores ────────────────────────────────
# auto_trust_engine.py appends one row per node per iteration to
# trust_log.csv — take each node's latest score and normalize the
# column names to what the rest of this script expects.
trust_log = pd.read_csv("trust_log.csv")
trust = (trust_log.groupby("node").tail(1)
         .rename(columns={"node": "address", "final_score": "final_trust"})
         [["address", "final_trust"]])

# ── Load ground truth labels ──────────────────────────────────
# The dataset has 25 duplicated Address rows (same FLAG both times) —
# drop_duplicates first so the merge doesn't double-count them.
gt = pd.read_csv("data/ethereum_fraud.csv")[["Address", "FLAG"]].drop_duplicates(subset="Address")

# ── Merge on address ──────────────────────────────────────────
df = trust.merge(gt, left_on="address", right_on="Address")

print("="*60)
print("PROPOSED SYSTEM — FULL EVALUATION")
print("="*60)
print(f"Total addresses evaluated: {len(df)}")
print()

# ─────────────────────────────────────────────────────────────
# PART 1 — Binary prediction from trust tier boundary
# LOW tier (score < 40) = predicted fraudulent
# ─────────────────────────────────────────────────────────────
df["predicted"] = (df["final_trust"] < 40).astype(int)

print("─── Part 1: Binary Prediction (LOW tier = fraud) ───")
print(f"Accuracy  : {accuracy_score(df.FLAG, df.predicted):.4f}")
print(f"Precision : {precision_score(df.FLAG, df.predicted):.4f}")
print(f"Recall    : {recall_score(df.FLAG, df.predicted):.4f}")
print(f"F1-Score  : {f1_score(df.FLAG, df.predicted):.4f}")
print()

# ─────────────────────────────────────────────────────────────
# PART 2 — Trust score separation
# Mean trust score for benign vs fraudulent addresses
# ─────────────────────────────────────────────────────────────
benign_trust = df[df.FLAG == 0]["final_trust"].mean()
fraud_trust  = df[df.FLAG == 1]["final_trust"].mean()
separation   = benign_trust - fraud_trust

print("─── Part 2: Trust Score Separation ───")
print(f"Mean trust score (Benign)    : {benign_trust:.2f}")
print(f"Mean trust score (Fraudulent): {fraud_trust:.2f}")
print(f"Separation gap               : {separation:.2f} points")
print()

# ─────────────────────────────────────────────────────────────
# PART 3 — Trust tier validation
# Fraud rate inside each tier
# ─────────────────────────────────────────────────────────────
print("─── Part 3: Trust Tier Validation ───")
for tier, lo, hi in [("HIGH  (>=70)", 70, 101),
                      ("MEDIUM (40-69)", 40, 70),
                      ("LOW   (<40)", 0, 40)]:
    tier_df     = df[(df.final_trust >= lo) & (df.final_trust < hi)]
    count       = len(tier_df)
    fraud_rate  = tier_df["FLAG"].mean() * 100 if count > 0 else 0
    print(f"  {tier}: {count:5d} addresses | "
          f"Actual fraud rate: {fraud_rate:.1f}%")
print()

# ─────────────────────────────────────────────────────────────
# PART 4 — Fraud containment rate
# What % of actual frauds land in LOW tier
# ─────────────────────────────────────────────────────────────
fraud_df   = df[df.FLAG == 1]
contained  = (fraud_df["final_trust"] < 40).sum()
containment = contained / len(fraud_df) * 100

print("─── Part 4: Fraud Containment Rate ───")
print(f"Fraudulent addresses in LOW tier : "
      f"{contained}/{len(fraud_df)} = {containment:.1f}%")
print()

# ─────────────────────────────────────────────────────────────
# PART 5 — False alarm rate
# What % of benign addresses land in LOW tier
# ─────────────────────────────────────────────────────────────
benign_df   = df[df.FLAG == 0]
false_alarm = (benign_df["final_trust"] < 40).sum()
far         = false_alarm / len(benign_df) * 100

print("─── Part 5: False Alarm Rate ───")
print(f"Benign addresses wrongly in LOW tier: "
      f"{false_alarm}/{len(benign_df)} = {far:.1f}%")
print()

# ─────────────────────────────────────────────────────────────
# PART 6 — Spearman correlation
# Trust score vs ground truth legitimacy
# ─────────────────────────────────────────────────────────────
corr, pval = spearmanr(df["final_trust"], 1 - df["FLAG"])

print("─── Part 6: Spearman Correlation ───")
print(f"Correlation (trust vs legitimacy): {corr:.4f}")
print(f"P-value                          : {pval:.6f}")
print(f"Significance: {'✓ Significant (p<0.001)' if pval < 0.001 else 'Not significant'}")
print()

# ─────────────────────────────────────────────────────────────
# PART 7 — Trust score distribution per class
# Mean, median, std for benign and fraudulent
# ─────────────────────────────────────────────────────────────
print("─── Part 7: Trust Score Distribution ───")
print(f"{'Category':<15} {'Count':>8} {'Mean':>8} "
      f"{'Median':>8} {'Std':>8}")
print("-"*50)
for label, name in [(0, "Benign"), (1, "Fraudulent")]:
    group = df[df.FLAG == label]["final_trust"]
    print(f"{name:<15} {len(group):>8} "
          f"{group.mean():>8.2f} "
          f"{group.median():>8.2f} "
          f"{group.std():>8.2f}")
print()

# ─────────────────────────────────────────────────────────────
# PART 8 — Save results to CSV
# ─────────────────────────────────────────────────────────────
results = pd.DataFrame([{
    "Metric": "Full System Accuracy (LOW=fraud)",
    "Value": round(accuracy_score(df.FLAG, df.predicted), 4)
}, {
    "Metric": "Full System Precision",
    "Value": round(precision_score(df.FLAG, df.predicted), 4)
}, {
    "Metric": "Full System Recall",
    "Value": round(recall_score(df.FLAG, df.predicted), 4)
}, {
    "Metric": "Full System F1",
    "Value": round(f1_score(df.FLAG, df.predicted), 4)
}, {
    "Metric": "Mean Trust - Benign",
    "Value": round(benign_trust, 2)
}, {
    "Metric": "Mean Trust - Fraudulent",
    "Value": round(fraud_trust, 2)
}, {
    "Metric": "Trust Separation Gap",
    "Value": round(separation, 2)
}, {
    "Metric": "Fraud Containment Rate (%)",
    "Value": round(containment, 2)
}, {
    "Metric": "False Alarm Rate (%)",
    "Value": round(far, 2)
}, {
    "Metric": "Spearman Correlation",
    "Value": round(corr, 4)
}])

results.to_csv("full_system_evaluation.csv", index=False)

print("="*60)
print("Full system evaluation saved to full_system_evaluation.csv")
print("="*60)