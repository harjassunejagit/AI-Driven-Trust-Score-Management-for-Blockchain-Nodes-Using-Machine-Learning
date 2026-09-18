import os, sys, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Windows consoles default stdout to cp1252, which can't encode the
# checkmark characters this script prints — force UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RESULT_DIR = "results"
TRUST_LOG  = "trust_log.csv"
os.makedirs(RESULT_DIR, exist_ok=True)

def plot_trust_scores(trust_csv=TRUST_LOG):
    if not os.path.exists(trust_csv): print(f"Not found: {trust_csv}"); return
    df = pd.read_csv(trust_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    plt.figure(figsize=(14,7))
    for node in df["node"].unique()[:10]:
        nd = df[df["node"]==node]
        plt.plot(nd["timestamp"], nd["final_score"],  # FIX: was "new_score"
                 marker="o", linewidth=2, label=node[:10]+"...")
    plt.title("Final Trust Score Evolution"); plt.xlabel("Timestamp"); plt.ylabel("Trust Score")
    plt.ylim(0,100); plt.grid(True); plt.xticks(rotation=30); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR,"trust_score_evolution.png"),dpi=300); plt.close()
    print("✓ trust_score_evolution.png")

def compare_scores(trust_csv=TRUST_LOG):
    if not os.path.exists(trust_csv): print(f"Not found: {trust_csv}"); return
    df     = pd.read_csv(trust_csv)
    latest = df.groupby("node").tail(1)
    avg = {"Random Forest": latest["rf_score"].mean(),
           "XGBoost":       latest["xgb_score"].mean(),
           "Logistic":      latest["logistic_score"].mean(),
           "SVM":           latest["svm_score"].mean(),
           "Decision Tree": latest["decision_tree_score"].mean(),
           "ML Ensemble":   latest["ml_score"].mean(),
           "Graph Score":   latest["graph_score"].mean(),
           "Final Score":   latest["final_score"].mean()}
    plt.figure(figsize=(10,5))
    bars = plt.bar(avg.keys(), avg.values(),
                   color=["#4C72B0","#55A868","#C44E52","#8172B2","#CCB974",
                          "#64B5CD","#E07B39","#2E75B6"])
    for bar, val in zip(bars, avg.values()):
        plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f"{val:.1f}", ha="center", fontsize=9)
    plt.ylabel("Average Trust Score"); plt.ylim(0,110)
    plt.title("Average Trust Score per Model / Stage")
    plt.xticks(rotation=20,ha="right"); plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR,"trust_score_comparison.png"),dpi=300); plt.close()
    print("✓ trust_score_comparison.png")

def trust_distribution(trust_csv=TRUST_LOG):
    if not os.path.exists(trust_csv): print(f"Not found: {trust_csv}"); return
    df     = pd.read_csv(trust_csv)
    latest = df.groupby("node").tail(1)
    plt.figure(figsize=(8,5))
    plt.hist(latest["final_score"],bins=20,color="#4C72B0",edgecolor="white",alpha=0.85)
    plt.axvline(40,color="orange",linestyle="--",label="Low/Medium (40)")
    plt.axvline(70,color="red",   linestyle="--",label="Medium/High (70)")
    plt.xlabel("Final Trust Score"); plt.ylabel("Number of Nodes")
    plt.title("Distribution of Final Trust Scores"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR,"trust_distribution.png"),dpi=300); plt.close()
    print("✓ trust_distribution.png")

def trust_tiers(trust_csv=TRUST_LOG):
    if not os.path.exists(trust_csv): print(f"Not found: {trust_csv}"); return
    df     = pd.read_csv(trust_csv)
    latest = df.groupby("node").tail(1)
    high   = (latest["final_score"] >= 70).sum()
    medium = ((latest["final_score"] >= 40) & (latest["final_score"] < 70)).sum()
    low    = (latest["final_score"] < 40).sum()
    plt.figure(figsize=(7,5))
    bars = plt.bar(["High (>=70)","Medium (40-69)","Low (<40)"],
                   [high,medium,low], color=["#55A868","#FFA500","#C44E52"])
    for bar,val in zip(bars,[high,medium,low]):
        plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 str(int(val)), ha="center", fontsize=12)
    plt.ylabel("Number of Nodes"); plt.title("Trust Tier Distribution"); plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR,"trust_tier_distribution.png"),dpi=300); plt.close()
    print("✓ trust_tier_distribution.png")

if __name__ == "__main__":
    plot_trust_scores(); compare_scores(); trust_distribution(); trust_tiers()
    print("\n" + "="*60 + "\nVisualization Completed Successfully\n" + "="*60)
