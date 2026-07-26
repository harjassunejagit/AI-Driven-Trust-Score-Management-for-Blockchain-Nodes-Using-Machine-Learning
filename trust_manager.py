import os
import pandas as pd

TRUST_LOG   = "trust_log.csv"
OUTPUT_FILE = "trust_scores.csv"

def classify_tier(score):
    if score >= 70: return "HIGH"
    elif score >= 40: return "MEDIUM"
    else: return "LOW"

def run():
    if not os.path.exists(TRUST_LOG):
        print(f"Error: {TRUST_LOG} not found. Run auto_trust_engine.py first."); return
    df     = pd.read_csv(TRUST_LOG)
    latest = df.groupby("node").tail(1).reset_index(drop=True)
    latest["trust_tier"] = latest["final_score"].apply(classify_tier)
    latest.to_csv(OUTPUT_FILE, index=False)
    print("="*60 + "\nTRUST SCORE SUMMARY\n" + "="*60)
    print(f"Total Nodes   : {len(latest)}")
    print(f"High Trust    : {(latest['trust_tier']=='HIGH').sum()}")
    print(f"Medium Trust  : {(latest['trust_tier']=='MEDIUM').sum()}")
    print(f"Low Trust     : {(latest['trust_tier']=='LOW').sum()}")
    print(f"Average Score : {latest['final_score'].mean():.2f}")
    print(f"Max Score     : {latest['final_score'].max():.2f}")
    print(f"Min Score     : {latest['final_score'].min():.2f}")
    print(f"\nOutput saved to: {OUTPUT_FILE}\n" + "="*60)
    print("\nTop 10 Trusted Nodes:")
    print(latest.sort_values("final_score",ascending=False)
          [["node","final_score","trust_tier"]].head(10).to_string(index=False))

if __name__ == "__main__":
    run()
