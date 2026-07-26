import os

def step(n, desc, script):
    print(f"\n{'='*70}\nStep {n} : {desc}\n{'='*70}")
    code = os.system(f"python {script}")
    if code != 0: print(f"\n⚠  Step {n} exited with code {code}")

print("\n" + "="*70 + "\nABNG : AI-Driven Blockchain Node Guard\n" + "="*70)
step(1, "Training All 5 ML Models",             "train_model.py")
step(2, "Training Isolation Forest",            "anomaly_detector.py")
step(3, "Evaluating Models + Generating Charts","evaluate_model.py")
step(4, "Generating Transaction Network",       "generate_transactions.py")
step(5, "Computing Graph Trust Scores",         "graph_trust.py")
step(6, "Running Trust Engine + Blockchain Sync","auto_trust_engine.py")
step(7, "Generating Trust Visualizations",      "visualize_trust_scores.py")
print("\n" + "="*70 + "\nABNG PROJECT EXECUTED SUCCESSFULLY\n" + "="*70)
print("\nOutput Files: fraud_model.pkl  xgboost_model.pkl  logistic_model.pkl")
print("              svm_model.pkl  decision_tree_model.pkl  rf_scaler.pkl")
print("              isolation_forest.pkl  transactions.csv  trust_log.csv")
print("              results/  (all charts, CSVs, SHAP plots)")
