
import sys
import joblib, shap, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Windows consoles default stdout to cp1252, which can't encode the
# checkmark characters this script prints — force UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

model        = joblib.load("fraud_model.pkl")
scaler       = joblib.load("rf_scaler.pkl")
feature_cols = joblib.load("feature_cols.pkl")
explainer    = shap.Explainer(model)

def preprocess(data):
    if isinstance(data, (dict, pd.Series)): df = pd.DataFrame([data])
    else: df = data.copy()
    df.columns = df.columns.str.strip()
    for col in feature_cols:
        if col not in df.columns: df[col] = 0
    return pd.DataFrame(scaler.transform(df[feature_cols].fillna(0)), columns=feature_cols)

def predict(data):
    p = model.predict_proba(preprocess(data))[0][1]
    return p, max(0, min(100, int((1-p)*100)))

def fraud_class(exp):
    # This SHAP version returns (samples, features, classes) for the RF
    # binary classifier — keep only the fraud (class 1) slice so downstream
    # code sees the (samples, features) shape it expects.
    return exp[:, :, 1] if exp.values.ndim == 3 else exp

def explain(data):
    X   = preprocess(data)
    exp = fraud_class(explainer(X))
    imp = pd.DataFrame({"Feature":feature_cols,"SHAP_Value":exp.values[0]})
    imp["Importance"] = imp["SHAP_Value"].abs()
    imp = imp.sort_values("Importance", ascending=False)
    p, t = predict(data)
    return {"fraud_probability":float(p),"trust_score":int(t),"feature_importance":imp.reset_index(drop=True)}

def save_summary_plot(dataset_path="data/ethereum_fraud.csv"):
    df  = pd.read_csv(dataset_path); df.columns = df.columns.str.strip()
    X   = pd.DataFrame(scaler.transform(df[feature_cols].fillna(0)), columns=feature_cols)
    exp = fraud_class(explainer(X))
    plt.figure(figsize=(12,8)); shap.plots.beeswarm(exp, show=False)
    plt.tight_layout(); plt.savefig("shap_summary.png",dpi=300,bbox_inches="tight"); plt.close()
    print("✓ shap_summary.png")

def save_waterfall_plot(data):
    X = preprocess(data); exp = fraud_class(explainer(X))
    plt.figure(figsize=(10,8)); shap.plots.waterfall(exp[0], show=False)
    plt.tight_layout(); plt.savefig("waterfall_plot.png",dpi=300,bbox_inches="tight"); plt.close()
    print("✓ waterfall_plot.png")

if __name__ == "__main__":
    df = pd.read_csv("data/ethereum_fraud.csv"); sample = df.iloc[0]
    r  = explain(sample)
    print(f"\nFraud Probability : {r['fraud_probability']}")
    print(f"Trust Score       : {r['trust_score']}")
    print("\nTop 10 Features:\n", r["feature_importance"].head(10))
    save_summary_plot(); save_waterfall_plot(sample)
    print("\nSHAP analysis completed.")
