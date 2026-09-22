import os, sys, joblib, shap, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Windows consoles default stdout to cp1252, which can't encode the
# checkmark character this script prints — force UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
warnings.filterwarnings("ignore")

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

print("Loading Dataset...")
df = pd.read_csv("data/ethereum_fraud.csv")
df.columns = df.columns.str.strip()
drop_cols = ["FLAG","Unnamed: 0","Index","Address",
             "ERC20 most sent token type","ERC20_most_rec_token_type"]
feature_cols = [c for c in df.columns if c not in drop_cols]  # FIX: keep before transform
X_raw = df[feature_cols].fillna(0)
y     = df["FLAG"]

print("Loading Scaler...")
scaler = joblib.load("rf_scaler.pkl")
# FIX: split first, then scale — no data leakage
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.25, random_state=42, stratify=y)
X_train_scaled = scaler.transform(X_train_raw)
X_test_scaled  = scaler.transform(X_test_raw)

print("Loading Trained Models...")
logistic      = joblib.load("logistic_model.pkl")
svm           = joblib.load("svm_model.pkl")
decision_tree = joblib.load("decision_tree_model.pkl")
random_forest = joblib.load("fraud_model.pkl")
xgboost       = joblib.load("xgboost_model.pkl")

models = {
    "Logistic Regression": (logistic,      X_test_scaled),
    "SVM":                 (svm,           X_test_scaled),
    "Decision Tree":       (decision_tree, X_test_raw),
    "Random Forest":       (random_forest, X_test_raw),
    "XGBoost":             (xgboost,       X_test_raw),
}

comparison = []
roc_info   = {}

print("\n" + "="*60 + "\nMODEL EVALUATION STARTED\n" + "="*60)

for model_name, (model, X_eval) in models.items():
    print(f"\n{'='*60}\n{model_name}\n{'='*60}")
    pred = model.predict(X_eval)
    prob = model.predict_proba(X_eval)[:, 1]
    acc  = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, zero_division=0)
    rec  = recall_score(y_test, pred, zero_division=0)
    f1   = f1_score(y_test, pred, zero_division=0)
    roc  = roc_auc_score(y_test, prob)
    comparison.append({"Model":model_name,"Accuracy":acc,"Precision":prec,
                        "Recall":rec,"F1 Score":f1,"ROC AUC":roc})
    print(f"Accuracy:{acc:.4f}  Precision:{prec:.4f}  Recall:{rec:.4f}  F1:{f1:.4f}  AUC:{roc:.4f}")
    cm = confusion_matrix(y_test, pred)
    print("\nConfusion Matrix\n", cm)
    print("\nClassification Report\n", classification_report(y_test, pred))
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Benign","Fraud"]); ax.set_yticklabels(["Benign","Fraud"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"{model_name} Confusion Matrix")
    thresh = cm.max()/2.0
    for i in range(2):
        for j in range(2):
            ax.text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=12,
                    color="white" if cm[i,j]>thresh else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, model_name.replace(" ","_")+"_confusion_matrix.png"),dpi=300)
    plt.close()
    fpr,tpr,_ = roc_curve(y_test, prob)
    roc_info[model_name] = (fpr,tpr,roc)

# ROC
plt.figure(figsize=(8,6))
for name,(fpr,tpr,roc) in roc_info.items():
    plt.plot(fpr,tpr,linewidth=2,label=f"{name} (AUC={roc:.3f})")
plt.plot([0,1],[0,1],"k--",label="Random Chance")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison"); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR,"roc_comparison.png"),dpi=300); plt.close()

comp_df = pd.DataFrame(comparison).sort_values("Accuracy",ascending=False)
comp_df.to_csv(os.path.join(RESULT_DIR,"model_comparison.csv"),index=False)
print("\n",comp_df)

plt.figure(figsize=(9,5))
bars = plt.bar(comp_df["Model"],comp_df["Accuracy"],
               color=["#4C72B0","#55A868","#C44E52","#8172B2","#CCB974"])
for bar,val in zip(bars,comp_df["Accuracy"]):
    plt.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.003,
             f"{val:.4f}",ha="center",va="bottom",fontsize=9)
plt.ylabel("Accuracy"); plt.title("Accuracy Comparison")
plt.xticks(rotation=20,ha="right"); plt.ylim(0.7,1.0); plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR,"accuracy_comparison.png"),dpi=300); plt.close()

x=np.arange(len(comp_df)); w=0.25
fig,ax=plt.subplots(figsize=(10,6))
ax.bar(x-w,comp_df["Precision"],w,label="Precision",color="#4C72B0")
ax.bar(x,  comp_df["Recall"],   w,label="Recall",   color="#55A868")
ax.bar(x+w,comp_df["F1 Score"], w,label="F1 Score", color="#C44E52")
ax.set_xticks(x); ax.set_xticklabels(comp_df["Model"],rotation=20,ha="right")
ax.set_ylabel("Score"); ax.set_title("Precision / Recall / F1"); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(RESULT_DIR,"precision_recall_f1.png"),dpi=300); plt.close()

# Threshold
prob_rf = random_forest.predict_proba(X_test_raw)[:,1]
rows=[]
for t in np.arange(0.30,0.81,0.05):
    pred=(prob_rf>=t).astype(int)
    rows.append({"Threshold":round(t,2),"Accuracy":accuracy_score(y_test,pred),
                 "Precision":precision_score(y_test,pred,zero_division=0),
                 "Recall":recall_score(y_test,pred,zero_division=0),
                 "F1 Score":f1_score(y_test,pred,zero_division=0)})
tdf=pd.DataFrame(rows)
tdf.to_csv(os.path.join(RESULT_DIR,"threshold_analysis.csv"),index=False)
plt.figure(figsize=(8,5))
for m in ["Accuracy","Precision","Recall","F1 Score"]:
    plt.plot(tdf["Threshold"],tdf[m],marker="o",label=m)
plt.xlabel("Threshold"); plt.ylabel("Metric"); plt.title("Threshold Analysis (RF)")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR,"threshold_analysis.png"),dpi=300); plt.close()

# Feature importance — XGBoost, the highest-performing ensemble component
# (Table 5/6/6b/6c all favor it over RF; McNemar's test confirms the edge
# is real), rather than Random Forest.
imp=pd.DataFrame({"Feature":feature_cols,"Importance":xgboost.feature_importances_})
imp=imp.sort_values("Importance",ascending=False)
imp.to_csv(os.path.join(RESULT_DIR,"feature_importance.csv"),index=False)
plt.figure(figsize=(10,8)); plt.barh(imp.head(20)["Feature"],imp.head(20)["Importance"],color="#4C72B0")
plt.gca().invert_yaxis(); plt.xlabel("Importance"); plt.title("Top-20 Feature Importances (XGBoost)")
plt.tight_layout(); plt.savefig(os.path.join(RESULT_DIR,"feature_importance.png"),dpi=300); plt.close()

# SHAP — FIX: robust ndim handling. Uses XGBoost for the same reason as
# feature importance above.
sample=X_test_raw.sample(min(500,len(X_test_raw)),random_state=42)
explainer=shap.TreeExplainer(xgboost)
sv=explainer.shap_values(sample)
if isinstance(sv,list): sv=sv[1]
elif sv.ndim==3: sv=sv[:,:,1]
plt.figure(); shap.summary_plot(sv,sample,feature_names=feature_cols,show=False)
plt.tight_layout(); plt.savefig(os.path.join(RESULT_DIR,"shap_summary.png"),dpi=300); plt.close()

# Save raw SHAP values + feature values for directional analysis (which
# features push fraud probability up vs down) used when writing up results.
np.save(os.path.join(RESULT_DIR,"shap_values_xgb.npy"), sv)
sample.to_csv(os.path.join(RESULT_DIR,"shap_sample_xgb.csv"), index=False)

print("\n"+"="*60+"\nEvaluation Completed Successfully\n"+"="*60)
print("\nGenerated Files:")
for f in sorted(os.listdir(RESULT_DIR)): print(f"  ✓ {f}")
