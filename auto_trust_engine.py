import os, sys, json, time, joblib
import pandas as pd
from datetime import datetime, timezone
from web3 import Web3
from graph_trust import GraphTrust
import trust_fusion
import warnings
warnings.filterwarnings("ignore")

# Windows consoles default stdout to cp1252, which can't encode the
# checkmark characters this script prints — force UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RPC_URL          = os.getenv("GANACHE_RPC",       "http://127.0.0.1:7545")
CONTRACT_ADDRESS = os.getenv("CONTRACT_ADDRESS",  "0x076Fc718CbB6A7404d9eD94bb203e8d13Bc1cAF9")
ABI_PATH         = "TrustScore_abi.json"
DATASET_PATH     = "data/ethereum_fraud.csv"
TRANSACTION_PATH = "transactions.csv"
LOG_FILE         = "trust_log.csv"
MAX_ITERATIONS   = int(os.getenv("MAX_ITERATIONS", "3"))
SLEEP_TIME       = 5

print("="*70 + "\nLoading Machine Learning Models\n" + "="*70)
rf_scaler           = joblib.load("rf_scaler.pkl")
rf_model            = joblib.load("fraud_model.pkl")
xgb_model           = joblib.load("xgboost_model.pkl")
logistic_model      = joblib.load("logistic_model.pkl")
svm_model           = joblib.load("svm_model.pkl")
decision_tree_model = joblib.load("decision_tree_model.pkl")
feature_columns     = joblib.load("feature_cols.pkl")
if_model            = joblib.load("isolation_forest.pkl")
if_scaler           = joblib.load("if_scaler.pkl")
print("✓ Random Forest  ✓ XGBoost  ✓ Logistic  ✓ SVM  ✓ Decision Tree  ✓ Isolation Forest")

print("\nLoading Dataset...")
df = pd.read_csv(DATASET_PATH)
df.columns = df.columns.str.strip()
print(f"Dataset Size before cleaning : {len(df)}")

# Remove invalid addresses — Ethereum addresses are exactly 42 characters
# Anything longer is a transaction hash, not an address
df = df[df["Address"].str.len() == 42].reset_index(drop=True)
df = df[df["Address"].str.startswith("0x")].reset_index(drop=True)
df = df.dropna(subset=["Address"]).reset_index(drop=True)

print(f"Dataset Size after cleaning  : {len(df)}")
print("\nConnecting to Ganache...")
w3 = Web3(Web3.HTTPProvider(RPC_URL))
if not w3.is_connected():
    raise Exception("Cannot connect to Ganache at " + RPC_URL)
print("✓ Connected")
with open(ABI_PATH) as f: abi = json.load(f)
contract = w3.eth.contract(address=Web3.to_checksum_address(CONTRACT_ADDRESS), abi=abi)
accounts = w3.eth.accounts
if not accounts: raise Exception("No Ganache accounts found.")
sender = accounts[0]
print(f"Blockchain Account : {sender}")

graph_model = GraphTrust()
if os.path.exists(TRANSACTION_PATH):
    graph_model.build_graph(pd.read_csv(TRANSACTION_PATH))
    print("✓ Transaction Graph Loaded")
else:
    print("Warning: transactions.csv not found — run generate_transactions.py first")

# FIX: top-level function, not nested
def initialize_log():
    if os.path.exists(LOG_FILE): return
    pd.DataFrame(columns=[
        "timestamp","node","old_score","rf_score","xgb_score","logistic_score",
        "svm_score","decision_tree_score","ml_score","graph_score","final_score",
        "fraud_probability","tx_hash","block_number"
    ]).to_csv(LOG_FILE, index=False)

def prob_to_trust(p):
    return (100 - p * 100).clip(0, 100).round(2)

def prepare_all_features(dataframe):
    features = dataframe.reindex(columns=feature_columns, fill_value=0).fillna(0)
    scaled = rf_scaler.transform(features)
    return features, scaled

def predict_trust_batch(dataframe):
    # Vectorized over the whole dataset at once — calling predict_proba
    # row-by-row on n_jobs=-1 models (RF/XGB) pays a large thread-pool
    # spin-up cost per call; batching it turns tens of minutes into seconds
    # with numerically identical results.
    features, scaled = prepare_all_features(dataframe)
    rf_p  = rf_model.predict_proba(features)[:, 1]
    xgb_p = xgb_model.predict_proba(features)[:, 1]
    lr_p  = logistic_model.predict_proba(scaled)[:, 1]
    sv_p  = svm_model.predict_proba(scaled)[:, 1]
    dt_p  = decision_tree_model.predict_proba(features)[:, 1]

    rf_s  = prob_to_trust(rf_p);  xgb_s = prob_to_trust(xgb_p)
    lr_s  = prob_to_trust(lr_p);  sv_s  = prob_to_trust(sv_p)
    dt_s  = prob_to_trust(dt_p)

    ml_score = (rf_s*0.30 + xgb_s*0.30 + lr_s*0.15 + sv_s*0.15 + dt_s*0.10).round(2)
    fraud_p  = (rf_p*0.30 + xgb_p*0.30 + lr_p*0.15 + sv_p*0.15 + dt_p*0.10).round(6)

    # Algorithm 4, step 2: Isolation Forest anomaly penalty applied to the
    # feature-based trust score T_i^(F) before graph fusion.
    anomaly_label = if_model.predict(if_scaler.transform(features))
    ml_score = pd.Series(ml_score).where(anomaly_label != -1,
                                          (ml_score - trust_fusion.ANOMALY_PENALTY).clip(0, None)).values

    return pd.DataFrame({
        "Address": dataframe["Address"].values,
        "fraud_probability": fraud_p, "rf_score": rf_s, "xgb_score": xgb_s,
        "logistic_score": lr_s, "svm_score": sv_s, "decision_tree_score": dt_s,
        "ml_score": ml_score, "anomaly_label": anomaly_label,
    })

def calculate_final_trust(address, ml_score):
    try:    graph_score = graph_model.graph_trust(address)
    except: graph_score = ml_score
    # FIX: trust_fusion.integrate() not fusion.compute_trust()
    return round(graph_score, 2), trust_fusion.integrate(ml_score, graph_score)

def read_onchain_score(address):
    cs = Web3.to_checksum_address(address)
    try:    return contract.functions.getTrust(cs).call()
    except:
        try: return contract.functions.trustScores(cs).call()
        except: return None

def update_onchain(address, score):
    cs = Web3.to_checksum_address(address)
    try:
        tx = contract.functions.updateTrust(cs, int(round(score))).transact({"from": sender})
        r  = w3.eth.wait_for_transaction_receipt(tx)
        return tx.hex(), r.blockNumber
    except Exception as e:
        print("Blockchain Error:", e); return None, None

def run():
    initialize_log()
    for iteration in range(MAX_ITERATIONS):
        print(f"\n{'='*70}\nIteration {iteration+1}/{MAX_ITERATIONS}\n{'='*70}")
        print("\nGenerating Trust Scores (vectorized)...")
        scored = predict_trust_batch(df)
        # 25 addresses in the dataset are exact duplicates, so "Address"
        # isn't a unique key — iterate records positionally instead of
        # building an address-keyed dict (which would raise on duplicates).
        records = scored.to_dict("records")
        trust_dict = dict(zip(scored["Address"], scored["ml_score"]))
        anomaly_count = int((scored["anomaly_label"] == -1).sum())
        graph_model.set_feature_trust(trust_dict)
        print(f"Isolation Forest flagged {anomaly_count}/{len(df)} addresses as anomalous")
        print("Graph Trust Updated\nSynchronizing Blockchain...\n")
        for i, row in enumerate(records):
            if i % 250 == 0: print(f"  Updating : {i}/{len(df)}")
            addr = row["Address"]
            graph_s, final_s = calculate_final_trust(addr, row["ml_score"])
            old = read_onchain_score(addr)
            tx, block = update_onchain(addr, final_s)
            ts = datetime.now(timezone.utc).isoformat()
            pd.DataFrame([{"timestamp":ts,"node":addr,"old_score":old,
                "rf_score":row["rf_score"],"xgb_score":row["xgb_score"],
                "logistic_score":row["logistic_score"],"svm_score":row["svm_score"],
                "decision_tree_score":row["decision_tree_score"],"ml_score":row["ml_score"],
                "graph_score":graph_s,"final_score":final_s,
                "fraud_probability":row["fraud_probability"],"tx_hash":tx,"block_number":block
            }]).to_csv(LOG_FILE, mode="a", header=False, index=False)
        print(f"\n  Total : {len(df)}  |  Log : {LOG_FILE}")
        if iteration < MAX_ITERATIONS - 1:
            print(f"\nSleeping {SLEEP_TIME}s..."); time.sleep(SLEEP_TIME)
    print("\n✓ AI TRUST ENGINE COMPLETED  ✓ Blockchain Updated")

if __name__ == "__main__":
    run()
