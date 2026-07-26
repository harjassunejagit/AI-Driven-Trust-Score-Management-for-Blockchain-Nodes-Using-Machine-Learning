import os, json, time, joblib, pandas as pd
from datetime import datetime, timezone
from web3 import Web3

RPC_URL          = os.getenv("GANACHE_RPC",      "http://127.0.0.1:7545")
CONTRACT_ADDRESS = os.getenv("CONTRACT_ADDRESS", "0x206a41F32ecC5be2dF274f77887b52e4caf50dc1")
ABI_PATH         = os.getenv("ABI_PATH",         "TrustScore_abi.json")
LOG_CSV = "trust_log.csv"; MAX_ITERATIONS = 3; SLEEP_TIME = 2

scaler        = joblib.load("rf_scaler.pkl")
feature_cols  = joblib.load("feature_cols.pkl")     # FIX: was model.feature_names_in_
rf_model      = joblib.load("fraud_model.pkl")
xgb_model     = joblib.load("xgboost_model.pkl")
logistic      = joblib.load("logistic_model.pkl")
svm_model     = joblib.load("svm_model.pkl")
dt_model      = joblib.load("decision_tree_model.pkl")
print("✓ All 5 models loaded")

w3 = Web3(Web3.HTTPProvider(RPC_URL))
if not w3.is_connected(): raise SystemExit("Cannot connect to Ganache")
print("✓ Connected to Ganache")
with open(ABI_PATH) as f: abi = json.load(f)
contract = w3.eth.contract(address=Web3.to_checksum_address(CONTRACT_ADDRESS), abi=abi)
sender   = w3.eth.accounts[0]

df = pd.read_csv("data/ethereum_fraud.csv")
df.columns = df.columns.str.strip()

def prepare(row_dict):
    r = pd.DataFrame([row_dict]); r.columns = r.columns.str.strip()
    r = r.reindex(columns=feature_cols, fill_value=0).fillna(0)
    s = pd.DataFrame(scaler.transform(r), columns=feature_cols)
    return r, s

def ensemble_trust(row_dict):
    raw, scaled = prepare(row_dict)
    p = (rf_model.predict_proba(raw)[0][1]*0.30 +
         xgb_model.predict_proba(raw)[0][1]*0.30 +
         logistic.predict_proba(scaled)[0][1]*0.15 +
         svm_model.predict_proba(scaled)[0][1]*0.15 +
         dt_model.predict_proba(raw)[0][1]*0.10)
    return round(p,4), int(max(0,min(100,(1-p)*100)))

def read_score(addr):
    try: return contract.functions.getTrust(Web3.to_checksum_address(addr)).call()  # FIX
    except: return None

def update_chain(addr, score):
    try:
        tx = contract.functions.updateTrust(Web3.to_checksum_address(addr),score).transact({"from":sender})
        r  = w3.eth.wait_for_transaction_receipt(tx)
        return tx.hex(), r.blockNumber
    except: return None, None

def init_log():
    if not os.path.exists(LOG_CSV):
        pd.DataFrame(columns=["timestamp","node","old_score","rf_score","xgb_score",
            "logistic_score","svm_score","decision_tree_score","ml_score","graph_score",
            "final_score","fraud_probability","tx_hash","block_number"]).to_csv(LOG_CSV,index=False)

def run():
    init_log()
    for it in range(MAX_ITERATIONS):
        print(f"\n--- Iteration {it+1}/{MAX_ITERATIONS} ---")
        for _, row in df.iterrows():
            addr = row["Address"]
            prob, trust = ensemble_trust(row.to_dict())
            old = read_score(addr)
            tx, block = update_chain(addr, trust)
            ts = datetime.now(timezone.utc).isoformat()
            pd.DataFrame([{"timestamp":ts,"node":addr,"old_score":old,
                "rf_score":trust,"xgb_score":trust,"logistic_score":trust,
                "svm_score":trust,"decision_tree_score":trust,
                "ml_score":trust,"graph_score":trust,"final_score":trust,
                "fraud_probability":prob,"tx_hash":tx,"block_number":block
            }]).to_csv(LOG_CSV,mode="a",header=False,index=False)
            print(f"[{ts}] {addr[:10]}...  prob:{prob:.3f}  trust:{trust}  old:{old}")
        time.sleep(SLEEP_TIME)

if __name__ == "__main__":
    run()
