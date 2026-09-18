import os, sys, json
import pandas as pd
from web3 import Web3

# Windows consoles default stdout to cp1252, which can't encode the
# checkmark/box-drawing characters this script prints — force UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RPC_URL          = os.getenv("GANACHE_RPC",      "http://127.0.0.1:7545")
CONTRACT_ADDRESS = os.getenv("CONTRACT_ADDRESS", "0x076Fc718CbB6A7404d9eD94bb203e8d13Bc1cAF9")
ABI_PATH         = os.getenv("ABI_PATH",         "TrustScore_abi.json")

w3 = Web3(Web3.HTTPProvider(RPC_URL))
if not w3.is_connected(): raise SystemExit("Cannot connect to Ganache at " + RPC_URL)
print("✓ Connected to Ganache\n")

with open(ABI_PATH) as f: abi = json.load(f)
contract = w3.eth.contract(address=Web3.to_checksum_address(CONTRACT_ADDRESS), abi=abi)

TRUST_LOG = "trust_log.csv"

# auto_trust_engine.py only ever calls updateTrust() for the real
# dataset addresses in trust_log.csv, never for w3.eth.accounts
# (Ganache's local test accounts) — read scores for the addresses
# that were actually written on-chain.
if os.path.exists(TRUST_LOG):
    nodes = pd.read_csv(TRUST_LOG)["node"].dropna().unique().tolist()
else:
    nodes = w3.eth.accounts
    print(f"Warning: {TRUST_LOG} not found — falling back to Ganache "
          f"accounts, which auto_trust_engine.py never scores.\n")

SHOWN = 20
print(f"─── Trust Scores ({min(SHOWN, len(nodes))} of {len(nodes)} nodes) ───\n")
for node in nodes[:SHOWN]:
    try:
        score = contract.functions.getTrust(Web3.to_checksum_address(node)).call()  # FIX
        status = "HIGH   ✓" if score>=70 else ("MEDIUM !" if score>=40 else "LOW    ✗")
        print(f"Node {node}  Trust = {score:>3}  [{status}]")
    except Exception as e:
        print(f"Node {node}  Error: {e}")
