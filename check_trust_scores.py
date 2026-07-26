import os, json
from web3 import Web3

RPC_URL          = os.getenv("GANACHE_RPC",      "http://127.0.0.1:7545")
CONTRACT_ADDRESS = os.getenv("CONTRACT_ADDRESS", "0x206a41F32ecC5be2dF274f77887b52e4caf50dc1")
ABI_PATH         = os.getenv("ABI_PATH",         "TrustScore_abi.json")

w3 = Web3(Web3.HTTPProvider(RPC_URL))
if not w3.is_connected(): raise SystemExit("Cannot connect to Ganache at " + RPC_URL)
print("✓ Connected to Ganache\n")

with open(ABI_PATH) as f: abi = json.load(f)
contract = w3.eth.contract(address=Web3.to_checksum_address(CONTRACT_ADDRESS), abi=abi)

print("─── Trust Scores of All Nodes ───\n")
for node in w3.eth.accounts:
    try:
        score = contract.functions.getTrust(Web3.to_checksum_address(node)).call()  # FIX
        status = "HIGH   ✓" if score>=70 else ("MEDIUM !" if score>=40 else "LOW    ✗")
        print(f"Node {node}  Trust = {score:>3}  [{status}]")
    except Exception as e:
        print(f"Node {node}  Error: {e}")
