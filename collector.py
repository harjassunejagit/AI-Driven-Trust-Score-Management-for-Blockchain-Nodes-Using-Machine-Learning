from web3 import Web3
import time, csv
web3 = Web3(Web3.HTTPProvider("http://127.0.0.1:7545"))
if not web3.is_connected(): print("Error: Not connected"); exit()
nodes = web3.eth.accounts
with open("node_activity.csv","w",newline="") as f:
    csv.writer(f).writerow(["timestamp","node","tx_count","gas_used","block_number"])
print("Collecting... CTRL+C to stop.")
try:
    while True:
        for node in nodes:
            b = web3.eth.get_block("latest")
            with open("node_activity.csv","a",newline="") as f:
                csv.writer(f).writerow([time.time(),node,web3.eth.get_transaction_count(node),b.gasUsed,b.number])
        time.sleep(2)
except KeyboardInterrupt:
    print("\nStopped.")
