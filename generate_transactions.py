
import random, pandas as pd

random.seed(42)
df        = pd.read_csv("data/ethereum_fraud.csv")
df.columns = df.columns.str.strip()
addresses = df["Address"].dropna().astype(str).unique().tolist()
NUM       = max(len(addresses)*5, 10000)
txns      = []
for _ in range(NUM):
    s = random.choice(addresses); r = random.choice(addresses)
    while r == s: r = random.choice(addresses)
    txns.append({"from":s,"to":r,"amount":round(random.uniform(0.1,250),4),
                 "gas_used":random.randint(21000,120000),
                 "block_number":random.randint(1000000,5000000),
                 "timestamp":random.randint(1600000000,1700000000)})
txdf = pd.DataFrame(txns)
txdf.to_csv("transactions.csv", index=False)
print("="*60 + f"\ntransactions.csv generated\nAddresses:{len(addresses)}  Transactions:{len(txdf)}\n" + "="*60)
print(txdf.head())
