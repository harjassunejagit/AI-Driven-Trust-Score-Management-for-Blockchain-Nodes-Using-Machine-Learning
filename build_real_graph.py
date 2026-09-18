# build_real_graph.py
# Builds a behavioral similarity graph from your existing dataset
# Uses exact column names from your ethereum_fraud.csv

import sys
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Windows consoles decode stdout as cp1252 by default, which garbles (though
# doesn't crash on) the em-dash this script prints — force UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Load dataset ──────────────────────────────────────────────
df = pd.read_csv("data/ethereum_fraud.csv")
df.columns = df.columns.str.strip()
print(f"Total addresses: {len(df)}")

# ── Graph features — exact names from your dataset ────────────
graph_features = [
    "Unique Received From Addresses",
    "Unique Sent To Addresses",
    "Sent tnx",
    "Received Tnx",
    "ERC20 uniq sent addr",
    "ERC20 uniq rec addr",
    "total Ether sent",
    "total ether received",
    "Avg min between sent tnx",
    "Avg min between received tnx",
]

# Keep only columns that exist in your dataset
available = [c for c in graph_features if c in df.columns]
missing   = [c for c in graph_features if c not in df.columns]
if missing:
    print(f"Skipping missing columns: {missing}")
print(f"Using {len(available)} features: {available}")

X         = df[available].fillna(0)
addresses = df["Address"].tolist()

# ── Normalize ─────────────────────────────────────────────────
scaler   = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ── Build similarity graph ────────────────────────────────────
SIMILARITY_THRESHOLD = 0.92
TOP_K                = 5
BATCH_SIZE           = 500

G = nx.DiGraph()
for addr in addresses:
    G.add_node(addr)

print(f"\nBuilding graph (threshold={SIMILARITY_THRESHOLD}, top-K={TOP_K})...")

n           = len(X_scaled)
total_edges = 0

for i in range(0, n, BATCH_SIZE):
    batch       = X_scaled[i:i+BATCH_SIZE]
    batch_addrs = addresses[i:i+BATCH_SIZE]
    sims        = cosine_similarity(batch, X_scaled)

    for local_idx, (addr_i, sim_row) in enumerate(zip(batch_addrs, sims)):
        global_idx  = i + local_idx
        top_indices = np.argsort(sim_row)[::-1]
        added = 0
        for j in top_indices:
            if j == global_idx:
                continue
            if added >= TOP_K:
                break
            if sim_row[j] >= SIMILARITY_THRESHOLD:
                addr_j = addresses[j]
                if not G.has_edge(addr_j, addr_i):
                    G.add_edge(addr_j, addr_i,
                               weight=float(round(sim_row[j], 4)))
                    total_edges += 1
                    added += 1

    if i % 1000 == 0:
        print(f"  {i}/{n} processed — edges: {total_edges}")

print(f"\nGraph complete:")
print(f"  Nodes        : {G.number_of_nodes()}")
print(f"  Edges        : {G.number_of_edges()}")
print(f"  Avg in-degree: {G.number_of_edges()/max(G.number_of_nodes(),1):.2f}")

# ── Save as transactions.csv ──────────────────────────────────
rows = [
    {
        "from":         src,
        "to":           dst,
        "weight":       data["weight"],
        "amount":       round(data["weight"] * 100, 2),
        "gas_used":     21000,
        "block_number": 1000000,
        "timestamp":    1600000000,
    }
    for src, dst, data in G.edges(data=True)
]

pd.DataFrame(rows).to_csv("transactions.csv", index=False)
print(f"\nSaved {len(rows)} edges to transactions.csv")
print("Done. Now run: python auto_trust_engine.py")