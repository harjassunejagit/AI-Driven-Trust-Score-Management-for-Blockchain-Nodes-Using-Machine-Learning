# gnn_baseline.py
# A GNN baseline (2-layer GCN and mean-aggregator GraphSAGE) for fraud
# node-classification, evaluated on the SAME train/test split, feature
# set, and metrics as every other model in this project, over the real
# behavioral-similarity graph (transactions.csv / build_real_graph.py).
#
# Implemented directly in PyTorch (sparse adjacency, manual propagation)
# rather than torch_geometric, to avoid its compiled-extension install
# risk on Windows — these are still standard GCN/GraphSAGE formulations.

import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, average_precision_score)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

torch.manual_seed(42)
np.random.seed(42)

DATASET_PATH = "data/ethereum_fraud.csv"
TRANSACTION_PATH = "transactions.csv"

# ── Load features + labels (same feature set as train_model.py) ────────
df = pd.read_csv(DATASET_PATH)
df.columns = df.columns.str.strip()
drop_cols = ["FLAG", "Unnamed: 0", "Index", "Address",
             "ERC20 most sent token type", "ERC20_most_rec_token_type"]
feature_cols = [c for c in df.columns if c not in drop_cols]
addresses = df["Address"].tolist()
addr_to_idx = {a: i for i, a in enumerate(addresses)}
X_raw = df[feature_cols].fillna(0).values
y = df["FLAG"].values
N = len(df)

# ── Same 75/25 split as everywhere else in the project ─────────────────
idx_all = np.arange(N)
idx_train, idx_test = train_test_split(idx_all, test_size=0.25, random_state=42, stratify=y)
train_mask = np.zeros(N, dtype=bool); train_mask[idx_train] = True
test_mask  = np.zeros(N, dtype=bool); test_mask[idx_test] = True

scaler = MinMaxScaler().fit(X_raw[train_mask])  # fit on train only, like everywhere else
X = scaler.transform(X_raw)

# ── Build the real behavioral-similarity graph as a sparse adjacency ───
print("Loading real graph (transactions.csv)...")
edges_df = pd.read_csv(TRANSACTION_PATH)
rows, cols, vals = [], [], []
dropped = 0
for s, t, w in zip(edges_df["from"], edges_df["to"], edges_df.get("weight", pd.Series([1.0]*len(edges_df)))):
    if s in addr_to_idx and t in addr_to_idx:
        i, j = addr_to_idx[s], addr_to_idx[t]
        rows.append(i); cols.append(j); vals.append(float(w))
        rows.append(j); cols.append(i); vals.append(float(w))  # symmetrize for undirected propagation
    else:
        dropped += 1
print(f"Graph edges used: {len(rows)//2}  (dropped {dropped} edges referencing addresses outside this split)")

# self-loops
for i in range(N):
    rows.append(i); cols.append(i); vals.append(1.0)

indices = torch.tensor([rows, cols], dtype=torch.long)
values = torch.tensor(vals, dtype=torch.float32)
A = torch.sparse_coo_tensor(indices, values, (N, N)).coalesce()

deg = torch.sparse.sum(A, dim=1).to_dense()
deg_inv_sqrt = deg.pow(-0.5)
deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
deg_inv = deg.pow(-1.0)
deg_inv[torch.isinf(deg_inv)] = 0

# GCN-style symmetric normalization: D^-1/2 (A+I) D^-1/2
norm_vals_gcn = values * deg_inv_sqrt[indices[0]] * deg_inv_sqrt[indices[1]]
A_gcn = torch.sparse_coo_tensor(indices, norm_vals_gcn, (N, N)).coalesce()

# GraphSAGE-style row (mean) normalization: D^-1 (A+I)
norm_vals_sage = values * deg_inv[indices[0]]
A_sage = torch.sparse_coo_tensor(indices, norm_vals_sage, (N, N)).coalesce()

X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.long)
train_idx = torch.tensor(idx_train, dtype=torch.long)
test_idx  = torch.tensor(idx_test, dtype=torch.long)

class GCN(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, A_norm):
        super().__init__()
        self.A = A_norm
        self.lin1 = nn.Linear(in_dim, hidden)
        self.lin2 = nn.Linear(hidden, out_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        h = torch.sparse.mm(self.A, self.lin1(x))
        h = F.relu(h)
        h = self.dropout(h)
        h = torch.sparse.mm(self.A, self.lin2(h))
        return h

class GraphSAGE(nn.Module):
    # mean-aggregator GraphSAGE: concat(self, mean(neighbors)) -> linear
    def __init__(self, in_dim, hidden, out_dim, A_mean):
        super().__init__()
        self.A = A_mean
        self.lin1_self = nn.Linear(in_dim, hidden)
        self.lin1_neigh = nn.Linear(in_dim, hidden)
        self.lin2_self = nn.Linear(hidden, out_dim)
        self.lin2_neigh = nn.Linear(hidden, out_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        neigh = torch.sparse.mm(self.A, x)
        h = F.relu(self.lin1_self(x) + self.lin1_neigh(neigh))
        h = self.dropout(h)
        neigh2 = torch.sparse.mm(self.A, h)
        out = self.lin2_self(h) + self.lin2_neigh(neigh2)
        return out

def train_and_eval(model, name, epochs=1500, lr=0.01, weight_decay=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # class-weighted loss to handle the 77.9%/22.1% imbalance
    class_counts = np.bincount(y[idx_train])
    weights = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float32)
    weights = weights / weights.sum() * 2
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X_t)
        loss = F.cross_entropy(out[train_idx], y_t[train_idx], weight=weights)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 250 == 0:
            print(f"  [{name}] epoch {epoch+1}/{epochs}  loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        out = model(X_t)
        prob = F.softmax(out, dim=1)[:, 1].numpy()
        pred = (prob >= 0.5).astype(int)

    y_test = y[idx_test]
    pred_test = pred[idx_test]
    prob_test = prob[idx_test]
    return {
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, pred_test), 4),
        "Precision": round(precision_score(y_test, pred_test, zero_division=0), 4),
        "Recall": round(recall_score(y_test, pred_test, zero_division=0), 4),
        "F1": round(f1_score(y_test, pred_test, zero_division=0), 4),
        "ROC_AUC": round(roc_auc_score(y_test, prob_test), 4),
        "PR_AUC": round(average_precision_score(y_test, prob_test), 4),
    }

print("\nTraining GCN...")
gcn = GCN(len(feature_cols), 64, 2, A_gcn)
gcn_result = train_and_eval(gcn, "GCN (real behavioral-similarity graph)")

print("\nTraining GraphSAGE...")
sage = GraphSAGE(len(feature_cols), 64, 2, A_sage)
sage_result = train_and_eval(sage, "GraphSAGE (real behavioral-similarity graph)")

results = pd.DataFrame([gcn_result, sage_result])
results.to_csv("results/gnn_baseline.csv", index=False)
print("\n" + "="*70)
print(results.to_string(index=False))
print("="*70)
print("\nFor comparison (from Table 5/6, same test split):")
print("  Random Forest : Accuracy=0.9594  ROC_AUC=0.9884  PR_AUC=0.972")
print("  XGBoost       : Accuracy=0.9699  ROC_AUC=0.9926  PR_AUC=0.981")
print("\nSaved: results/gnn_baseline.csv")
