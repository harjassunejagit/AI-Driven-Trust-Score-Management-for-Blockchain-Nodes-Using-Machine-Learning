import os
import pandas as pd
import networkx as nx

class GraphTrust:
    def __init__(self):
        self.graph        = nx.DiGraph()
        self.trust_scores = {}

    def load_transactions(self, csv_path):
        return pd.read_csv(csv_path)

    def build_graph(self, df, sender_col="from", receiver_col="to"):
        self.graph.clear()
        grouped = df.groupby([sender_col, receiver_col]).size().reset_index(name="frequency")
        for _, row in grouped.iterrows():
            self.graph.add_edge(row[sender_col], row[receiver_col], weight=row["frequency"])

    def set_feature_trust(self, trust_dictionary):
        self.trust_scores = trust_dictionary.copy()

    def incoming_neighbors(self, address):
        if address not in self.graph:
            return []
        return [{"address": n, "trust": self.trust_scores.get(n, 50),
                 "weight": self.graph[n][address]["weight"]}
                for n in self.graph.predecessors(address)]

    def graph_trust(self, address):
        neighbors = self.incoming_neighbors(address)
        if not neighbors:
            return self.trust_scores.get(address, 50)
        total = sum(n["weight"] for n in neighbors)
        score = sum(n["trust"] * n["weight"] for n in neighbors) / total
        return round(score, 2)

    def compute_all_graph_scores(self):
        return {node: self.graph_trust(node) for node in self.graph.nodes()}

    def node_statistics(self, address):
        if address not in self.graph:
            return None
        return {"incoming_degree": self.graph.in_degree(address),
                "outgoing_degree": self.graph.out_degree(address),
                "total_degree":    self.graph.degree(address)}

    def export_scores(self, output_file="graph_trust_scores.csv"):
        scores = self.compute_all_graph_scores()
        result = pd.DataFrame({"Address": list(scores.keys()),
                                "GraphTrust": list(scores.values())
                               }).sort_values("GraphTrust", ascending=False)
        result.to_csv(output_file, index=False)
        print("\n" + "="*60 + "\nGRAPH TRUST SCORES GENERATED")
        print(f"Total Nodes : {len(result)}  |  Output : {output_file}\n" + "="*60)


if __name__ == "__main__":
    gt = GraphTrust()
    transactions = gt.load_transactions("transactions.csv")
    gt.build_graph(transactions)
    print(f"Nodes : {gt.graph.number_of_nodes()}  Edges : {gt.graph.number_of_edges()}")

    unique = set(transactions["from"]).union(set(transactions["to"]))
    trust_dict = {a: 50 for a in unique}

    if os.path.exists("trust_log.csv"):
        try:
            log    = pd.read_csv("trust_log.csv")
            latest = log.groupby("node")["final_score"].last()
            for addr, score in latest.items():
                trust_dict[addr] = score
            print("Loaded scores from trust_log.csv")
        except Exception:
            print("Using default trust scores.")

    gt.set_feature_trust(trust_dict)
    gt.export_scores()

    scores = gt.compute_all_graph_scores()
    top10  = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 Nodes:")
    for addr, score in top10:
        s = gt.node_statistics(addr)
        print(f"  {addr[:16]}...  Trust={score}  In={s['incoming_degree']}  Out={s['outgoing_degree']}")
