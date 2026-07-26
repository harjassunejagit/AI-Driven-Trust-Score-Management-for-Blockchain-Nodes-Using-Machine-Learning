import pandas as pd

LAMBDA          = 0.7
ANOMALY_PENALTY = 15

def apply_anomaly_penalty(feature_trust, anomaly):
    return max(0, feature_trust - ANOMALY_PENALTY) if anomaly == -1 else feature_trust

def graph_trust_score(neighbours):
    if not neighbours:
        return 0
    total = sum(n["weight"] for n in neighbours)
    if total == 0:
        return 0
    return sum(n["trust"] * n["weight"] for n in neighbours) / total

def integrate(feature_trust, graph_trust):
    return round(max(0, min(100, LAMBDA*feature_trust + (1-LAMBDA)*graph_trust)), 2)

def compute_final_score(fraud_probability, anomaly_label, neighbours):
    ft = (1 - fraud_probability) * 100
    ft = apply_anomaly_penalty(ft, anomaly_label)
    gs = graph_trust_score(neighbours)
    return {"fraud_probability": fraud_probability, "feature_trust": round(ft,2),
            "graph_trust": round(gs,2), "final_trust": integrate(ft, gs)}

if __name__ == "__main__":
    result = compute_final_score(0.18, 1,
        [{"trust":92,"weight":6},{"trust":75,"weight":2},{"trust":60,"weight":4}])
    print(pd.DataFrame([result]))
