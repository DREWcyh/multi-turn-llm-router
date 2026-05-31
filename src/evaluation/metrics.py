import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
def safe_auc(y, prob):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, prob))
def probabilities(model, x):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        z = model.decision_function(x)
        return 1.0 / (1.0 + np.exp(-z))
    return model.predict(x).astype(float)
def classification_metrics(y, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    if len(np.unique(y)) < 2:
        pr = float("nan")
    else:
        pr = float(average_precision_score(y, prob))
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": safe_auc(y, prob),
        "pr_auc": pr,
    }
def route_metrics(y, prob, weak, strong, threshold=0.5, weak_cost=1.0, strong_cost=10.0):
    use_strong = (prob >= threshold).astype(int)
    q = np.where(use_strong == 1, strong, weak)
    cost = np.where(use_strong == 1, strong_cost, weak_cost)
    wq = float(np.mean(weak))
    sq = float(np.mean(strong))
    return {
        "threshold": float(threshold),
        "avg_quality": float(np.mean(q)),
        "avg_cost": float(np.mean(cost)),
        "strong_call_rate": float(np.mean(use_strong)),
        "cost_saving_vs_always_strong": float((strong_cost - np.mean(cost)) / strong_cost),
        "quality_gain_vs_always_weak": float(np.mean(q) - wq),
        "quality_gap_to_always_strong": float(sq - np.mean(q)),
    }
def combined_metrics(name, y, prob, weak, strong, threshold=0.5):
    ans = {"model": name}
    ans.update(classification_metrics(y, prob, threshold))
    ans.update(route_metrics(y, prob, weak, strong, threshold))
    return ans
def baseline_metrics(y, weak, strong):
    n = len(y)
    rows = []
    rows.append(combined_metrics("always_weak", y, np.zeros(n), weak, strong, 0.5))
    rows.append(combined_metrics("always_strong", y, np.ones(n), weak, strong, 0.5))
    oracle = ((strong - weak) >= 2).astype(float)
    rows.append(combined_metrics("oracle", y, oracle, weak, strong, 0.5))
    return rows
def threshold_metrics(split, model_name, y, prob, weak, strong, thresholds):
    rows = []
    for t in thresholds:
        r = {"split": split, "model": model_name}
        r.update(classification_metrics(y, prob, t))
        r.update(route_metrics(y, prob, weak, strong, t))
        rows.append(r)
    return rows
def pick_best_threshold(rows):
    return max(rows, key=lambda r: (r["f1"], r["avg_quality"], -r["avg_cost"]))
def pick_threshold(rows, policy="f1", max_quality_gap=0.30):
    if policy == "f1":
        return pick_best_threshold(rows)
    if policy == "quality_cost":
        ok = []
        for r in rows:
            if r["quality_gap_to_always_strong"] <= max_quality_gap:
                ok.append(r)
        if ok:
            return min(ok, key=lambda r: (r["avg_cost"], -r["avg_quality"], -r["f1"]))
        return min(rows, key=lambda r: (r["quality_gap_to_always_strong"], r["avg_cost"]))
    raise ValueError("unknown threshold policy: " + policy)
def add_normalized_quality(rows):
    if not rows:
        return []
    best_q = max(r["avg_quality"] + r["quality_gap_to_always_strong"] for r in rows)
    weak_q = rows[0]["avg_quality"] - rows[0]["quality_gain_vs_always_weak"]
    gap = max(best_q - weak_q, 1e-12)
    ans = []
    for r in rows:
        x = dict(r)
        x["normalized_quality"] = float((r["avg_quality"] - weak_q) / gap)
        ans.append(x)
    return ans
def cost_quality_auc(rows):
    rows = add_normalized_quality(rows)
    pts = [(0.0, 1.0), (0.9, 0.0)]
    for r in rows:
        pts.append((r["cost_saving_vs_always_strong"], r["normalized_quality"]))
    pts = sorted(pts)
    mp = {}
    for x, y in pts:
        mp[x] = max(y, mp.get(x, -1.0))
    xs = np.array(sorted(mp), dtype=float)
    ys = np.array([mp[x] for x in xs], dtype=float)
    return float(np.trapezoid(ys, xs))
