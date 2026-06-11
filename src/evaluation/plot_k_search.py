import argparse
import csv
import json
import os
from pathlib import Path

for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[name] = "8"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from evaluation.metrics import cost_quality_auc
from router_utils import ensure_dir, json_default


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def metric_stem(r, s):
    return f"recent_semantic_r{r}_s{s}_nall_xgboost_quality_cost_gap050"


def get_model_metric(data):
    for row in data["metrics"]:
        if row.get("model") == "xgboost" and row.get("split") == "test":
            return row
    raise ValueError("no xgboost test row")


def one_result(results_dir, r, s):
    stem = metric_stem(r, s)
    metric_file = results_dir / f"{stem}_metrics.json"
    threshold_file = results_dir / f"{stem}_thresholds.json"
    metric_data = load_json(metric_file)
    threshold_rows = load_json(threshold_file)
    valid_rows = [x for x in threshold_rows if x["split"] == "valid" and x["model"] == "xgboost"]
    test_rows = [x for x in threshold_rows if x["split"] == "test" and x["model"] == "xgboost"]
    test_metric = get_model_metric(metric_data)
    if not valid_rows or not test_rows:
        raise ValueError("missing threshold rows: " + stem)
    return {
        "recent_k": r,
        "semantic_k": s,
        "valid_roc_auc": valid_rows[0]["roc_auc"],
        "valid_pr_auc": valid_rows[0]["pr_auc"],
        "valid_cq_auc": cost_quality_auc(valid_rows),
        "test_roc_auc": test_metric["roc_auc"],
        "test_pr_auc": test_metric["pr_auc"],
        "test_cq_auc": cost_quality_auc(test_rows),
        "threshold": test_metric["threshold"],
        "f1": test_metric["f1"],
        "avg_quality": test_metric["avg_quality"],
        "cost_saving": test_metric["cost_saving_vs_always_strong"],
        "quality_gap": test_metric["quality_gap_to_always_strong"],
    }


def best_row(rows):
    best_cq = max(x["valid_cq_auc"] for x in rows)
    good = [x for x in rows if best_cq - x["valid_cq_auc"] <= 0.005]
    return max(good, key=lambda x: (x["valid_pr_auc"], -(x["recent_k"] + x["semantic_k"])))


def write_csv(path, rows):
    cols = [
        "recent_k",
        "semantic_k",
        "valid_roc_auc",
        "valid_pr_auc",
        "valid_cq_auc",
        "test_roc_auc",
        "test_pr_auc",
        "test_cq_auc",
        "threshold",
        "f1",
        "avg_quality",
        "cost_saving",
        "quality_gap",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow({c: row[c] for c in cols})


def plot_heatmap(rows, key, title, out_path):
    rs = sorted(set(x["recent_k"] for x in rows))
    ss = sorted(set(x["semantic_k"] for x in rows))
    mat = np.full((len(rs), len(ss)), np.nan)
    for row in rows:
        i = rs.index(row["recent_k"])
        j = ss.index(row["semantic_k"])
        mat[i, j] = row[key]
    plt.figure(figsize=(6.4, 5.2))
    im = plt.imshow(mat, origin="lower", cmap="viridis")
    plt.xticks(range(len(ss)), ss)
    plt.yticks(range(len(rs)), rs)
    plt.xlabel("semantic_k")
    plt.ylabel("recent_k")
    plt.title(title)
    plt.colorbar(im)
    for i in range(len(rs)):
        for j in range(len(ss)):
            if not np.isnan(mat[i, j]):
                plt.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", color="white", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".png"), dpi=300)
    plt.savefig(out_path.with_suffix(".pdf"))
    plt.close()


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="artifacts/results/07_find_recent_semantic_k")
    p.add_argument("--output-dir", default="artifacts/plots/07_find_recent_semantic_k")
    p.add_argument("--max-k", type=int, default=5)
    return p.parse_args()


def main():
    args = get_args()
    results_dir = Path(args.results_dir)
    out_dir = ensure_dir(args.output_dir)
    rows = []
    for r in range(args.max_k + 1):
        for s in range(args.max_k + 1):
            if r == 0 and s == 0:
                continue
            rows.append(one_result(results_dir, r, s))
    rows = sorted(rows, key=lambda x: (x["recent_k"], x["semantic_k"]))
    best = best_row(rows)
    with open(out_dir / "k_search_summary.json", "w", encoding="utf-8") as f:
        json.dump({"best": best, "rows": rows}, f, indent=2, default=json_default)
    write_csv(out_dir / "k_search_summary.csv", rows)
    plot_heatmap(rows, "valid_roc_auc", "Valid ROC-AUC", out_dir / "k_search_valid_roc_auc")
    plot_heatmap(rows, "valid_pr_auc", "Valid PR-AUC", out_dir / "k_search_valid_pr_auc")
    plot_heatmap(rows, "valid_cq_auc", "Valid CQ-AUC", out_dir / "k_search_valid_cq_auc")
    print("best:", best)
    print("summary:", out_dir / "k_search_summary.json")


if __name__ == "__main__":
    main()
