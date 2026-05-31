import argparse
import json
import os
from pathlib import Path

for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[name] = "8"

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from evaluation.metrics import add_normalized_quality, cost_quality_auc, probabilities
from router_utils import ensure_dir, json_default
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


STRATEGIES = ["query_only", "full", "recent_k", "semantic_top_k", "recent_semantic"]
LABELS = {
    "query_only": "Query Only",
    "full": "Full",
    "recent_k": "Recent-k",
    "semantic_top_k": "Semantic Top-k",
    "recent_semantic": "Recent + Semantic",
}


def feature_path(feature_dir, strategy):
    return feature_dir / f"{strategy}_nall.npz"


def result_stem(strategy):
    return f"{strategy}_nall_xgboost_quality_cost_gap050"


def load_test_data(path):
    data = np.load(path, allow_pickle=True)
    split = data["split"].astype(str)
    mask = split == "test"
    return data["X"][mask], data["y"][mask]


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_strategy_metric(results_dir, strategy):
    path = results_dir / f"{result_stem(strategy)}_metrics.json"
    data = load_json(path)
    rows = [r for r in data["metrics"] if r.get("model") == "xgboost"]
    if not rows:
        raise ValueError(f"No xgboost metric found in {path}")
    return rows[0]


def load_test_threshold_rows(results_dir, strategy):
    path = results_dir / f"{result_stem(strategy)}_thresholds.json"
    rows = load_json(path)
    return [r for r in rows if r["split"] == "test" and r["model"] == "xgboost"]


def plot_roc(feature_dir, results_dir, output_dir, strategies):
    rows = []
    plt.figure(figsize=(7.2, 5.4))

    for strategy in strategies:
        x_test, y_test = load_test_data(feature_path(feature_dir, strategy))
        model = joblib.load(results_dir / f"{strategy}_nall_xgboost.joblib")
        prob = probabilities(model, x_test)
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = float(roc_auc_score(y_test, prob))
        rows.append({"strategy": strategy, "roc_auc": auc})
        plt.plot(fpr, tpr, linewidth=1.8, label=f"{LABELS[strategy]} ({auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Context Strategies")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "context_roc_curves.png", dpi=300)
    plt.savefig(output_dir / "context_roc_curves.pdf")
    plt.close()
    return rows


def plot_pr(feature_dir, results_dir, output_dir, strategies):
    rows = []
    plt.figure(figsize=(7.2, 5.4))

    for strategy in strategies:
        x_test, y_test = load_test_data(feature_path(feature_dir, strategy))
        model = joblib.load(results_dir / f"{strategy}_nall_xgboost.joblib")
        prob = probabilities(model, x_test)
        precision, recall, _ = precision_recall_curve(y_test, prob)
        auc = float(average_precision_score(y_test, prob))
        rows.append({"strategy": strategy, "pr_auc": auc})
        plt.plot(recall, precision, linewidth=1.8, label=f"{LABELS[strategy]} ({auc:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves for Context Strategies")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "context_pr_curves.png", dpi=300)
    plt.savefig(output_dir / "context_pr_curves.pdf")
    plt.close()
    return rows


def plot_cost_quality(results_dir, output_dir, strategies):
    rows = []
    plt.figure(figsize=(7.2, 5.4))

    for strategy in strategies:
        test_rows = load_test_threshold_rows(results_dir, strategy)
        test_rows = sorted(add_normalized_quality(test_rows), key=lambda r: r["cost_saving_vs_always_strong"])
        cq_auc = cost_quality_auc(test_rows)
        rows.append({"strategy": strategy, "cq_auc": cq_auc})

        xs = [0.0] + [r["cost_saving_vs_always_strong"] for r in test_rows] + [0.9]
        ys = [1.0] + [r["normalized_quality"] for r in test_rows] + [0.0]
        plt.plot(xs, ys, marker="o", markersize=3, linewidth=1.6, label=f"{LABELS[strategy]} ({cq_auc:.3f})")

    plt.xlabel("Cost Saving vs. Always Strong")
    plt.ylabel("Normalized Quality")
    plt.title("Cost-Quality Curves for Context Strategies")
    plt.xlim(0, 0.9)
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "context_cost_quality_curves.png", dpi=300)
    plt.savefig(output_dir / "context_cost_quality_curves.pdf")
    plt.close()
    return rows


def write_summary(results_dir, output_dir, roc_rows, pr_rows, cq_rows, strategies):
    by_strategy = {}
    for row in roc_rows + pr_rows + cq_rows:
        by_strategy.setdefault(row["strategy"], {}).update(row)

    rows = []
    for strategy in strategies:
        metric = load_strategy_metric(results_dir, strategy)
        row = {
            "strategy": strategy,
            "label": LABELS[strategy],
            **by_strategy[strategy],
            "threshold": metric["threshold"],
            "f1": metric["f1"],
            "avg_quality": metric["avg_quality"],
            "avg_cost": metric["avg_cost"],
            "cost_saving": metric["cost_saving_vs_always_strong"],
            "quality_gap": metric["quality_gap_to_always_strong"],
        }
        rows.append(row)

    with open(output_dir / "context_strategy_summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=json_default)

    lines = [
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "Strategy & Threshold & F1 $\\uparrow$ & ROC-AUC $\\uparrow$ & PR-AUC $\\uparrow$ & CQ-AUC $\\uparrow$ & Cost Saving $\\uparrow$ & Quality Gap $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['label']} & {row['threshold']:.2f} & {row['f1']:.4f} & {row['roc_auc']:.4f} & "
            f"{row['pr_auc']:.4f} & {row['cq_auc']:.4f} & {row['cost_saving']:.4f} & {row['quality_gap']:.4f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])

    with open(output_dir / "context_strategy_summary.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot context strategy results.")
    parser.add_argument("--feature-dir", default="artifacts/features")
    parser.add_argument("--results-dir", default="artifacts/results/context_strategy_gap050")
    parser.add_argument("--output-dir", default="artifacts/plots/context_strategy")
    parser.add_argument("--strategies", default="query_only,full,recent_k,semantic_top_k,recent_semantic")
    return parser.parse_args()


def main():
    args = parse_args()
    feature_dir = Path(args.feature_dir)
    results_dir = Path(args.results_dir)
    output_dir = ensure_dir(args.output_dir)
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    roc_rows = plot_roc(feature_dir, results_dir, output_dir, strategies)
    pr_rows = plot_pr(feature_dir, results_dir, output_dir, strategies)
    cq_rows = plot_cost_quality(results_dir, output_dir, strategies)
    write_summary(results_dir, output_dir, roc_rows, pr_rows, cq_rows, strategies)
    print(f"Wrote context strategy plots to {output_dir}")


if __name__ == "__main__":
    main()
