import argparse
import json
import os
from pathlib import Path

for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[name] = "8"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
from evaluation.metrics import add_normalized_quality, cost_quality_auc, probabilities
from features.feature_selection import get_feature_meta, select_feature_matrix
from router_utils import ensure_dir, json_default
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


FEATURE_SETS = [
    "full_features",
    "remove_semantic",
    "remove_attribute",
    "remove_query_semantic",
    "remove_context_semantic",
    "remove_relation_semantic",
    "remove_query_attributes",
    "remove_context_scale",
    "remove_compression_attributes",
    "remove_multi_turn_markers",
    "remove_language_attributes",
]

LABELS = {
    "full_features": "Full",
    "remove_semantic": "- Semantic",
    "remove_attribute": "- Attribute",
    "remove_query_semantic": "- Query Sem.",
    "remove_context_semantic": "- Context Sem.",
    "remove_relation_semantic": "- Relation Sem.",
    "remove_query_attributes": "- Query Attr.",
    "remove_context_scale": "- Context Scale",
    "remove_compression_attributes": "- Compression",
    "remove_multi_turn_markers": "- Markers",
    "remove_language_attributes": "- Language",
}


def suffix(feature_set):
    return "" if feature_set == "full_features" else f"_{feature_set}"


def stem(feature_set):
    return f"recent_semantic_nall{suffix(feature_set)}_xgboost_quality_cost_gap050"


def model_path(results_dir, feature_set):
    return results_dir / f"recent_semantic_nall{suffix(feature_set)}_xgboost.joblib"


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_metric(results_dir, feature_set):
    data = load_json(results_dir / f"{stem(feature_set)}_metrics.json")
    for row in data["metrics"]:
        if row.get("model") == "xgboost" and row.get("split") == "test":
            return row
    raise ValueError(f"No xgboost test metric found for {feature_set}")


def load_threshold_rows(results_dir, feature_set):
    rows = load_json(results_dir / f"{stem(feature_set)}_thresholds.json")
    return [r for r in rows if r["split"] == "test" and r["model"] == "xgboost"]


def load_test_data(data, feature_set):
    x = data["X"]
    _, feature_groups = get_feature_meta(data, x)
    x = select_feature_matrix(x, feature_groups, feature_set)
    mask = data["split"].astype(str) == "test"
    return x[mask], data["y"][mask]


def summarize(results_dir, feature_sets):
    rows = []
    for fs in feature_sets:
        metric = load_metric(results_dir, fs)
        threshold_rows = load_threshold_rows(results_dir, fs)
        rows.append(
            {
                "feature_set": fs,
                "label": LABELS.get(fs, fs),
                "threshold": metric["threshold"],
                "f1": metric["f1"],
                "roc_auc": metric["roc_auc"],
                "pr_auc": metric["pr_auc"],
                "cq_auc": cost_quality_auc(threshold_rows),
                "avg_quality": metric["avg_quality"],
                "cost_saving": metric["cost_saving_vs_always_strong"],
                "quality_gap": metric["quality_gap_to_always_strong"],
            }
        )
    return rows


def plot_roc(data, results_dir, output_dir, feature_sets):
    rows = []
    plt.figure(figsize=(7.8, 5.6))

    for fs in feature_sets:
        x_test, y_test = load_test_data(data, fs)
        model = joblib.load(model_path(results_dir, fs))
        prob = probabilities(model, x_test)
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = float(roc_auc_score(y_test, prob))
        rows.append({"feature_set": fs, "roc_auc": auc})
        plt.plot(fpr, tpr, linewidth=1.6, label=f"{LABELS.get(fs, fs)} ({auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Feature Removal")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output_dir / "feature_ablation_roc_curves.png", dpi=300)
    plt.savefig(output_dir / "feature_ablation_roc_curves.pdf")
    plt.close()
    return rows


def plot_pr(data, results_dir, output_dir, feature_sets):
    rows = []
    plt.figure(figsize=(7.8, 5.6))

    for fs in feature_sets:
        x_test, y_test = load_test_data(data, fs)
        model = joblib.load(model_path(results_dir, fs))
        prob = probabilities(model, x_test)
        precision, recall, _ = precision_recall_curve(y_test, prob)
        auc = float(average_precision_score(y_test, prob))
        rows.append({"feature_set": fs, "pr_auc": auc})
        plt.plot(recall, precision, linewidth=1.6, label=f"{LABELS.get(fs, fs)} ({auc:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves for Feature Removal")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output_dir / "feature_ablation_pr_curves.png", dpi=300)
    plt.savefig(output_dir / "feature_ablation_pr_curves.pdf")
    plt.close()
    return rows


def plot_cost_quality(results_dir, output_dir, feature_sets):
    rows = []
    plt.figure(figsize=(7.8, 5.6))

    for fs in feature_sets:
        test_rows = load_threshold_rows(results_dir, fs)
        test_rows = sorted(add_normalized_quality(test_rows), key=lambda r: r["cost_saving_vs_always_strong"])
        cq_auc = cost_quality_auc(test_rows)
        rows.append({"feature_set": fs, "cq_auc": cq_auc})

        xs = [0.0] + [r["cost_saving_vs_always_strong"] for r in test_rows] + [0.9]
        ys = [1.0] + [r["normalized_quality"] for r in test_rows] + [0.0]
        plt.plot(xs, ys, marker="o", markersize=2.6, linewidth=1.4, label=f"{LABELS.get(fs, fs)} ({cq_auc:.3f})")

    plt.xlabel("Cost Saving vs. Always Strong")
    plt.ylabel("Normalized Quality")
    plt.title("Cost-Quality Curves for Feature Removal")
    plt.xlim(0, 0.9)
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output_dir / "feature_ablation_cost_quality_curves.png", dpi=300)
    plt.savefig(output_dir / "feature_ablation_cost_quality_curves.pdf")
    plt.close()
    return rows


def remove_old_bar_plots(output_dir):
    names = [
        "feature_ablation_roc_auc",
        "feature_ablation_pr_auc",
        "feature_ablation_cq_auc",
    ]
    for name in names:
        for ext in [".png", ".pdf"]:
            path = output_dir / f"{name}{ext}"
            if path.exists():
                path.unlink()


def write_summary(rows, output_dir):
    with open(output_dir / "feature_ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=json_default)

    lines = [
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Feature Set & F1 $\\uparrow$ & ROC-AUC $\\uparrow$ & PR-AUC $\\uparrow$ & CQ-AUC $\\uparrow$ & Cost Saving $\\uparrow$ & Quality Gap $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['label']} & {row['f1']:.4f} & {row['roc_auc']:.4f} & {row['pr_auc']:.4f} & "
            f"{row['cq_auc']:.4f} & {row['cost_saving']:.4f} & {row['quality_gap']:.4f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    with open(output_dir / "feature_ablation_summary.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize feature ablation experiments.")
    parser.add_argument("--features", default="artifacts/features/recent_semantic_nall.npz")
    parser.add_argument("--results-dir", default="artifacts/results/03_feature_ablation")
    parser.add_argument("--output-dir", default="artifacts/plots/03_feature_ablation")
    parser.add_argument("--feature-sets", default=",".join(FEATURE_SETS))
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = ensure_dir(args.output_dir)
    feature_sets = [x.strip() for x in args.feature_sets.split(",") if x.strip()]
    data = np.load(args.features, allow_pickle=True)
    rows = summarize(results_dir, feature_sets)
    write_summary(rows, output_dir)
    remove_old_bar_plots(output_dir)
    plot_roc(data, results_dir, output_dir, feature_sets)
    plot_pr(data, results_dir, output_dir, feature_sets)
    plot_cost_quality(results_dir, output_dir, feature_sets)
    print(f"Wrote feature ablation summary to {output_dir}")


if __name__ == "__main__":
    main()
