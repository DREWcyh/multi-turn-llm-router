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


MODEL_NAMES = ["logreg", "svm", "gnb", "knn", "dt", "rf", "xgboost", "mlp"]
MODEL_LABELS = {
    "logreg": "Logistic Regression",
    "svm": "Linear SVM",
    "gnb": "Gaussian NB",
    "knn": "KNN",
    "dt": "Decision Tree",
    "rf": "Random Forest",
    "xgboost": "XGBoost",
    "mlp": "MLP",
}


def find_model_file(results_dir, model_name):
    files = sorted(results_dir.glob(f"*_{model_name}.joblib"))
    if not files:
        raise FileNotFoundError(f"No joblib file found for {model_name} in {results_dir}")
    return files[0]


def find_threshold_file(results_dir, model_name):
    files = sorted(results_dir.glob(f"*_{model_name}_*thresholds.json"))
    if not files:
        files = sorted(results_dir.glob(f"*_{model_name}_thresholds.json"))
    if not files:
        raise FileNotFoundError(f"No threshold file found for {model_name} in {results_dir}")
    return files[0]


def load_test_data(feature_path):
    data = np.load(feature_path, allow_pickle=True)
    split = data["split"].astype(str)
    mask = split == "test"
    return data["X"][mask], data["y"][mask]


def plot_roc(feature_path, results_dir, output_dir, models):
    x_test, y_test = load_test_data(feature_path)
    rows = []

    plt.figure(figsize=(7.2, 5.4))
    for model_name in models:
        model = joblib.load(find_model_file(results_dir, model_name))
        prob = probabilities(model, x_test)
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = float(roc_auc_score(y_test, prob))
        rows.append({"model": model_name, "roc_auc": auc})
        plt.plot(fpr, tpr, linewidth=1.8, label=f"{MODEL_LABELS[model_name]} ({auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves on Test Set")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curves.png", dpi=300)
    plt.savefig(output_dir / "roc_curves.pdf")
    plt.close()
    return rows


def plot_pr(feature_path, results_dir, output_dir, models):
    x_test, y_test = load_test_data(feature_path)
    rows = []

    plt.figure(figsize=(7.2, 5.4))
    for model_name in models:
        model = joblib.load(find_model_file(results_dir, model_name))
        prob = probabilities(model, x_test)
        precision, recall, _ = precision_recall_curve(y_test, prob)
        auc = float(average_precision_score(y_test, prob))
        rows.append({"model": model_name, "pr_auc": auc})
        plt.plot(recall, precision, linewidth=1.8, label=f"{MODEL_LABELS[model_name]} ({auc:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves on Test Set")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curves.png", dpi=300)
    plt.savefig(output_dir / "pr_curves.pdf")
    plt.close()
    return rows


def load_test_threshold_rows(path):
    rows = json.load(open(path, encoding="utf-8"))
    return [r for r in rows if r["split"] == "test"]


def plot_cost_quality(results_dir, output_dir, models):
    rows = []

    plt.figure(figsize=(7.2, 5.4))
    for model_name in models:
        threshold_file = find_threshold_file(results_dir, model_name)
        test_rows = load_test_threshold_rows(threshold_file)
        test_rows = sorted(add_normalized_quality(test_rows), key=lambda r: r["cost_saving_vs_always_strong"])
        cq_auc = cost_quality_auc(test_rows)
        rows.append({"model": model_name, "cq_auc": cq_auc})

        xs = [0.0] + [r["cost_saving_vs_always_strong"] for r in test_rows] + [0.9]
        ys = [1.0] + [r["normalized_quality"] for r in test_rows] + [0.0]
        plt.plot(xs, ys, marker="o", markersize=3, linewidth=1.6, label=f"{MODEL_LABELS[model_name]} ({cq_auc:.3f})")

    plt.xlabel("Cost Saving vs. Always Strong")
    plt.ylabel("Normalized Quality")
    plt.title("Cost-Quality Curves on Test Set")
    plt.xlim(0, 0.9)
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "cost_quality_curves.png", dpi=300)
    plt.savefig(output_dir / "cost_quality_curves.pdf")
    plt.close()
    return rows


def write_summary(results_dir, output_dir, roc_rows, pr_rows, cq_rows):
    by_model = {}
    for row in roc_rows + pr_rows + cq_rows:
        by_model.setdefault(row["model"], {}).update(row)

    rows = []
    for model_name in MODEL_NAMES:
        if model_name in by_model:
            metric_file = find_metrics_file(results_dir, model_name)
            metrics = load_model_metric(metric_file, model_name) if metric_file else {}
            row = {"model": model_name, "label": MODEL_LABELS[model_name], **by_model[model_name], **metrics}
            rows.append(row)

    with open(output_dir / "curve_summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=json_default)

    lines = [
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Model & ROC-AUC $\\uparrow$ & PR-AUC $\\uparrow$ & CQ-AUC $\\uparrow$ & F1 $\\uparrow$ & Cost Saving $\\uparrow$ & Quality Gap $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['label']} & {row['roc_auc']:.4f} & {row['pr_auc']:.4f} & {row['cq_auc']:.4f} & "
            f"{row.get('f1', 0.0):.4f} & {row.get('cost_saving', 0.0):.4f} & {row.get('quality_gap', 0.0):.4f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    with open(output_dir / "curve_summary.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def find_metrics_file(results_dir, model_name):
    files = sorted(results_dir.glob(f"*_{model_name}_*metrics.json"))
    if not files:
        files = sorted(results_dir.glob(f"*_{model_name}_metrics.json"))
    return files[0] if files else None


def load_model_metric(path, model_name):
    data = json.load(open(path, encoding="utf-8"))
    for row in data.get("metrics", []):
        if row.get("model") == model_name and row.get("split") == "test":
            return {
                "f1": row["f1"],
                "cost_saving": row["cost_saving_vs_always_strong"],
                "quality_gap": row["quality_gap_to_always_strong"],
            }
    return {}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot ROC and cost-quality curves.")
    parser.add_argument("--features", default="artifacts/features/recent_semantic_nall.npz")
    parser.add_argument("--results-dir", default="artifacts/results/model_selection_gap050")
    parser.add_argument("--output-dir", default="artifacts/plots")
    parser.add_argument("--models", default="logreg,svm,gnb,knn,dt,rf,xgboost,mlp")
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = ensure_dir(args.output_dir)
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    roc_rows = plot_roc(Path(args.features), results_dir, output_dir, models)
    pr_rows = plot_pr(Path(args.features), results_dir, output_dir, models)
    cq_rows = plot_cost_quality(results_dir, output_dir, models)
    write_summary(results_dir, output_dir, roc_rows, pr_rows, cq_rows)
    print(f"Wrote plots and summary to {output_dir}")


if __name__ == "__main__":
    main()
