import argparse
import json
import os
from pathlib import Path

for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[name] = "8"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from evaluation.metrics import add_normalized_quality, cost_quality_auc
from router_utils import ensure_dir, json_default


METHODS = [
    ("classification_xgboost", "Classification Router"),
    ("gain_xgboost", "Gain Prediction Router"),
]


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_threshold_file(results_dir, method_name):
    if method_name == "classification_xgboost":
        files = sorted(results_dir.glob("*_xgboost_quality_cost_gap050_thresholds.json"))
        files = [p for p in files if "_gain_xgboost_" not in p.name]
    else:
        files = sorted(results_dir.glob("*_gain_xgboost_thresholds.json"))

    if not files:
        raise FileNotFoundError(f"No threshold file found for {method_name} in {results_dir}")
    return files[0]


def load_test_rows(path, model_name):
    rows = load_json(path)
    return [r for r in rows if r["split"] == "test" and r["model"] == model_name]


def plot_cq(results_dir, output_dir):
    summary = []
    plt.figure(figsize=(7.2, 5.4))

    for method_name, label in METHODS:
        threshold_file = find_threshold_file(results_dir, method_name)
        model_name = "xgboost" if method_name == "classification_xgboost" else "gain_xgboost"
        rows = load_test_rows(threshold_file, model_name)
        rows = sorted(add_normalized_quality(rows), key=lambda r: r["cost_saving_vs_always_strong"])
        cq_auc = cost_quality_auc(rows)

        summary.append(
            {
                "method": method_name,
                "label": label,
                "threshold_file": str(threshold_file),
                "cq_auc": cq_auc,
            }
        )

        xs = [0.0] + [r["cost_saving_vs_always_strong"] for r in rows] + [0.9]
        ys = [1.0] + [r["normalized_quality"] for r in rows] + [0.0]
        plt.plot(xs, ys, marker="o", markersize=3, linewidth=1.8, label=f"{label} ({cq_auc:.3f})")

    plt.xlabel("Cost Saving vs. Always Strong")
    plt.ylabel("Normalized Quality")
    plt.title("CQ Curves for Modeling Formulations")
    plt.xlim(0, 0.9)
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "gain_cq_curve.png", dpi=300)
    plt.savefig(output_dir / "gain_cq_curve.pdf")
    plt.close()
    return summary


def write_summary(output_dir, rows):
    with open(output_dir / "gain_cq_summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=json_default)

    best = max(rows, key=lambda r: r["cq_auc"])
    second = sorted(rows, key=lambda r: r["cq_auc"], reverse=True)[1] if len(rows) > 1 else None

    lines = [
        "\\begin{tabular}{lc}",
        "\\toprule",
        "Method & CQ-AUC $\\uparrow$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        value = f"{row['cq_auc']:.4f}"
        if row is best:
            value = "\\textcolor{red}{\\textbf{" + value + "}}"
        elif second is not None and row is second:
            value = "\\textcolor{blue}{\\textbf{" + value + "}}"
        lines.append(f"{row['label']} & {value} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])

    with open(output_dir / "gain_cq_summary.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot CQ curves for classification and gain prediction routers.")
    parser.add_argument("--results-dir", default="artifacts/results/05_gain_prediction")
    parser.add_argument("--output-dir", default="artifacts/plots/05_gain_prediction")
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = ensure_dir(args.output_dir)

    rows = plot_cq(results_dir, output_dir)
    write_summary(output_dir, rows)
    print(f"Wrote CQ plot and summary to {output_dir}")


if __name__ == "__main__":
    main()
