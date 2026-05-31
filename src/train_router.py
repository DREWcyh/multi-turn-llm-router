import argparse
import json
import os
from pathlib import Path
for x in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[x] = "8"
import joblib
import numpy as np
from evaluation.metrics import baseline_metrics, combined_metrics, cost_quality_auc, pick_threshold, probabilities, threshold_metrics
from features.feature_selection import feature_set_suffix, get_feature_meta, select_feature_matrix
from models import DEFAULT_MODELS, build_models
from utils.io_utils import ensure_dir
def model_suffix(names):
    if len(names) == 1:
        return "_" + names[0]
    return "_models"
def threshold_suffix(policy, gap):
    if policy == "f1":
        return ""
    if policy == "quality_cost":
        return "_quality_cost_gap" + str(int(round(gap * 100))).zfill(3)
    return "_" + policy
def json_default(x):
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--output-dir", default="artifacts/results")
    p.add_argument("--models", default=",".join(DEFAULT_MODELS))
    p.add_argument("--feature-set", default="full_features")
    p.add_argument("--threshold-policy", default="f1", choices=["f1", "quality_cost"])
    p.add_argument("--max-quality-gap", type=float, default=0.30)
    p.add_argument("--num-thresholds", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true")
    return p.parse_args()
def main():
    args = get_args()
    out_dir = ensure_dir(args.output_dir)
    d = np.load(args.features, allow_pickle=True)
    x = d["X"]
    try:
        _, groups = get_feature_meta(d, x)
        x = select_feature_matrix(x, groups, args.feature_set)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    y = d["y"]
    weak = d["weak_score"]
    strong = d["strong_score"]
    split = d["split"].astype(str)
    tr = split == "train"
    va = split == "valid"
    te = split == "test"
    if not tr.any() or not va.any() or not te.any():
        raise SystemExit("need train valid test split")
    names = []
    for m in args.models.split(","):
        m = m.strip()
        if m:
            names.append(m)
    ms = build_models(args.seed, names)
    if not ms:
        raise SystemExit("no model selected")
    base = Path(args.features).stem + feature_set_suffix(args.feature_set)
    res_name = base + model_suffix(list(ms.keys())) + threshold_suffix(args.threshold_policy, args.max_quality_gap)
    metrics = []
    th_rows = []
    chosen = {}
    for r in baseline_metrics(y[te], weak[te], strong[te]):
        r = {"split": "test", **r}
        metrics.append(r)
    best = None
    thresholds = np.linspace(0.01, 0.99, args.num_thresholds)
    for name, model in ms.items():
        print("training", name, x[tr].shape)
        model.fit(x[tr], y[tr])
        vp = probabilities(model, x[va])
        vr = threshold_metrics("valid", name, y[va], vp, weak[va], strong[va], thresholds)
        th_rows += vr
        try:
            br = pick_threshold(vr, args.threshold_policy, args.max_quality_gap)
        except ValueError as e:
            raise SystemExit(str(e)) from e
        chosen[name] = br
        tp = probabilities(model, x[te])
        tt = threshold_metrics("test", name, y[te], tp, weak[te], strong[te], thresholds)
        th_rows += tt
        one = {
            "split": "test",
            "selected_valid_threshold": br,
            "cq_auc": cost_quality_auc(tt),
        }
        one.update(combined_metrics(name, y[te], tp, weak[te], strong[te], br["threshold"]))
        metrics.append(one)
        model_path = out_dir / f"{base}_{name}.joblib"
        joblib.dump(model, model_path)
        if best is None or one["f1"] > best["metrics"]["f1"]:
            best = {"model": name, "path": str(model_path), "metrics": one}
    metrics_path = out_dir / f"{res_name}_metrics.json"
    thresholds_path = out_dir / f"{res_name}_thresholds.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "feature_file": args.features,
            "feature_set": args.feature_set,
            "feature_dim": int(x.shape[1]),
            "threshold_policy": args.threshold_policy,
            "max_quality_gap": args.max_quality_gap,
            "num_thresholds": args.num_thresholds,
            "selected_thresholds": chosen,
            "best": best,
            "metrics": metrics,
        }, f, indent=2, default=json_default)
    with open(thresholds_path, "w", encoding="utf-8") as f:
        json.dump(th_rows, f, indent=2, default=json_default)
    print("metrics:", metrics_path)
    print("thresholds:", thresholds_path)
    print(json.dumps(best, indent=2, default=json_default))
if __name__ == "__main__":
    main()
