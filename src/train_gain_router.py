import argparse
import json
import os
from pathlib import Path

for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[name] = "8"

import joblib
import numpy as np
from evaluation.metrics import add_normalized_quality, cost_quality_auc, route_metrics
from features.feature_selection import feature_set_suffix, get_feature_meta, select_feature_matrix
from router_utils import ensure_dir, json_default
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


def build_model(seed: int):
    return XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=seed,
    )


def make_thresholds(scores, num_thresholds: int):
    lo = float(np.min(scores))
    hi = float(np.max(scores))
    if lo == hi:
        return np.array([lo], dtype=np.float32)
    return np.linspace(lo, hi, num_thresholds, dtype=np.float32)


def score_route_rows(split, model_name, scores, weak, strong, thresholds):
    rows = []
    dummy_y = np.zeros(len(scores), dtype=np.int64)
    for threshold in thresholds:
        rows.append(
            {
                "split": split,
                "model": model_name,
                **route_metrics(dummy_y, scores, weak, strong, threshold),
            }
        )
    return add_normalized_quality(rows)


def regression_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson_corr": corr,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train a gain prediction router.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--output-dir", default="artifacts/results/05_gain_prediction")
    parser.add_argument("--feature-set", default="full_features")
    parser.add_argument("--num-thresholds", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)

    data = np.load(args.features, allow_pickle=True)
    x = data["X"]
    try:
        _, feature_groups = get_feature_meta(data, x)
        x = select_feature_matrix(x, feature_groups, args.feature_set)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    weak = data["weak_score"].astype(np.float32)
    strong = data["strong_score"].astype(np.float32)
    gain = (strong - weak).astype(np.float32)
    split = data["split"].astype(str)

    train_mask = split == "train"
    valid_mask = split == "valid"
    test_mask = split == "test"
    if not train_mask.any() or not valid_mask.any() or not test_mask.any():
        raise SystemExit("Expected train/valid/test splits in feature file.")

    model_name = "gain_xgboost"
    model = build_model(args.seed)
    print(f"Training {model_name} on {x[train_mask].shape}")
    model.fit(x[train_mask], gain[train_mask])

    valid_score = model.predict(x[valid_mask]).astype(np.float32)
    test_score = model.predict(x[test_mask]).astype(np.float32)
    thresholds = make_thresholds(valid_score, args.num_thresholds)

    valid_rows = score_route_rows("valid", model_name, valid_score, weak[valid_mask], strong[valid_mask], thresholds)
    test_rows = score_route_rows("test", model_name, test_score, weak[test_mask], strong[test_mask], thresholds)

    valid_cq_auc = cost_quality_auc(valid_rows)
    test_cq_auc = cost_quality_auc(test_rows)
    valid_reg = regression_metrics(gain[valid_mask], valid_score)
    test_reg = regression_metrics(gain[test_mask], test_score)

    base_prefix = Path(args.features).stem + feature_set_suffix(args.feature_set)
    result_prefix = base_prefix + "_gain_xgboost"

    model_path = out_dir / f"{result_prefix}.joblib"
    metrics_path = out_dir / f"{result_prefix}_metrics.json"
    thresholds_path = out_dir / f"{result_prefix}_thresholds.json"

    joblib.dump(model, model_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_file": args.features,
                "feature_set": args.feature_set,
                "feature_dim": int(x.shape[1]),
                "model": model_name,
                "target": "strong_score - weak_score",
                "num_thresholds": int(len(thresholds)),
                "score_threshold_min": float(np.min(thresholds)),
                "score_threshold_max": float(np.max(thresholds)),
                "metrics": [
                    {
                        "split": "valid",
                        "model": model_name,
                        "cq_auc": valid_cq_auc,
                        **valid_reg,
                    },
                    {
                        "split": "test",
                        "model": model_name,
                        "cq_auc": test_cq_auc,
                        **test_reg,
                    },
                ],
            },
            f,
            indent=2,
            default=json_default,
        )
    with open(thresholds_path, "w", encoding="utf-8") as f:
        json.dump(valid_rows + test_rows, f, indent=2, default=json_default)

    print(f"Wrote model to {model_path}")
    print(f"Wrote metrics to {metrics_path}")
    print(f"Wrote threshold rows to {thresholds_path}")


if __name__ == "__main__":
    main()
