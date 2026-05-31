#!/usr/bin/env bash
FEATURES=artifacts/features/recent_semantic_nall.npz
OUT=artifacts/results/05_gain_prediction
PLOTS=artifacts/plots/05_gain_prediction
PYTHONPATH=src python src/train_router.py --features $FEATURES --output-dir $OUT --models xgboost --feature-set full_features --threshold-policy quality_cost --max-quality-gap 0.50 --num-thresholds 50
PYTHONPATH=src python src/train_gain_router.py --features $FEATURES --output-dir $OUT --feature-set full_features --num-thresholds 50
PYTHONPATH=src python src/evaluation/plot_gain_cq.py --results-dir $OUT --output-dir $PLOTS
