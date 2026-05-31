#!/usr/bin/env bash
FEATURES=artifacts/features/query_only_nall.npz
OUT=artifacts/results/01_model_selection_query_only
PYTHONPATH=src python src/train_router.py --features $FEATURES --output-dir $OUT --models logreg --feature-set full_features --threshold-policy quality_cost --max-quality-gap 0.50 --num-thresholds 50
PYTHONPATH=src python src/train_router.py --features $FEATURES --output-dir $OUT --models svm --feature-set full_features --threshold-policy quality_cost --max-quality-gap 0.50 --num-thresholds 50
PYTHONPATH=src python src/train_router.py --features $FEATURES --output-dir $OUT --models gnb --feature-set full_features --threshold-policy quality_cost --max-quality-gap 0.50 --num-thresholds 50
PYTHONPATH=src python src/train_router.py --features $FEATURES --output-dir $OUT --models knn --feature-set full_features --threshold-policy quality_cost --max-quality-gap 0.50 --num-thresholds 50
PYTHONPATH=src python src/train_router.py --features $FEATURES --output-dir $OUT --models dt --feature-set full_features --threshold-policy quality_cost --max-quality-gap 0.50 --num-thresholds 50
PYTHONPATH=src python src/train_router.py --features $FEATURES --output-dir $OUT --models rf --feature-set full_features --threshold-policy quality_cost --max-quality-gap 0.50 --num-thresholds 50
PYTHONPATH=src python src/train_router.py --features $FEATURES --output-dir $OUT --models xgboost --feature-set full_features --threshold-policy quality_cost --max-quality-gap 0.50 --num-thresholds 50
PYTHONPATH=src python src/train_router.py --features $FEATURES --output-dir $OUT --models mlp --feature-set full_features --threshold-policy quality_cost --max-quality-gap 0.50 --num-thresholds 50
PYTHONPATH=src python src/evaluation/plot_curves.py --features $FEATURES --results-dir $OUT --output-dir artifacts/plots/01_model_selection_query_only
