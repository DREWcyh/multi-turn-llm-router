#!/usr/bin/env bash
FEATURE_DIR=artifacts/features/k_search
OUT=artifacts/results/07_find_recent_semantic_k
PLOTS=artifacts/plots/07_find_recent_semantic_k
for r in 0 1 2 3 4 5
do
for s in 0 1 2 3 4 5
do
if [ "$r" = "0" ] && [ "$s" = "0" ]; then
continue
fi
NAME=recent_semantic_r${r}_s${s}_nall
PYTHONPATH=src python src/features/build_features.py --strategy recent_semantic --recent-k $r --semantic-k $s --output-dir $FEATURE_DIR --output-name $NAME
PYTHONPATH=src python src/train_router.py --features $FEATURE_DIR/${NAME}.npz --output-dir $OUT --models xgboost --feature-set full_features --threshold-policy quality_cost --max-quality-gap 0.50 --num-thresholds 50
done
done
PYTHONPATH=src python src/evaluation/plot_k_search.py --results-dir $OUT --output-dir $PLOTS
