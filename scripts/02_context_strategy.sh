#!/usr/bin/env bash
MODEL=xgboost
OUT=artifacts/results/02_context_strategy
PYTHONPATH=src python src/train_router.py --features artifacts/features/query_only_nall.npz --output-dir $OUT --models $MODEL --feature-set full_features --threshold-policy quality_cost --max-quality-gap 0.50 --num-thresholds 50
PYTHONPATH=src python src/train_router.py --features artifacts/features/full_nall.npz --output-dir $OUT --models $MODEL --feature-set full_features --threshold-policy quality_cost --max-quality-gap 0.50 --num-thresholds 50
PYTHONPATH=src python src/train_router.py --features artifacts/features/recent_k_nall.npz --output-dir $OUT --models $MODEL --feature-set full_features --threshold-policy quality_cost --max-quality-gap 0.50 --num-thresholds 50
PYTHONPATH=src python src/train_router.py --features artifacts/features/semantic_top_k_nall.npz --output-dir $OUT --models $MODEL --feature-set full_features --threshold-policy quality_cost --max-quality-gap 0.50 --num-thresholds 50
PYTHONPATH=src python src/train_router.py --features artifacts/features/recent_semantic_nall.npz --output-dir $OUT --models $MODEL --feature-set full_features --threshold-policy quality_cost --max-quality-gap 0.50 --num-thresholds 50
PYTHONPATH=src python src/evaluation/plot_context_strategy.py --feature-dir artifacts/features --results-dir $OUT --output-dir artifacts/plots/02_context_strategy
