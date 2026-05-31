#!/usr/bin/env bash
MODEL=xgboost
FEATURES=artifacts/features/recent_semantic_nall.npz
OUT=artifacts/results/04_single_feature_groups
PYTHONPATH=src python src/train_router.py --features $FEATURES --output-dir $OUT --models $MODEL --feature-set full_features --threshold-policy quality_cost --max-quality-gap 0.50 --num-thresholds 50
PYTHONPATH=src python src/train_router.py --features $FEATURES --output-dir $OUT --models $MODEL --feature-set only_semantic --threshold-policy quality_cost --max-quality-gap 0.50 --num-thresholds 50
PYTHONPATH=src python src/train_router.py --features $FEATURES --output-dir $OUT --models $MODEL --feature-set only_attribute --threshold-policy quality_cost --max-quality-gap 0.50 --num-thresholds 50
PYTHONPATH=src python src/train_router.py --features $FEATURES --output-dir $OUT --models $MODEL --feature-set only_query_semantic --threshold-policy quality_cost --max-quality-gap 0.50 --num-thresholds 50
PYTHONPATH=src python src/train_router.py --features $FEATURES --output-dir $OUT --models $MODEL --feature-set only_context_semantic --threshold-policy quality_cost --max-quality-gap 0.50 --num-thresholds 50
PYTHONPATH=src python src/train_router.py --features $FEATURES --output-dir $OUT --models $MODEL --feature-set only_relation_semantic --threshold-policy quality_cost --max-quality-gap 0.50 --num-thresholds 50
PYTHONPATH=src python src/train_router.py --features $FEATURES --output-dir $OUT --models $MODEL --feature-set only_query_attributes --threshold-policy quality_cost --max-quality-gap 0.50 --num-thresholds 50
PYTHONPATH=src python src/train_router.py --features $FEATURES --output-dir $OUT --models $MODEL --feature-set only_context_scale --threshold-policy quality_cost --max-quality-gap 0.50 --num-thresholds 50
PYTHONPATH=src python src/train_router.py --features $FEATURES --output-dir $OUT --models $MODEL --feature-set only_compression_attributes --threshold-policy quality_cost --max-quality-gap 0.50 --num-thresholds 50
PYTHONPATH=src python src/train_router.py --features $FEATURES --output-dir $OUT --models $MODEL --feature-set only_multi_turn_markers --threshold-policy quality_cost --max-quality-gap 0.50 --num-thresholds 50
PYTHONPATH=src python src/train_router.py --features $FEATURES --output-dir $OUT --models $MODEL --feature-set only_language_attributes --threshold-policy quality_cost --max-quality-gap 0.50 --num-thresholds 50
PYTHONPATH=src python src/evaluation/plot_single_feature_groups.py --results-dir $OUT --output-dir artifacts/plots/04_single_feature_groups
