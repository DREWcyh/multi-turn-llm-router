#!/usr/bin/env bash
MODEL=xgboost
FEATURES=artifacts/features/recent_semantic_nall.npz
OUT=artifacts/results/03_feature_ablation
PYTHONPATH=src python src/train_router.py --features $FEATURES --output-dir $OUT --models $MODEL --feature-set full_features --threshold-policy quality_cost --max-quality-gap 0.50 --num-thresholds 50
PYTHONPATH=src python src/train_router.py --features $FEATURES --output-dir $OUT --models $MODEL --feature-set remove_semantic --threshold-policy quality_cost --max-quality-gap 0.50 --num-thresholds 50
PYTHONPATH=src python src/train_router.py --features $FEATURES --output-dir $OUT --models $MODEL --feature-set remove_attribute --threshold-policy quality_cost --max-quality-gap 0.50 --num-thresholds 50
PYTHONPATH=src python src/train_router.py --features $FEATURES --output-dir $OUT --models $MODEL --feature-set remove_query_semantic --threshold-policy quality_cost --max-quality-gap 0.50 --num-thresholds 50
PYTHONPATH=src python src/train_router.py --features $FEATURES --output-dir $OUT --models $MODEL --feature-set remove_context_semantic --threshold-policy quality_cost --max-quality-gap 0.50 --num-thresholds 50
PYTHONPATH=src python src/train_router.py --features $FEATURES --output-dir $OUT --models $MODEL --feature-set remove_relation_semantic --threshold-policy quality_cost --max-quality-gap 0.50 --num-thresholds 50
PYTHONPATH=src python src/train_router.py --features $FEATURES --output-dir $OUT --models $MODEL --feature-set remove_query_attributes --threshold-policy quality_cost --max-quality-gap 0.50 --num-thresholds 50
PYTHONPATH=src python src/train_router.py --features $FEATURES --output-dir $OUT --models $MODEL --feature-set remove_context_scale --threshold-policy quality_cost --max-quality-gap 0.50 --num-thresholds 50
PYTHONPATH=src python src/train_router.py --features $FEATURES --output-dir $OUT --models $MODEL --feature-set remove_compression_attributes --threshold-policy quality_cost --max-quality-gap 0.50 --num-thresholds 50
PYTHONPATH=src python src/train_router.py --features $FEATURES --output-dir $OUT --models $MODEL --feature-set remove_multi_turn_markers --threshold-policy quality_cost --max-quality-gap 0.50 --num-thresholds 50
PYTHONPATH=src python src/train_router.py --features $FEATURES --output-dir $OUT --models $MODEL --feature-set remove_language_attributes --threshold-policy quality_cost --max-quality-gap 0.50 --num-thresholds 50
PYTHONPATH=src python src/evaluation/plot_feature_ablation.py --results-dir $OUT --output-dir artifacts/plots/03_feature_ablation
