# Multi-turn LLM Router

该项目主要针对多轮对话场景下的 LLM 路由：根据当前问题和历史上下文，判断这一轮应该调用弱模型还是强模型。


## 目录

```text
src/data/          数据预处理
src/features/      特征构建和特征选择
src/evaluation/    指标计算和画图
src/train_router.py        分类路由训练
src/train_gain_router.py   gain 预测实验
scripts/           实验脚本
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 准备 encoder 模型

本项目使用 `intfloat/multilingual-e5-small` 作为 sentence encoder。可以下载到本地：

```bash
pip install -U huggingface_hub
f
export HF_ENDPOINT=https://hf-mirror.com
hf download intfloat/multilingual-e5-small \
  --local-dir models/multilingual-e5-small
```

后面构建特征时使用：

```text
--embedding-model models/multilingual-e5-small
```

## 数据预处理

```bash
PYTHONPATH=src python -m data.build_dataset \
  --conversations data/conversations.jsonl \
  --features data/features/qwen06b_20k.jsonl \
  --output artifacts/samples/samples.jsonl \
  --label-field label_t2 \
  --force
```

## 特征构建

五种上下文策略分别构建特征：

```bash
PYTHONPATH=src python -m features.build_features --strategy query_only --embedding-model models/multilingual-e5-small --force
PYTHONPATH=src python -m features.build_features --strategy full --embedding-model models/multilingual-e5-small --force
PYTHONPATH=src python -m features.build_features --strategy recent_k --embedding-model models/multilingual-e5-small --force
PYTHONPATH=src python -m features.build_features --strategy semantic_top_k --embedding-model models/multilingual-e5-small --force
PYTHONPATH=src python -m features.build_features --strategy recent_semantic --embedding-model models/multilingual-e5-small --force
```

## 训练

单独训练一个模型：

```bash
PYTHONPATH=src python src/train_router.py \
  --features artifacts/features/recent_semantic_nall.npz \
  --models xgboost \
  --feature-set full_features \
  --threshold-policy quality_cost \
  --max-quality-gap 0.50
```

## 实验

按顺序运行：

```bash
./scripts/01_model_selection_query_only.sh
./scripts/02_context_strategy.sh
./scripts/03_feature_ablation.sh
./scripts/04_single_feature_groups.sh
./scripts/05_gain_prediction.sh
./scripts/06_5context_8model.sh
./scripts/07_find_recent_semantic_k.sh
```

其中：

```text
01  比较 query only 下的 8 个分类模型
02  比较 5 种上下文选择策略
03  特征大类消融实验
04  只使用某一类特征的实验
05  gain 预测实验
06  5 种上下文策略 × 8 个模型
07  recent_semantic 中 recent_k 和 semantic_k 的搜索
```

实验结果保存在：

```text
artifacts/results/
artifacts/plots/
```

## 主要指标

实验主要看三个 AUC：

```text
roc_auc    分类区分能力
pr_auc     正类识别能力
cq_auc     成本-质量权衡能力
```

其他辅助指标：

```text
accuracy, precision, recall, f1,
avg_quality, avg_cost, strong_call_rate,
cost_saving_vs_always_strong,
quality_gap_to_always_strong
```
