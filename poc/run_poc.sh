#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root, where data/ already exists.
python poc/prepare_pricing_poc.py \
  --data-dir data \
  --output-dir poc/artifacts/pricing

python poc/baseline_only.py \
  --prepared-dir poc/artifacts/pricing

python poc/train_qwen_lora.py \
  --prepared-dir poc/artifacts/pricing \
  --output-dir poc/artifacts/pricing/qwen_lora \
  --epochs 1 \
  --max-length 1024

python poc/predict_qwen_lora.py \
  --prepared-dir poc/artifacts/pricing \
  --adapter-dir poc/artifacts/pricing/qwen_lora/adapter \
  --output poc/artifacts/pricing/predictions.jsonl

python poc/evaluate_pricing_poc.py \
  --prepared-dir poc/artifacts/pricing \
  --predictions poc/artifacts/pricing/predictions.jsonl \
  --output poc/artifacts/pricing/metrics.json
