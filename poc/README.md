# Bonus POC — Qwen2.5-0.5B on Twin-2K-500 pricing

Leakage-aware `data → train → predict → eval` loop on a six-product binary pricing slice. The write-up is `reports/D6_poc_training.md`.

```text
safe Waves 1–3 persona + T1 labels
        ↓  PID 70/15/15 (seed 2026)
LoRA fine-tune Qwen2.5-0.5B-Instruct
        ↓  freeze preds for unseen PIDs
score vs Wave-4 / T2
        ↓
compare to train-only item majority
```

## Contract

- Split by **PID**, not row. A person is never in both train and test.
- Train/val labels: **T1**. Primary test label: **T2**. Same frozen preds also scored vs T1.
- Human ceiling on the slice: T1 vs T2.
- Baseline: per-item majority fit on **train T1 only**.
- Persona from `wave1_3_persona_json` only. Never `full_persona`.

| File | Contents |
|---|---|
| `train.jsonl` / `validation.jsonl` | persona + question + T1 |
| `test_inputs.jsonl` | persona + question, **no answers** |
| `test_labels.jsonl` | T1 and T2; evaluator only |
| `baseline.json` | train-T1 majority map |

Counts: **8,646 / 1,854 / 1,848**.

## Slice and model

Six pricing categories, matched by product text (slots are randomized): headache remedies, carbonated soft drinks, cereal, bottled water, fresh eggs, fresh fruit. Each item is `1` = would buy, `2` = would not.

Persona is a compact fact list (≤28 facts / 2,800 chars) so a 1,024-token context still has room for the question.

`Qwen/Qwen2.5-0.5B-Instruct` (~0.49B) meets the assignment's fewer-than-0.5B constraint. **LoRA, not QLoRA**: the base model fits a modest GPU, so 4-bit quantization is extra complexity.

```yaml
max_length: 1024
epochs: 1
learning_rate: 1e-4
micro_batch_size: 2
gradient_accumulation_steps: 8
scheduler: cosine
warmup_ratio: 0.05
weight_decay: 0.01
lora: {r: 16, alpha: 32, dropout: 0.05, target_modules: [q_proj, k_proj, v_proj, o_proj]}
```

Loss is on the assistant completion (`1`/`2`) only. Prompt tokens are masked.

## Run

Needs `data/mega_persona_json/` already downloaded. From the repo root, with the `asf-env` Conda environment active (`conda activate asf-env`; see the root `README.md` and `environment.yml`):

```bash
pip install -r requirements.txt
bash poc/run_poc.sh
```

Or stage by stage:

```bash
python poc/prepare_pricing_poc.py --data-dir data --output-dir poc/artifacts/pricing
python poc/baseline_only.py --prepared-dir poc/artifacts/pricing
python poc/train_qwen_lora.py --prepared-dir poc/artifacts/pricing \
  --output-dir poc/artifacts/pricing/qwen_lora --epochs 1 --max-length 1024
python poc/predict_qwen_lora.py --prepared-dir poc/artifacts/pricing \
  --adapter-dir poc/artifacts/pricing/qwen_lora/adapter \
  --output poc/artifacts/pricing/predictions.jsonl
python poc/evaluate_pricing_poc.py --prepared-dir poc/artifacts/pricing \
  --predictions poc/artifacts/pricing/predictions.jsonl \
  --output poc/artifacts/pricing/metrics.json
```

Predict never opens `test_labels.jsonl`.

Smoke test (two categories, fewer PIDs):

```bash
python poc/prepare_pricing_poc.py --data-dir data --output-dir poc/artifacts/smoke \
  --max-train-pids 128 --max-val-pids 64 --max-test-pids 64 \
  --categories "Pain Remedies - Headache;;Fresh Eggs"
python poc/train_qwen_lora.py --prepared-dir poc/artifacts/smoke \
  --output-dir poc/artifacts/smoke/qwen_lora --epochs 1 --max-length 768
```

Then predict/evaluate against `poc/artifacts/smoke`.

## Results

Source: `poc/artifacts/pricing/metrics.json`. One LoRA epoch, `n = 1,848` (308 people × 6 products). Invalid outputs: 0.

| Model | T2 | Δ vs item majority | vs T1 | Human T1↔T2 |
|---|---:|---:|---:|---:|
| Global majority (train T1) | 50.16% | — | — | 85.01% |
| Item majority (train T1) | 62.07% | — | — | 85.01% |
| Qwen2.5-0.5B + LoRA | **75.97%** | **+13.91 pp** | 75.92% | 85.01% |

PID-bootstrap 95% CI on the lift: **[+11.31, +16.56] pp**. Model / human reliability: **89.37%**.

| Product | Model | Item majority | Human |
|---|---:|---:|---:|
| Fresh Eggs | 80.84% | 80.84% | 89.29% |
| Bottled Water | 79.55% | 70.78% | 82.14% |
| Fresh Fruit | 79.22% | 54.87% | 85.06% |
| Pain Remedies - Headache | 72.73% | 49.35% | 82.79% |
| Soft Drinks - Carbonated | 72.40% | 67.86% | 87.99% |
| Cereal - Ready to Eat | 71.10% | 48.70% | 82.79% |

Lift is large where the population prior is weak (fruit, headache, cereal) and zero on eggs, where majority is already strong. T1 ≈ T2, so this is not a T1-only overfit. Interpretation and caveats: `reports/D6_poc_training.md`.

## Files

```text
poc/
├── README.md
├── requirements.txt
├── run_poc.sh
├── prepare_pricing_poc.py
├── baseline_only.py
├── train_qwen_lora.py
├── predict_qwen_lora.py
└── evaluate_pricing_poc.py
```
