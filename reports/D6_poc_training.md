# Bonus: Proof-of-Concept Fine-Tuning

This note reports a leakage-aware LoRA run of `Qwen/Qwen2.5-0.5B-Instruct` on six Twin-2K-500 pricing items. The goal is a complete, auditable `data → train → predict → eval` loop under the assignment's fewer-than-0.5B constraint, not a full 17-task behavior model. Commands live in `poc/README.md`.

```text
safe Waves 1–3 persona + T1 labels
        ↓  PID split 70/15/15, seed 2026
LoRA fine-tune Qwen2.5-0.5B-Instruct
        ↓  freeze predictions for unseen people
score vs Wave-4 / T2
        ↓
compare with train-only item majority
```

## Task

Six product categories from the 40-item pricing task, identified by **product/category text** because Qualtrics slots are randomized:

- Pain Remedies - Headache
- Soft Drinks - Carbonated
- Cereal - Ready to Eat
- Bottled Water
- Fresh Eggs
- Fresh Fruit

Each target is binary (`1` = would purchase, `2` = would not). That yields several thousand supervised examples and one metric: exact-match accuracy.

The input persona is built only from `wave1_3_persona_json` (compact facts, ≤28 / 2,800 chars). `full_persona` is not used: the Hugging Face card notes that it substitutes Wave-4 answers on repeated items.

## Split and leakage

Deterministic **participant** split, seed `2026`: 8,646 train / 1,854 validation / 1,848 test examples.

| Split | Labels the model may see |
|---|---|
| Train / validation | T1 |
| Test inference (`test_inputs.jsonl`) | none |
| Primary score | frozen prediction vs **T2** |
| Secondary diagnostic | same prediction vs T1 |
| Human reliability | T1 vs T2 on the test PIDs |

`test_labels.jsonl` is opened only by the evaluator. Training and prediction scripts never read it.

## Model

`Qwen/Qwen2.5-0.5B-Instruct` is ~0.49B parameters. I used **LoRA rather than QLoRA** because the base model already fits a modest GPU; 4-bit quantization would add a `bitsandbytes` dependency without changing the scientific claim.

One epoch, context 1,024, LoRA `r=16` on `q/k/v/o` projections, learning rate `1e-4`, effective batch 16. Loss is restricted to the assistant completion (`1` or `2`).

## Results

Source: `poc/artifacts/pricing/metrics.json`. Test `n = 1,848` (308 people × 6 products). Every prediction parsed as `1` or `2`.

| Model | T2 accuracy | Δ vs item majority | vs T1 | Human T1↔T2 |
|---|---:|---:|---:|---:|
| Global majority (train T1) | 50.16% | — | — | 85.01% |
| Item majority (train T1) | 62.07% | — | — | 85.01% |
| Qwen2.5-0.5B + LoRA | **75.97%** | **+13.91 pp** | 75.92% | 85.01% |

The PID-bootstrap 95% CI on the item-majority lift is **[+11.31, +16.56] pp**. Relative to human retest on this slice, the model reaches **89.37%**.

| Product | Model | Item majority | Human T1↔T2 |
|---|---:|---:|---:|
| Fresh Eggs | 80.84% | 80.84% | 89.29% |
| Bottled Water | 79.55% | 70.78% | 82.14% |
| Fresh Fruit | 79.22% | 54.87% | 85.06% |
| Pain Remedies - Headache | 72.73% | 49.35% | 82.79% |
| Soft Drinks - Carbonated | 72.40% | 67.86% | 87.99% |
| Cereal - Ready to Eat | 71.10% | 48.70% | 82.79% |

## What this does and does not show

The 0.49B adapter **beats the strongest trivial baseline I allowed it**: a product-specific majority fitted only on training people’s original answers. The CI excludes zero, and invalid-output rate is 0%, so the structured-output path works.

Two patterns matter more than the headline:

1. **T1 and T2 scores are the same** (75.92% vs 75.97%). Frozen test predictions are not an obvious overfit to the original held-out answers at the expense of Wave 4.
2. **Lift tracks how weak the population prior is.** Eggs already have an 80.84% majority; the model matches it and adds nothing. Cereal, raspberries, and Tylenol sit near a coin-flip majority and improve by more than 20 pp. Personalized modeling is most useful where “what people usually do” is a poor guess.

I cannot claim the gain is *from the persona*. A no-persona ablation (question-only LoRA on the same split) is the next measurement; this run does not include it.

This slice also does **not** stand in for the Twin-2K-500 paper benchmark. It is six binary purchase items, not the 17-task range-normalized MAD protocol, and 89% of *this* human ceiling is not 89% of the published 81.7% test–retest number.

Reproduce with `bash poc/run_poc.sh` (details in `poc/README.md`).
