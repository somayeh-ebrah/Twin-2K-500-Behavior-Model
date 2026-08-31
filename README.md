# Twin-2K-500-Behavior-Model


The prediction task is: given a person's leakage-safe Waves 1–3 persona, predict how that same person answers the held-out Wave-4 questions, and score those predictions against Wave-4 answers (T2) and against human test–retest (T1 vs T2).

This repository is the submission. The reports are the main deliverable; the code is the exploration notebook, shared helpers, and a six-product pricing POC under the assignment's fewer-than-0.5B constraint.

## Deliverables

| # | Assignment item | In this repo |
|---|---|---|
| 1 | Data exploration | [`reports/D1_data_exploration.md`](reports/D1_data_exploration.md), [`notebooks/01_data_exploration.ipynb`](notebooks/01_data_exploration.ipynb) |
| 2 | Plan to build the model | [`reports/D2_plan_to_build_the_model.md`](reports/D2_plan_to_build_the_model.md) |
| 3 | Evaluation strategy | [`reports/D3_evaluation_strategy.md`](reports/D3_evaluation_strategy.md) |
| 4 | Business applications | [`reports/D4_business_applications.md`](reports/D4_business_applications.md) |
| 5 | Long-run maintenance | [`reports/D5_long_run_maintenance.md`](reports/D5_long_run_maintenance.md) |
| 6 | Bonus POC | [`reports/D6_poc_training.md`](reports/D6_poc_training.md), [`poc/README.md`](poc/README.md) |

Start with D1 (data, leakage trap, human ceiling) then D2–D5. D6 is a slice, not a 17-task model.

## Repository layout

```text
.
├── download_dataset.py          # writes data/ and cache/ from Hugging Face
├── src/
│   ├── data_utils.py            # local file loaders and schema helpers
│   └── evaluation_utils.py      # MAD / test–retest scoring from T1/T2 JSON
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_poc_demo_test_sample.ipynb
├── reports/                     # D1–D6 write-ups and figures
├── poc/                         # pricing LoRA loop (data → train → predict → eval)
├── requirements.txt
└── environment.yml              # Conda env name: asf-env
```

`data/` is created by `download_dataset.py`. It is not in the repo (see `.gitignore`).

## Environment

The working environment is the Conda env **`asf-env`** (Python 3.11). Recreate it from this repository:

```bash
conda env create -f environment.yml
conda activate asf-env
```

If `asf-env` already exists, install into it instead:

```bash
conda activate asf-env
pip install -r requirements.txt
```

`environment.yml` installs the same packages listed in `requirements.txt`. Versions match the `asf-env` used to run the notebooks and POC.

PyTorch in that env was `2.7.1+cu128`. `requirements.txt` pins `torch>=2.7.1` without a CUDA wheel tag so a CPU install still resolves. For a CUDA 12.8 wheel like the original env:

```bash
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

POC training (`poc/train_qwen_lora.py`) expects a GPU. Prepare, baseline, and evaluate are CPU scripts. The demo notebook reads frozen artifacts and does not train.

## Download the dataset

From the repository root (creates `data/` and `./cache`):

```bash
python download_dataset.py
```

That script loads Hugging Face configs `wave_split` and `full_persona` plus the eight raw Qualtrics CSVs, and writes:

| Local path | Source |
|---|---|
| `data/mega_persona_json/mega_persona/` | `wave1_3_persona_json` |
| `data/mega_persona_json/answer_blocks/` | T1 `wave4_Q_wave1_3_A`, T2 `wave4_Q_wave4_A` |
| `data/mega_persona_summary_text/` | `full_persona` summaries |
| `data/wave_csv/` | `raw_data/wave_*_{labels,numbers}_anonymized.csv` |

Dataset: [LLM-Digital-Twin/Twin-2K-500](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500) (CC BY 4.0).


## Trained model weights

The trained LoRA adapter used for the reported POC results is available here:

[Google Drive — Qwen2.5-0.5B LoRA adapter](https://drive.google.com/file/d/1tY1yYzI66frrhe7Ks9gWj-9WFDPNC_gx/view?usp=sharing
)

Base model: `Qwen/Qwen2.5-0.5B-Instruct`

## Data exploration

With `data/` present, from the repository root:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

Helpers: `src/data_utils.py`, `src/evaluation_utils.py`. Headline from the local files: human test–retest **81.73%** equal-task-weight accuracy (17 tasks, N = 2,058). Details in D1.

## Bonus POC

Leakage-aware LoRA on six binary pricing items using `Qwen/Qwen2.5-0.5B-Instruct`. Full contract, hyperparameters, and per-product table: [`poc/README.md`](poc/README.md). Write-up: [`reports/D6_poc_training.md`](reports/D6_poc_training.md).

Needs `data/mega_persona_json/` already downloaded. From the repository root, with `asf-env` active:

```bash
bash poc/run_poc.sh
```

That shell script runs, in order:

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

Smoke test (two categories, fewer PIDs):

```bash
python poc/prepare_pricing_poc.py --data-dir data --output-dir poc/artifacts/smoke \
  --max-train-pids 128 --max-val-pids 64 --max-test-pids 64 \
  --categories "Pain Remedies - Headache;;Fresh Eggs"
python poc/train_qwen_lora.py --prepared-dir poc/artifacts/smoke \
  --output-dir poc/artifacts/smoke/qwen_lora --epochs 1 --max-length 768
```

Then predict/evaluate against `poc/artifacts/smoke`.

Inspect frozen test rows without a GPU:

```bash
jupyter notebook notebooks/02_poc_demo_test_sample.ipynb
```

That notebook needs `poc/artifacts/pricing/{test_inputs,test_labels,predictions,baseline,metrics}.json(l)`.

### POC results

Source: `poc/artifacts/pricing/metrics.json`. One LoRA epoch, n = 1,848 (308 people × 6 products). Invalid outputs: 0.

| Model | T2 accuracy | Δ vs item majority | vs T1 | Human T1↔T2 |
|---|---:|---:|---:|---:|
| Global majority (train T1) | 50.16% | — | — | 85.01% |
| Item majority (train T1) | 62.07% | — | — | 85.01% |
| Qwen2.5-0.5B + LoRA | **75.97%** | **+13.91 pp** | 75.92% | 85.01% |

PID-bootstrap 95% CI on the lift: **[+11.31, +16.56] pp**. This is six binary purchase items, not the paper's 17-task MAD benchmark. Caveats: `reports/D6_poc_training.md`.

## Leakage

- Split by **PID**, not row. Training never sees a test person's target labels.
- Persona input is `wave1_3_persona_json` only. Do not use `full_persona` / `persona_summary` as the primary persona: the Hugging Face card notes that `full_persona` substitutes Wave-4 answers on repeated items.
- Train/validation labels are **T1**. Primary test label is **T2**. `poc/predict_qwen_lora.py` reads `test_inputs.jsonl` only; `test_labels.jsonl` is opened by the evaluator.

## Dataset and paper

- Toubia, O., Gui, G. Z., Peng, T., Merlau, D. J., Li, A., & Chen, H. (2025). *Twin-2K-500: A dataset for building digital twins of over 2,000 people based on their answers to over 500 questions*. [arXiv:2505.17479](https://arxiv.org/abs/2505.17479)
- Dataset docs: [digital-twin-simulation-version2.readthedocs.io](https://digital-twin-simulation-version2.readthedocs.io/en/latest/index.html)
- Reference implementation: [tianyipeng-lab/Digital-Twin-Simulation](https://github.com/tianyipeng-lab/Digital-Twin-Simulation)

