#!/usr/bin/env python3
"""LoRA fine-tune Qwen2.5-0.5B-Instruct on the prepared pricing POC.

This script intentionally reads only train.jsonl and validation.jsonl. It never
opens test_labels.jsonl.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

MODEL_ID_DEFAULT = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM_PROMPT = (
    "You predict how a specific survey participant would answer a purchase question. "
    "Use the participant persona and exact target stimulus. "
    "Return exactly one digit: 1 for 'Yes, I would purchase' or 2 for 'No, I would not purchase'."
)


def read_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def as_token_ids(encoded) -> List[int]:
    """Normalize apply_chat_template(tokenize=True) across transformers versions.

    Transformers 5+ returns a BatchEncoding dict {"input_ids", "attention_mask"}
    (return_dict defaults to True). Older versions returned a bare list of ids.
    The SFT mask logic must see the id list, not the dict keys.
    """
    if hasattr(encoded, "keys") and "input_ids" in encoded:
        ids = encoded["input_ids"]
    else:
        ids = encoded
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def chat_token_ids(tokenizer, messages: Sequence[dict], *, add_generation_prompt: bool) -> List[int]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    return as_token_ids(encoded)


def user_prompt(row: dict) -> str:
    options = "\n".join(f"{i+1}. {text}" for i, text in enumerate(row["options"]))
    return (
        "<PERSONA>\n"
        f"{row['persona']}\n"
        "</PERSONA>\n\n"
        "<TARGET>\n"
        f"qid={row['qid']}\n"
        f"{row['target_question']}\n"
        f"Options:\n{options}\n"
        "</TARGET>\n\n"
        "Answer with exactly 1 or 2."
    )


class SFTDataset(Dataset):
    def __init__(self, rows: List[dict], tokenizer, max_length: int):
        self.examples = []
        for row in rows:
            prompt_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt(row)},
            ]
            full_messages = prompt_messages + [
                {"role": "assistant", "content": str(int(row["answer"]))}
            ]

            prompt_ids = chat_token_ids(
                tokenizer, prompt_messages, add_generation_prompt=True
            )
            full_ids = chat_token_ids(
                tokenizer, full_messages, add_generation_prompt=False
            )

            # If over length, trim from the LEFT so the target question and answer survive.
            if len(full_ids) > max_length:
                overflow = len(full_ids) - max_length
                full_ids = full_ids[overflow:]
                prompt_ids = prompt_ids[min(overflow, len(prompt_ids)) :]

            # Find common prefix after any truncation; mask prompt tokens from loss.
            prefix = 0
            for a, b in zip(prompt_ids, full_ids):
                if a != b:
                    break
                prefix += 1
            labels = [-100] * prefix + full_ids[prefix:]
            if all(x == -100 for x in labels):
                raise RuntimeError("No assistant target tokens remain after tokenization.")

            self.examples.append(
                {
                    "input_ids": full_ids,
                    "attention_mask": [1] * len(full_ids),
                    "labels": labels,
                }
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


@dataclass
class CausalLMCollator:
    pad_token_id: int

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in features)
        input_ids, attention_mask, labels = [], [], []
        for x in features:
            n = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [self.pad_token_id] * n)
            attention_mask.append(x["attention_mask"] + [0] * n)
            labels.append(x["labels"] + [-100] * n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared-dir", type=Path, default=Path("poc/artifacts/pricing"))
    ap.add_argument("--output-dir", type=Path, default=Path("poc/artifacts/pricing/qwen_lora"))
    ap.add_argument("--model-id", type=str, default=MODEL_ID_DEFAULT)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    set_seed(args.seed)
    train_rows = read_jsonl(args.prepared_dir / "train.jsonl")
    val_rows = read_jsonl(args.prepared_dir / "validation.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Base model parameters: {n_params:,}")
    if n_params >= 500_000_000:
        raise RuntimeError(
            f"Assignment requires <0.5B parameters; loaded model has {n_params:,}."
        )

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    train_ds = SFTDataset(train_rows, tokenizer, args.max_length)
    val_ds = SFTDataset(val_rows, tokenizer, args.max_length)

    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    use_fp16 = bool(torch.cuda.is_available() and not use_bf16)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size * 2),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=True,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=CausalLMCollator(tokenizer.pad_token_id),
    )
    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(args.output_dir / "adapter")
    tokenizer.save_pretrained(args.output_dir / "adapter")
    (args.output_dir / "training_config.json").write_text(
        json.dumps(vars(args), default=str, indent=2), encoding="utf-8"
    )
    print(f"Saved adapter to {args.output_dir / 'adapter'}")


if __name__ == "__main__":
    main()
