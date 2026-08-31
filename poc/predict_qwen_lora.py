#!/usr/bin/env python3
"""Generate frozen predictions for test_inputs.jsonl without reading test labels."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID_DEFAULT = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM_PROMPT = (
    "You predict how a specific survey participant would answer a purchase question. "
    "Use the participant persona and exact target stimulus. "
    "Return exactly one digit: 1 for 'Yes, I would purchase' or 2 for 'No, I would not purchase'."
)


def read_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


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


def parse_answer(text: str):
    m = re.search(r"(?<!\d)([12])(?!\d)", text.strip())
    return int(m.group(1)) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared-dir", type=Path, default=Path("poc/artifacts/pricing"))
    ap.add_argument("--adapter-dir", type=Path, default=Path("poc/artifacts/pricing/qwen_lora/adapter"))
    ap.add_argument("--model-id", type=str, default=MODEL_ID_DEFAULT)
    ap.add_argument("--output", type=Path, default=Path("poc/artifacts/pricing/predictions.jsonl"))
    ap.add_argument("--max-input-length", type=int, default=1024)
    ap.add_argument("--max-new-tokens", type=int, default=4)
    args = ap.parse_args()

    rows = read_jsonl(args.prepared_dir / "test_inputs.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        device = "cuda"
    else:
        dtype = torch.float32
        device = "cpu"

    base = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype)
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.to(device)
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_invalid = 0
    with args.output.open("w", encoding="utf-8") as f, torch.inference_mode():
        for idx, row in enumerate(rows, start=1):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt(row)},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            batch = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_input_length,
            ).to(device)
            generated = model.generate(
                **batch,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            new_tokens = generated[0, batch["input_ids"].shape[1] :]
            raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            pred = parse_answer(raw)
            if pred is None:
                n_invalid += 1
            record = {
                "pid": row["pid"],
                "qid": row["qid"],
                "item_key": row["item_key"],
                "category": row["category"],
                "prediction": pred,
                "raw_output": raw,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if idx % 100 == 0:
                print(f"predicted {idx}/{len(rows)}")

    print(f"Wrote {len(rows)} predictions to {args.output}; invalid={n_invalid}")


if __name__ == "__main__":
    main()
