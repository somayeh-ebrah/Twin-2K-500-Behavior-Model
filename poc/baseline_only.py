#!/usr/bin/env python3
"""Evaluate train-only majority baselines without requiring model predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared-dir", type=Path, default=Path("poc/artifacts/pricing"))
    args = ap.parse_args()
    labels = read_jsonl(args.prepared_dir / "test_labels.jsonl")
    cfg = json.loads((args.prepared_dir / "baseline.json").read_text(encoding="utf-8"))
    per = {str(k): int(v) for k, v in cfg["per_item_majority"].items()}
    global_m = int(cfg["global_majority"])

    n = len(labels)
    per_acc = sum(per[str(x["item_key"])] == int(x["t2"]) for x in labels) / n
    glob_acc = sum(global_m == int(x["t2"]) for x in labels) / n
    human = sum(int(x["t1"]) == int(x["t2"]) for x in labels) / n
    print(json.dumps({
        "n": n,
        "item_majority_t2_accuracy": per_acc,
        "global_majority_t2_accuracy": glob_acc,
        "human_t1_t2_reliability_on_slice": human,
    }, indent=2))


if __name__ == "__main__":
    main()
