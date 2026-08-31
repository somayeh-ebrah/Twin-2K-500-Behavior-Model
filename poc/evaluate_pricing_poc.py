#!/usr/bin/env python3
"""Evaluate frozen POC predictions against T2 and train-only baselines."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def read_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def accuracy(pairs: Sequence[Tuple[int, int]]) -> float:
    return sum(int(a == b) for a, b in pairs) / len(pairs) if pairs else float("nan")


def cluster_bootstrap_delta(records: List[dict], n_boot: int = 2000, seed: int = 2026):
    by_pid: Dict[int, List[dict]] = defaultdict(list)
    for r in records:
        by_pid[int(r["pid"])].append(r)
    pids = sorted(by_pid)
    rng = random.Random(seed)
    deltas = []
    for _ in range(n_boot):
        sample = [rng.choice(pids) for _ in pids]
        m_num = m_den = b_num = b_den = 0
        for pid in sample:
            for r in by_pid[pid]:
                if r["prediction"] is not None:
                    m_num += int(r["prediction"] == r["t2"])
                    m_den += 1
                b_num += int(r["baseline"] == r["t2"])
                b_den += 1
        if m_den and b_den:
            deltas.append(m_num / m_den - b_num / b_den)
    deltas.sort()
    if not deltas:
        return (float("nan"), float("nan"))
    lo = deltas[int(0.025 * (len(deltas) - 1))]
    hi = deltas[int(0.975 * (len(deltas) - 1))]
    return lo, hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared-dir", type=Path, default=Path("poc/artifacts/pricing"))
    ap.add_argument("--predictions", type=Path, default=Path("poc/artifacts/pricing/predictions.jsonl"))
    ap.add_argument("--output", type=Path, default=Path("poc/artifacts/pricing/metrics.json"))
    args = ap.parse_args()

    labels = read_jsonl(args.prepared_dir / "test_labels.jsonl")
    preds = read_jsonl(args.predictions)
    baseline_cfg = json.loads((args.prepared_dir / "baseline.json").read_text(encoding="utf-8"))

    label_map = {(int(x["pid"]), str(x["item_key"])): x for x in labels}
    pred_map = {(int(x["pid"]), str(x["item_key"])): x for x in preds}
    if set(label_map) != set(pred_map):
        missing = set(label_map) - set(pred_map)
        extra = set(pred_map) - set(label_map)
        raise RuntimeError(f"Prediction/label key mismatch: missing={len(missing)} extra={len(extra)}")

    per_item_majority = {str(k): int(v) for k, v in baseline_cfg["per_item_majority"].items()}
    global_majority = int(baseline_cfg["global_majority"])

    records = []
    for key in sorted(label_map):
        lab = label_map[key]
        pr = pred_map[key]
        records.append(
            {
                "pid": key[0],
                "item_key": key[1],
                "category": lab["category"],
                "prediction": pr.get("prediction"),
                "t1": int(lab["t1"]),
                "t2": int(lab["t2"]),
                "baseline": per_item_majority[key[1]],
                "global_baseline": global_majority,
            }
        )

    valid = [r for r in records if r["prediction"] in (1, 2)]
    invalid_rate = 1.0 - len(valid) / len(records)
    model_t2 = accuracy([(r["prediction"], r["t2"]) for r in valid])
    model_t1 = accuracy([(r["prediction"], r["t1"]) for r in valid])
    baseline_t2 = accuracy([(r["baseline"], r["t2"]) for r in records])
    global_t2 = accuracy([(r["global_baseline"], r["t2"]) for r in records])
    human = accuracy([(r["t1"], r["t2"]) for r in records])
    ci_lo, ci_hi = cluster_bootstrap_delta(records)

    by_item = {}
    for item_key in sorted({r["item_key"] for r in records}):
        rr = [r for r in records if r["item_key"] == item_key]
        vv = [r for r in rr if r["prediction"] in (1, 2)]
        by_item[item_key] = {
            "category": rr[0]["category"],
            "n": len(rr),
            "model_t2_accuracy": accuracy([(r["prediction"], r["t2"]) for r in vv]),
            "baseline_t2_accuracy": accuracy([(r["baseline"], r["t2"]) for r in rr]),
            "human_t1_t2_accuracy": accuracy([(r["t1"], r["t2"]) for r in rr]),
        }

    metrics = {
        "n_examples": len(records),
        "n_valid_model_predictions": len(valid),
        "invalid_output_rate": invalid_rate,
        "primary_model_t2_accuracy": model_t2,
        "item_majority_baseline_t2_accuracy": baseline_t2,
        "global_majority_baseline_t2_accuracy": global_t2,
        "secondary_model_t1_accuracy": model_t1,
        "human_t1_t2_reliability": human,
        "model_over_item_baseline_delta": model_t2 - baseline_t2,
        "paired_pid_bootstrap_delta_95ci": [ci_lo, ci_hi],
        "model_fraction_of_human_reliability": model_t2 / human if human else None,
        "by_item": by_item,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
