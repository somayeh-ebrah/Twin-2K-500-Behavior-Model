#!/usr/bin/env python3
"""Prepare a leakage-safe pricing-task POC from Twin-2K-500.

Outputs:
  train.jsonl         safe persona + pricing target + T1 label (train PIDs)
  validation.jsonl    safe persona + pricing target + T1 label (val PIDs)
  test_inputs.jsonl   safe persona + pricing target only (test PIDs)
  test_labels.jsonl   quarantined T1/T2 labels for final evaluation
  baseline.json       train-only majority baselines
  pid_split_v1.json   deterministic participant split

The script reads only wave1_3 safe personas and T1/T2 answer blocks.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

SEED_DEFAULT = 2026
PRODUCT_CATEGORIES_DEFAULT = [
    "Pain Remedies - Headache",
    "Soft Drinks - Carbonated",
    "Cereal - Ready to Eat",
    "Bottled Water",
    "Fresh Eggs",
    "Fresh Fruit",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, str):
        obj = json.loads(obj)
    return obj


def iter_elements(obj: Any) -> Iterator[dict]:
    if isinstance(obj, list):
        for item in obj:
            yield from iter_elements(item)
    elif isinstance(obj, dict):
        if "Questions" in obj:
            yield obj
        nested = obj.get("Elements")
        if isinstance(nested, list):
            for item in nested:
                yield from iter_elements(item)


def iter_questions(obj: Any) -> Iterator[Tuple[str, dict]]:
    for el in iter_elements(obj):
        block = str(el.get("BlockName") or el.get("Description") or "Unknown")
        for q in el.get("Questions") or []:
            if isinstance(q, dict):
                yield block, q


def answer_text(q: dict, index: Optional[int] = None) -> Optional[str]:
    answers = q.get("Answers") or {}
    qtype = q.get("QuestionType")

    if qtype == "DB" or not answers:
        return None

    if qtype == "TE":
        value = answers.get("Text")
        return None if value in (None, "") else str(value)

    if qtype == "Slider":
        vals = answers.get("Values") or []
        if not isinstance(vals, list):
            vals = [vals]
        if not vals:
            return None
        if index is None:
            return str(vals[0])
        return str(vals[index]) if index < len(vals) else None

    selected_text = answers.get("SelectedText")
    selected_pos = answers.get("SelectedByPosition")
    if qtype == "Matrix":
        texts = selected_text if isinstance(selected_text, list) else []
        poss = selected_pos if isinstance(selected_pos, list) else []
        if index is None:
            index = 0
        text = texts[index] if index < len(texts) else None
        pos = poss[index] if index < len(poss) else None
        if text not in (None, "") and pos not in (None, ""):
            return f"{pos} | {text}"
        if text not in (None, ""):
            return str(text)
        if pos not in (None, ""):
            return str(pos)
        return None

    if isinstance(selected_text, list):
        selected_text = selected_text[0] if selected_text else None
    if isinstance(selected_pos, list):
        selected_pos = selected_pos[0] if selected_pos else None
    if selected_text not in (None, "") and selected_pos not in (None, ""):
        return f"{selected_pos} | {selected_text}"
    if selected_text not in (None, ""):
        return str(selected_text)
    if selected_pos not in (None, ""):
        return str(selected_pos)
    return None


def compact(text: Any, max_chars: int = 220) -> str:
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    return s if len(s) <= max_chars else s[: max_chars - 1] + "…"


def flatten_persona_facts(persona: Any) -> List[Tuple[int, str]]:
    """Return prioritized, leakage-safe readable persona facts.

    Priority is intentionally simple for the POC:
      0 demographics
      1 economic preferences
      2 personality
      3 cognitive/other

    This is not claimed to be optimal retrieval; it keeps the POC small and auditable.
    """
    facts: List[Tuple[int, str]] = []
    for block, q in iter_questions(persona):
        qtype = q.get("QuestionType")
        if qtype == "DB":
            continue

        b = block.lower()
        if "demographic" in b:
            priority = 0
        elif "economic" in b:
            priority = 1
        elif "personality" in b:
            priority = 2
        else:
            priority = 3

        qtext = compact(q.get("QuestionText"), 180)
        qid = str(q.get("QuestionID") or "")

        if qtype == "Matrix":
            rows = q.get("Rows") or q.get("Statements") or []
            n = max(len(rows), len((q.get("Answers") or {}).get("SelectedByPosition") or []))
            for idx in range(n):
                ans = answer_text(q, idx)
                if ans is None:
                    continue
                row = compact(rows[idx] if idx < len(rows) else f"item {idx + 1}", 120)
                fact = f"[{block}] {qid}:{idx+1} | {qtext} | {row} -> {compact(ans, 100)}"
                facts.append((priority, fact))
        else:
            ans = answer_text(q)
            if ans is None:
                continue
            fact = f"[{block}] {qid} | {qtext} -> {compact(ans, 100)}"
            facts.append((priority, fact))

    return facts


def build_compact_persona(persona: Any, max_chars: int = 2800, max_facts: int = 28) -> str:
    facts = flatten_persona_facts(persona)
    # Stable sort by priority while preserving original within-priority order.
    facts = [f for _, f in sorted(enumerate(facts), key=lambda x: (x[1][0], x[0]))]
    lines: List[str] = []
    used = 0
    for priority, fact in facts:
        del priority
        addition = len(fact) + 1
        if len(lines) >= max_facts or used + addition > max_chars:
            break
        lines.append(fact)
        used += addition
    return "\n".join(lines)


def parse_pricing_target(question_text: str) -> Optional[dict]:
    text = re.sub(r"\s+", " ", str(question_text or "")).strip()
    pattern = re.compile(
        r"product category:\s*(.*?)\.\s*Suppose.*?following product in that category:\s*"
        r"(.*?)\.\s*The product is priced at:\s*\$([0-9]+(?:\.[0-9]+)?)",
        flags=re.IGNORECASE,
    )
    m = pattern.search(text)
    if not m:
        return None
    category = m.group(1).strip()
    product = m.group(2).strip()
    price = float(m.group(3))
    return {
        "category": category,
        "product": product,
        "price": price,
        "item_key": f"{category} | {product}",
    }


def extract_pricing_questions(answer_block: Any, categories: Sequence[str]) -> Dict[str, dict]:
    wanted = set(categories)
    out: Dict[str, dict] = {}
    for block, q in iter_questions(answer_block):
        qid = str(q.get("QuestionID") or "")
        if not qid.startswith("QID9_") or q.get("QuestionType") != "MC":
            continue
        parsed = parse_pricing_target(q.get("QuestionText") or "")
        if parsed is None or parsed["category"] not in wanted:
            continue
        pos = (q.get("Answers") or {}).get("SelectedByPosition")
        if isinstance(pos, list):
            pos = pos[0] if pos else None
        try:
            label = int(pos)
        except (TypeError, ValueError):
            continue
        if label not in (1, 2):
            continue
        out[parsed["item_key"]] = {
            "qid": qid,
            "block": block,
            "question": compact(q.get("QuestionText"), 1200),
            "options": list(q.get("Options") or []),
            "label": label,
            **parsed,
        }
    return out


def find_pids(persona_dir: Path) -> List[int]:
    pids = []
    for p in persona_dir.glob("pid_*_mega_persona.json"):
        m = re.search(r"pid_(\d+)", p.name)
        if m:
            pids.append(int(m.group(1)))
    return sorted(set(pids))


def make_split(pids: Sequence[int], seed: int) -> Dict[str, List[int]]:
    pids = list(pids)
    rng = random.Random(seed)
    rng.shuffle(pids)
    n = len(pids)
    n_train = int(round(n * 0.70))
    n_val = int(round(n * 0.15))
    train = sorted(pids[:n_train])
    val = sorted(pids[n_train : n_train + n_val])
    test = sorted(pids[n_train + n_val :])
    return {"seed": seed, "train": train, "validation": val, "test": test}


def cap_pids(pids: Sequence[int], cap: Optional[int]) -> List[int]:
    if cap is None or cap <= 0 or cap >= len(pids):
        return list(pids)
    return list(pids)[:cap]


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def mode(values: Sequence[int]) -> int:
    counts = Counter(values)
    # Deterministic tie break toward lower answer id.
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--output-dir", type=Path, default=Path("poc/artifacts/pricing"))
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument(
        "--categories",
        type=str,
        default=";;".join(PRODUCT_CATEGORIES_DEFAULT),
        help="Pricing categories separated by ' ;; '",
    )
    ap.add_argument("--persona-max-chars", type=int, default=2800)
    ap.add_argument("--persona-max-facts", type=int, default=28)
    ap.add_argument("--max-train-pids", type=int, default=0, help="0 = all train PIDs")
    ap.add_argument("--max-val-pids", type=int, default=0, help="0 = all validation PIDs")
    ap.add_argument("--max-test-pids", type=int, default=0, help="0 = all test PIDs")
    args = ap.parse_args()

    categories = [x.strip() for x in args.categories.split(";;") if x.strip()]
    persona_dir = args.data_dir / "mega_persona_json" / "mega_persona"
    answer_dir = args.data_dir / "mega_persona_json" / "answer_blocks"
    if not persona_dir.exists() or not answer_dir.exists():
        raise SystemExit(
            f"Expected Twin-2K-500 data under {args.data_dir}. "
            "Run download_dataset.py first."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_pids = find_pids(persona_dir)
    split = make_split(all_pids, args.seed)
    split["train"] = cap_pids(split["train"], args.max_train_pids)
    split["validation"] = cap_pids(split["validation"], args.max_val_pids)
    split["test"] = cap_pids(split["test"], args.max_test_pids)

    split_path = args.output_dir / "pid_split_v1.json"
    split_path.write_text(json.dumps(split, indent=2), encoding="utf-8")

    persona_cache: Dict[int, str] = {}

    def persona_text(pid: int) -> str:
        if pid not in persona_cache:
            raw = load_json(persona_dir / f"pid_{pid}_mega_persona.json")
            persona_cache[pid] = build_compact_persona(
                raw,
                max_chars=args.persona_max_chars,
                max_facts=args.persona_max_facts,
            )
        return persona_cache[pid]

    train_rows: List[dict] = []
    val_rows: List[dict] = []
    test_inputs: List[dict] = []
    test_labels: List[dict] = []

    train_labels_by_item: Dict[str, List[int]] = defaultdict(list)
    train_all_labels: List[int] = []

    for split_name in ("train", "validation", "test"):
        for pid in split[split_name]:
            t1 = load_json(answer_dir / f"pid_{pid}_wave4_Q_wave1_3_A.json")
            t2 = load_json(answer_dir / f"pid_{pid}_wave4_Q_wave4_A.json")
            p1 = extract_pricing_questions(t1, categories)
            p2 = extract_pricing_questions(t2, categories)
            common = sorted(set(p1) & set(p2))
            for item_key in common:
                q1, q2 = p1[item_key], p2[item_key]
                # Structural safety check: same target wording/options at T1 and T2.
                if q1["question"] != q2["question"] or q1["options"] != q2["options"]:
                    raise RuntimeError(f"T1/T2 pricing mismatch pid={pid} item={item_key}")

                base = {
                    "pid": pid,
                    "qid": q1["qid"],
                    "item_key": item_key,
                    "category": q1["category"],
                    "product": q1["product"],
                    "price": q1["price"],
                    "persona": persona_text(pid),
                    "target_question": q1["question"],
                    "options": q1["options"],
                }

                if split_name == "train":
                    row = {**base, "answer": q1["label"]}
                    train_rows.append(row)
                    train_labels_by_item[item_key].append(q1["label"])
                    train_all_labels.append(q1["label"])
                elif split_name == "validation":
                    val_rows.append({**base, "answer": q1["label"]})
                else:
                    # Test inputs intentionally contain no target labels.
                    test_inputs.append(base)
                    test_labels.append(
                        {
                            "pid": pid,
                            "qid": q1["qid"],
                            "item_key": item_key,
                            "category": q1["category"],
                            "product": q1["product"],
                            "price": q1["price"],
                            "t1": q1["label"],
                            "t2": q2["label"],
                        }
                    )

    if not train_rows or not val_rows or not test_inputs:
        raise RuntimeError("Prepared split is empty; check categories and data paths.")

    baseline = {
        "fit_on": "training T1 labels only",
        "global_majority": mode(train_all_labels),
        "per_item_majority": {
            str(k): mode(v) for k, v in sorted(train_labels_by_item.items())
        },
        "train_label_counts": dict(Counter(train_all_labels)),
        "categories": categories,
    }

    counts = {
        "train": write_jsonl(args.output_dir / "train.jsonl", train_rows),
        "validation": write_jsonl(args.output_dir / "validation.jsonl", val_rows),
        "test_inputs": write_jsonl(args.output_dir / "test_inputs.jsonl", test_inputs),
        "test_labels": write_jsonl(args.output_dir / "test_labels.jsonl", test_labels),
    }
    (args.output_dir / "baseline.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "categories": categories,
                "counts": counts,
                "persona_max_chars": args.persona_max_chars,
                "persona_max_facts": args.persona_max_facts,
                "note": "train/validation labels are T1; test_inputs contains no labels; final primary label is T2",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps({"output_dir": str(args.output_dir), "counts": counts, "baseline": baseline}, indent=2))


if __name__ == "__main__":
    main()
