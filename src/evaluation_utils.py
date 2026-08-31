"""Independent MAD scoring for Twin-2K-500 hold-out tasks.

The protocol follows evaluation/mad_accuracy_evaluation.py and the paper:
  accuracy = 1 - |prediction - truth| / allowed_range
Binary items (range 1) reduce to exact match. Unbounded anchoring estimates
are converted to deciles before scoring.

Task names, column ranges, and the equal-task aggregation are taken from the
repository evaluation code. Extraction is implemented here from the T1/T2
answer-block JSON rather than by calling the paper's evaluation script.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import sem, t as t_dist

from .data_utils import (
    ANSWER_BLOCK_DIR,
    REPO_ROOT,
    WAVE_CSV_DIR,
    iter_questions,
    load_json,
    pid_from_filename,
)

RNG_SEED = 42

# Parallel false-consensus batteries use Qualtrics choice IDs that skip 8 and 9.
FALSE_CONSENSUS_ITEM_IDS = ["1", "2", "3", "4", "5", "6", "7", "10", "11", "12"]

ANCHOR_GROUP_A = ["Q164", "Q166"]  # African-countries numeric estimates
ANCHOR_GROUP_B = ["Q168", "Q170"]  # redwood numeric estimates


def get_column_ranges() -> Dict[str, Tuple[float, float]]:
    """Min/max used by evaluation/mad_accuracy_evaluation.py (uppercased keys).

    Comments in that file swap Allais/myside/WTA labels; the numeric ranges
    match the actual option counts in the JSON/docs, so we keep the values.
    """
    ranges: Dict[str, Tuple[float, float]] = {}
    ranges.update({f"FALSE CONS. SELF _{i}": (1, 5) for i in range(1, 11)})
    ranges.update({f"FALSE CONS. OTHERS _{i}": (0, 100) for i in [1, 2, 3, 4, 5, 6, 7, 10, 11, 12]})
    ranges["Q156_1"] = (0, 100)
    ranges["FORM A _1"] = (0, 100)
    for code in ["157", "158"] + [f"160_{i}" for i in (1, 2, 3)] + [f"159_{i}" for i in (1, 2, 3)]:
        ranges[f"Q{code}"] = (1, 6)
    for code in ("161", "162"):
        ranges[f"Q{code}"] = (1, 7)
    for code in ("164", "166", "168", "170"):
        ranges[f"Q{code}"] = (1, 10)  # after decile transform
    for code in ("171", "172", "173", "174", "175", "176"):
        ranges[f"Q{code}"] = (1, 5)
    for code in ("177", "178", "179"):
        ranges[f"Q{code}"] = (1, 6)
    ranges["Q181"] = (0, 20)
    ranges["Q182"] = (0, 20)
    for code in ("183", "184"):
        ranges[f"Q{code}"] = (1, 2)
    for code in ("189", "190", "191"):
        ranges[f"Q{code}"] = (1, 10)
    for code in ("192", "193"):
        ranges[f"Q{code}"] = (1, 2)
    for code in ("194", "195"):
        ranges[f"Q{code}"] = (1, 6)
    ranges.update({f"Q198_{i}": (1, 2) for i in range(1, 11)})
    ranges.update({f"Q203_{i}": (1, 2) for i in range(1, 7)})
    ranges.update({f"NONSEPARABILTY BENE _{i}": (1, 7) for i in range(1, 5)})
    ranges.update({f"NONSEPARABILITY RIS _{i}": (1, 7) for i in range(1, 5)})
    ranges["OMISSION BIAS "] = (1, 4)
    ranges["DENOMINATOR NEGLECT "] = (1, 2)
    ranges.update({f"{i}_Q295": (1, 2) for i in range(1, 41)})
    return {k.upper(): v for k, v in ranges.items()}


def get_qid_to_task() -> Dict[str, str]:
    """Column (wave-4 name) -> 17 evaluation tasks. Keys uppercased."""
    raw = {
        **{f"False Cons. self _{i}": "false consensus" for i in range(1, 11)},
        **{f"False cons. others _{i}": "false consensus" for i in [1, 2, 3, 4, 5, 6, 7, 10, 11, 12]},
        "Q156_1": "base rate",
        "Form A _1": "base rate",
        "Q157": "framing problem",
        "Q158": "framing problem",
        **{f"Q160_{i}": "conjunction problem (Linda)" for i in [1, 2, 3]},
        **{f"Q159_{i}": "conjunction problem (Linda)" for i in [1, 2, 3]},
        "Q161": "outcome bias",
        "Q162": "outcome bias",
        "Q164": "anchoring and adjustment",
        "Q166": "anchoring and adjustment",
        "Q168": "anchoring and adjustment",
        "Q170": "anchoring and adjustment",
        **{f"Q17{i}": "less is more" for i in range(1, 10)},
        "Q181": "sunk cost fallacy",
        "Q182": "sunk cost fallacy",
        "Q183": "absolute vs. relative savings",
        "Q184": "absolute vs. relative savings",
        "Q189": "WTA/WTP-Thaler",
        "Q190": "WTA/WTP-Thaler",
        "Q191": "WTA/WTP-Thaler",
        "Q192": "Allais",
        "Q193": "Allais",
        "Q194": "myside",
        "Q195": "myside",
        **{f"Q198_{i}": "prob matching vs. max" for i in range(1, 11)},
        **{f"Q203_{i}": "prob matching vs. max" for i in range(1, 7)},
        **{f"nonseparabilty bene _{i}": "non-separability of risks and benefits" for i in range(1, 5)},
        **{f"nonseparability ris _{i}": "non-separability of risks and benefits" for i in range(1, 5)},
        "Omission bias ": "omission",
        "Denominator neglect ": "denominator neglect",
        **{f"{i}_Q295": "pricing" for i in range(1, 41)},
    }
    return {k.upper(): v for k, v in raw.items()}


def task_catalog() -> pd.DataFrame:
    """Compact table of scored tasks, item counts, and scoring method."""
    qid_to_task = get_qid_to_task()
    ranges = get_column_ranges()
    rows = []
    for task in sorted(set(qid_to_task.values())):
        cols = [c for c, t in qid_to_task.items() if t == task]
        rs = [ranges[c] for c in cols if c in ranges]
        uniq = sorted(set(rs))
        if uniq == [(1, 2)]:
            method = "Exact match (binary); accuracy 0/1"
            rtype = "binary / 2-option MC"
        elif task == "anchoring and adjustment":
            method = "Decile-bin unbounded estimates, then 1 - |d|/9"
            rtype = "unbounded numeric (TE)"
        elif task in {"base rate"} or "false consensus" in task:
            method = "1 - |pred-truth| / range  (Likert 1–5 and/or 0–100)"
            rtype = "ordinal Likert + 0–100 slider"
        elif task == "pricing":
            method = "Exact match (buy/not buy)"
            rtype = "binary MC"
        else:
            lo, hi = uniq[0] if len(uniq) == 1 else (min(u[0] for u in uniq), max(u[1] for u in uniq))
            method = f"1 - |pred-truth| / ({hi}-{lo})"
            rtype = "ordinal / bounded numeric"
        rows.append(
            {
                "task": task,
                "response_type": rtype,
                "n_scored_columns": len(cols),
                "note": "Between-subject tasks: each person answers one variant, not all columns",
                "scoring_method": method,
            }
        )
    return pd.DataFrame(rows)


def _to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if pd.isna(value):
            return None
        return float(value)
    try:
        return float(str(value).strip().replace(",", ""))
    except (TypeError, ValueError):
        return None


def _item_ids(question: dict, n_items: int) -> List[str]:
    qid = question.get("QuestionID") or ""
    rows_id = question.get("RowsID") or []
    statements_id = question.get("StatementsID") or []
    if rows_id:
        return [str(x) for x in rows_id]
    if statements_id:
        return [str(x) for x in statements_id]
    # False-consensus "others" slider has 10 statements but Qualtrics IDs skip 8–9.
    if n_items == 10 and qid in {"QID287", "QID290"}:
        return list(FALSE_CONSENSUS_ITEM_IDS)
    if n_items == 10 and not rows_id and not statements_id:
        # Same policy battery as QID287; keep ImportId alignment.
        text = str(question.get("QuestionText") or "")
        if "percentage of the public" in text.lower() or "supports the following" in text.lower():
            return list(FALSE_CONSENSUS_ITEM_IDS)
    return [str(i + 1) for i in range(n_items)]


def extract_numeric_answers(elements: Any) -> Dict[str, float]:
    """Flatten one answer-block JSON into QID keys (and TE `_TEXT` aliases)."""
    out: Dict[str, float] = {}
    for _, question in iter_questions(elements):
        qid = question.get("QuestionID")
        qtype = question.get("QuestionType")
        answers = question.get("Answers") or {}
        if not qid or qtype == "DB" or not answers:
            continue

        if qtype == "Matrix":
            positions = answers.get("SelectedByPosition") or []
            if not isinstance(positions, list):
                positions = [positions]
            ids = _item_ids(question, len(question.get("Rows") or positions))
            for i, item_id in enumerate(ids):
                if i >= len(positions):
                    continue
                val = _to_float(positions[i])
                if val is not None:
                    out[f"{qid}_{item_id}"] = val

        elif qtype == "MC":
            pos = answers.get("SelectedByPosition")
            if isinstance(pos, list):
                pos = pos[0] if pos else None
            val = _to_float(pos)
            if val is not None:
                out[qid] = val

        elif qtype == "Slider":
            values = answers.get("Values") or []
            if not isinstance(values, list):
                values = [values]
            statements = question.get("Statements") or []
            meaningful_statements = [s for s in statements if s not in (None, "")]
            if len(values) > 1 or len(meaningful_statements) > 1:
                ids = _item_ids(question, max(len(values), len(statements)))
                for i, item_id in enumerate(ids):
                    if i >= len(values):
                        continue
                    val = _to_float(values[i])
                    if val is not None:
                        out[f"{qid}_{item_id}"] = val
            elif values:
                val = _to_float(values[0])
                if val is not None:
                    out[qid] = val
                    out[f"{qid}_1"] = val

        elif qtype == "TE":
            text = answers.get("Text")
            val = _to_float(text)
            if val is not None:
                out[qid] = val
                out[f"{qid}_TEXT"] = val

    return out


def load_importid_mapping(wave4_csv: Optional[Path] = None) -> Dict[str, str]:
    """Map ImportId / JSON keys to wave-4 column names.

    Combines the Qualtrics ImportId row with evaluation/column_mapping.csv
    (needed for base-rate sliders and pricing items).
    """
    path = wave4_csv or (WAVE_CSV_DIR / "wave_4_numbers_anonymized.csv")
    mapping: Dict[str, str] = {}
    with open(path, encoding="utf-8", newline="") as f:
        rows = [next(csv.reader(f)) for _ in range(3)]
    headers, import_row = rows[0], rows[2]
    for header, cell in zip(headers, import_row):
        cell = (cell or "").strip()
        if not (cell.startswith("{") and "ImportId" in cell):
            continue
        try:
            import_id = json.loads(cell).get("ImportId")
        except json.JSONDecodeError:
            continue
        if import_id:
            mapping[str(import_id)] = header

    manual_path = REPO_ROOT / "evaluation" / "column_mapping.csv"
    if manual_path.exists():
        manual = pd.read_csv(manual_path)
        for _, row in manual.iterrows():
            wave4_col = str(row["wave4_column_name"]).strip()
            input_col = str(row["input_column_name"]).strip()
            if wave4_col and input_col:
                mapping[input_col] = wave4_col
    return mapping


def answers_to_eval_columns(raw: Dict[str, float], import_map: Dict[str, str]) -> Dict[str, float]:
    """Translate JSON keys into the uppercased evaluation column names."""
    out: Dict[str, float] = {}
    for key, value in raw.items():
        col = import_map.get(key)
        if col is None:
            continue
        out[col.upper()] = value
    return out


def load_holdout_matrices() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """Build T1 and T2 DataFrames indexed by TWIN_ID, columns = eval names."""
    import_map = load_importid_mapping()
    scored = set(get_qid_to_task())
    t1_paths = sorted(ANSWER_BLOCK_DIR.glob("pid_*_wave4_Q_wave1_3_A.json"))
    t1_rows: List[Dict[str, Any]] = []
    t2_rows: List[Dict[str, Any]] = []
    for t1_path in t1_paths:
        pid = pid_from_filename(t1_path)
        t2_path = ANSWER_BLOCK_DIR / f"pid_{pid}_wave4_Q_wave4_A.json"
        t1_raw = extract_numeric_answers(load_json(t1_path))
        t2_raw = extract_numeric_answers(load_json(t2_path))
        t1_cols = answers_to_eval_columns(t1_raw, import_map)
        t2_cols = answers_to_eval_columns(t2_raw, import_map)
        t1_cols = {k: v for k, v in t1_cols.items() if k in scored}
        t2_cols = {k: v for k, v in t2_cols.items() if k in scored}
        t1_cols["TWIN_ID"] = pid
        t2_cols["TWIN_ID"] = pid
        t1_rows.append(t1_cols)
        t2_rows.append(t2_cols)
    t1 = pd.DataFrame(t1_rows).set_index("TWIN_ID").sort_index()
    t2 = pd.DataFrame(t2_rows).set_index("TWIN_ID").sort_index()
    return t1, t2, import_map


def assign_decile(value: Any, thresholds: np.ndarray) -> float:
    if pd.isna(value):
        return np.nan
    for i, thresh in enumerate(thresholds):
        if value <= thresh:
            return float(i + 1)
    return 10.0


def apply_anchoring_deciles(t1: pd.DataFrame, t2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Bin unbounded anchoring estimates using T1 (waves 1–3) percentiles.

    Matches the repository: cutpoints from T1, applied to both T1 and T2.
    The paper text says "wave 2"; T1 *is* the original administration of
    those wave-2 items.
    """
    t1 = t1.copy()
    t2 = t2.copy()
    for group in (ANCHOR_GROUP_A, ANCHOR_GROUP_B):
        existing = [c for c in group if c in t1.columns]
        if not existing:
            continue
        combined = pd.concat([t1[c] for c in existing], ignore_index=True).dropna()
        if combined.empty:
            continue
        thresholds = np.percentile(combined, np.arange(10, 100, 10))
        for frame in (t1, t2):
            for col in existing:
                if col in frame.columns:
                    frame[col] = frame[col].apply(lambda x, th=thresholds: assign_decile(x, th))
    return t1, t2


def item_accuracy(pred: float, truth: float, lo: float, hi: float) -> float:
    rng = hi - lo
    if rng <= 0:
        return float(pred == truth)
    return float(1.0 - abs(pred - truth) / rng)


def mean_and_ci(values: Iterable[float], confidence: float = 0.95) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return {"mean": np.nan, "se": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": 0}
    mean = float(arr.mean())
    if n == 1:
        return {"mean": mean, "se": 0.0, "ci_low": mean, "ci_high": mean, "n": 1}
    se = float(sem(arr))
    lo, hi = t_dist.interval(confidence, n - 1, loc=mean, scale=se)
    return {"mean": mean, "se": se, "ci_low": float(lo), "ci_high": float(hi), "n": n}


def compute_test_retest(
    t1: pd.DataFrame, t2: pd.DataFrame
) -> Dict[str, Any]:
    """Paper-style human test-retest: person-then-task, then unweighted tasks.

    For each person and task, average item-level accuracies. For each task,
    average people. Overall = unweighted mean of the 17 task means.
    Person-level accuracy = unweighted mean of that person's task accuracies.
    """
    ranges = get_column_ranges()
    qid_to_task = get_qid_to_task()
    common_index = t1.index.intersection(t2.index)
    t1 = t1.loc[common_index]
    t2 = t2.loc[common_index]
    cols = [c for c in t1.columns if c in t2.columns and c in ranges and c in qid_to_task]

    person_task: Dict[int, Dict[str, List[float]]] = {int(i): {} for i in common_index}
    item_records = []

    for col in cols:
        lo, hi = ranges[col]
        task = qid_to_task[col]
        mask = t1[col].notna() & t2[col].notna()
        for pid in t1.index[mask]:
            acc = item_accuracy(float(t1.at[pid, col]), float(t2.at[pid, col]), lo, hi)
            person_task[int(pid)].setdefault(task, []).append(acc)
            item_records.append({"TWIN_ID": int(pid), "column": col, "task": task, "accuracy": acc})

    person_task_acc = []
    for pid, tasks in person_task.items():
        for task, accs in tasks.items():
            person_task_acc.append(
                {"TWIN_ID": pid, "task": task, "accuracy": float(np.mean(accs)), "n_items": len(accs)}
            )
    person_task_df = pd.DataFrame(person_task_acc)

    task_rows = []
    for task, grp in person_task_df.groupby("task"):
        stats = mean_and_ci(grp["accuracy"])
        task_rows.append(
            {
                "task": task,
                "n_respondents": int(stats["n"]),
                "n_items_per_respondent_mean": float(grp["n_items"].mean()),
                "accuracy": stats["mean"],
                "se": stats["se"],
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
            }
        )
    task_df = pd.DataFrame(task_rows).sort_values("accuracy")

    overall_task_mean = float(task_df["accuracy"].mean()) if len(task_df) else np.nan
    # CI on the 17 task means (small-n); also person-level CI below.
    overall_from_tasks = mean_and_ci(task_df["accuracy"])

    person_rows = []
    for pid, grp in person_task_df.groupby("TWIN_ID"):
        person_rows.append(
            {
                "TWIN_ID": int(pid),
                "n_tasks": int(len(grp)),
                "accuracy": float(grp["accuracy"].mean()),
            }
        )
    person_df = pd.DataFrame(person_rows)
    person_stats = mean_and_ci(person_df["accuracy"]) if len(person_df) else mean_and_ci([])

    return {
        "task_df": task_df.reset_index(drop=True),
        "person_df": person_df,
        "person_task_df": person_task_df,
        "item_df": pd.DataFrame(item_records),
        "n_scored_columns": len(cols),
        "n_participants": int(len(common_index)),
        "overall_equal_task_mean": overall_task_mean,
        "overall_from_task_means": overall_from_tasks,
        "overall_from_persons": person_stats,
    }
