"""Data loading and schema helpers for Twin-2K-500 local files.

All paths are resolved relative to the repository root. Raw files under data/
are never modified.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
PERSONA_DIR = DATA_DIR / "mega_persona_json" / "mega_persona"
ANSWER_BLOCK_DIR = DATA_DIR / "mega_persona_json" / "answer_blocks"
SUMMARY_DIR = DATA_DIR / "mega_persona_summary_text"
WAVE_CSV_DIR = DATA_DIR / "wave_csv"

# Qualtrics export layout observed in all eight wave CSVs:
# row 0 = variable names, row 1 = question labels, row 2 = ImportId JSON,
# row 3+ = participant records.
QUALTRICS_SKIPROWS = [1, 2]

PID_RE = re.compile(r"pid_(\d+)")

# Wave-1 Qualtrics column names for the 14 demographic items (QID11–QID24).
DEMOGRAPHIC_COLUMNS = {
    "region": "Q11",
    "sex": "Q12",
    "age": "Q13",
    "education": "Q14",
    "race": "Q15",
    "us_citizen": "Q16",
    "marital_status": "Q17",
    "religion": "Q18",
    "religious_attendance": "Q19",
    "political_party": "Q20",
    "income": "Q21",
    "political_ideology": "Q22",
    "household_size": "Q23",
    "employment": "Q24",
}


def load_json(path: Path) -> Any:
    """Load a JSON file, including files that store JSON as a quoted string."""
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, str):
        obj = json.loads(obj)
    return obj


def pid_from_filename(path: Path) -> Optional[int]:
    match = PID_RE.search(path.name)
    return int(match.group(1)) if match else None


def iter_elements(elements: Any) -> Iterator[dict]:
    """Yield nested Qualtrics-style Block/Branch dicts."""
    if not isinstance(elements, list):
        if isinstance(elements, dict):
            elements = elements.get("Elements", [elements])
        else:
            return
    for element in elements:
        if not isinstance(element, dict):
            continue
        yield element
        nested = element.get("Elements")
        if isinstance(nested, list):
            yield from iter_elements(nested)


def iter_questions(elements: Any) -> Iterator[Tuple[Optional[str], dict]]:
    """Yield (block_name, question_dict) pairs, including nested branches."""
    for element in iter_elements(elements):
        questions = element.get("Questions")
        if not isinstance(questions, list):
            continue
        block_name = element.get("BlockName") or element.get("Description")
        for question in questions:
            if isinstance(question, dict):
                yield block_name, question


def count_local_files() -> pd.DataFrame:
    persona = sorted(PERSONA_DIR.glob("pid_*_mega_persona.json"))
    t1 = sorted(ANSWER_BLOCK_DIR.glob("pid_*_wave4_Q_wave1_3_A.json"))
    t2 = sorted(ANSWER_BLOCK_DIR.glob("pid_*_wave4_Q_wave4_A.json"))
    summaries = sorted(SUMMARY_DIR.glob("pid_*_mega_persona.txt"))
    csvs = sorted(WAVE_CSV_DIR.glob("*.csv"))
    rows = [
        {
            "representation": "wave1_3_persona_json (persona)",
            "participant_count": len(persona),
            "role": "Waves 1–3 non-holdout responses; model/persona input",
            "local_location": str(PERSONA_DIR.relative_to(REPO_ROOT)),
        },
        {
            "representation": "wave4_Q_wave1_3_A (T1)",
            "participant_count": len(t1),
            "role": "Hold-out tasks with original (waves 1–3) answers",
            "local_location": str(ANSWER_BLOCK_DIR.relative_to(REPO_ROOT)),
        },
        {
            "representation": "wave4_Q_wave4_A (T2)",
            "participant_count": len(t2),
            "role": "Same hold-out tasks with Wave 4 repeated answers",
            "local_location": str(ANSWER_BLOCK_DIR.relative_to(REPO_ROOT)),
        },
        {
            "representation": "persona_summary (full_persona)",
            "participant_count": len(summaries),
            "role": "Prose summary of the person; optional persona encoding",
            "local_location": str(SUMMARY_DIR.relative_to(REPO_ROOT)),
        },
        {
            "representation": "raw Qualtrics wave CSVs",
            "participant_count": "see per-file row counts",
            "role": "Original survey exports (labels + numeric codes)",
            "local_location": str(WAVE_CSV_DIR.relative_to(REPO_ROOT)),
        },
    ]
    return pd.DataFrame(rows)


def collect_pid_sets() -> Dict[str, set]:
    return {
        "persona": {pid_from_filename(p) for p in PERSONA_DIR.glob("pid_*_mega_persona.json")},
        "t1": {pid_from_filename(p) for p in ANSWER_BLOCK_DIR.glob("pid_*_wave4_Q_wave1_3_A.json")},
        "t2": {pid_from_filename(p) for p in ANSWER_BLOCK_DIR.glob("pid_*_wave4_Q_wave4_A.json")},
        "summary": {pid_from_filename(p) for p in SUMMARY_DIR.glob("pid_*_mega_persona.txt")},
    }


def load_wave_csv(path: Path, *, dtype: Optional[dict] = None) -> pd.DataFrame:
    """Load a Qualtrics anonymized CSV, skipping label and ImportId rows."""
    df = pd.read_csv(path, skiprows=QUALTRICS_SKIPROWS, low_memory=False, dtype=dtype)
    if "TWIN_ID" in df.columns:
        df["TWIN_ID"] = pd.to_numeric(df["TWIN_ID"], errors="coerce").astype("Int64")
    return df


def load_wave(wave: int, kind: str = "labels") -> pd.DataFrame:
    if kind not in {"labels", "numbers"}:
        raise ValueError("kind must be 'labels' or 'numbers'")
    path = WAVE_CSV_DIR / f"wave_{wave}_{kind}_anonymized.csv"
    return load_wave_csv(path)


def inspect_csv_header(path: Path, n_preview: int = 4) -> Dict[str, Any]:
    raw = pd.read_csv(path, header=None, nrows=n_preview, dtype=str, low_memory=False)
    n_import = [
        int(sum(str(x).startswith("{") and "ImportId" in str(x) for x in raw.iloc[i]))
        for i in range(len(raw))
    ]
    return {
        "file": path.name,
        "n_columns": int(raw.shape[1]),
        "row0_first": list(raw.iloc[0, :6]),
        "row1_first": [str(x)[:50] for x in raw.iloc[1, :4]] if len(raw) > 1 else [],
        "importid_counts": n_import,
    }


def load_demographics() -> pd.DataFrame:
    """Participant demographics from Wave 1 labels (one row per completer)."""
    df = load_wave(1, "labels")
    keep = ["TWIN_ID"] + list(DEMOGRAPHIC_COLUMNS.values())
    out = df[keep].copy()
    out = out.rename(columns={v: k for k, v in DEMOGRAPHIC_COLUMNS.items()})
    return out


def question_type_counts(elements: Any) -> Counter:
    counts: Counter = Counter()
    for _, question in iter_questions(elements):
        counts[question.get("QuestionType") or "MISSING"] += 1
    return counts


def item_level_response_count(question: dict) -> int:
    """Count scored/item-level responses inside one JSON question object.

    Matrix/slider batteries contain many items in a single object. DB
    (descriptive) objects contribute zero responses.
    """
    qtype = question.get("QuestionType")
    if qtype == "DB":
        return 0
    answers = question.get("Answers") or {}
    if qtype == "Matrix":
        rows = question.get("Rows") or question.get("Statements") or []
        return len(rows) if rows else 1
    if qtype == "Slider":
        values = answers.get("Values") or []
        statements = question.get("Statements") or question.get("Rows") or []
        if len(values) > 1:
            return len(values)
        if statements and statements != [""]:
            return max(len(statements), 1)
        return 1
    return 1


def summarize_persona_files(n_files: Optional[int] = None) -> pd.DataFrame:
    paths = sorted(PERSONA_DIR.glob("pid_*_mega_persona.json"))
    if n_files is not None:
        paths = paths[:n_files]
    rows = []
    for path in paths:
        data = load_json(path)
        questions = list(iter_questions(data))
        n_blocks = sum(
            1
            for el in iter_elements(data)
            if el.get("ElementType") == "Block" or "Questions" in el
        )
        type_counts = Counter(q.get("QuestionType") for _, q in questions)
        n_items = sum(item_level_response_count(q) for _, q in questions)
        rows.append(
            {
                "pid": pid_from_filename(path),
                "n_blocks": n_blocks,
                "n_json_question_objects": len(questions),
                "n_item_level_responses": n_items,
                "n_non_db_objects": sum(q.get("QuestionType") != "DB" for _, q in questions),
                **{f"type_{k}": v for k, v in type_counts.items()},
            }
        )
    return pd.DataFrame(rows)


def summarize_answer_block_files(kind: str = "t1", n_files: Optional[int] = None) -> pd.DataFrame:
    pattern = (
        "pid_*_wave4_Q_wave1_3_A.json" if kind == "t1" else "pid_*_wave4_Q_wave4_A.json"
    )
    paths = sorted(ANSWER_BLOCK_DIR.glob(pattern))
    if n_files is not None:
        paths = paths[:n_files]
    rows = []
    for path in paths:
        data = load_json(path)
        questions = list(iter_questions(data))
        type_counts = Counter(q.get("QuestionType") for _, q in questions)
        block_names = [
            el.get("BlockName")
            for el in iter_elements(data)
            if el.get("ElementType") == "Block" or el.get("BlockName")
        ]
        rows.append(
            {
                "pid": pid_from_filename(path),
                "n_blocks": len(block_names),
                "n_json_question_objects": len(questions),
                "n_item_level_responses": sum(item_level_response_count(q) for _, q in questions),
                **{f"type_{k}": v for k, v in type_counts.items()},
            }
        )
    return pd.DataFrame(rows)


def question_fingerprint(question: dict) -> Tuple:
    """Structural identity of a question, excluding answers."""
    qtype = question.get("QuestionType")
    return (
        question.get("QuestionID"),
        qtype,
        tuple(question.get("Rows") or []),
        tuple(question.get("Columns") or []),
        tuple(question.get("Options") or []),
        tuple(question.get("Statements") or []),
        tuple(question.get("RowsID") or []),
        tuple(question.get("StatementsID") or []),
    )


def compare_t1_t2_structure(pid: int) -> Dict[str, Any]:
    t1 = load_json(ANSWER_BLOCK_DIR / f"pid_{pid}_wave4_Q_wave1_3_A.json")
    t2 = load_json(ANSWER_BLOCK_DIR / f"pid_{pid}_wave4_Q_wave4_A.json")
    fp1 = [question_fingerprint(q) for _, q in iter_questions(t1)]
    fp2 = [question_fingerprint(q) for _, q in iter_questions(t2)]
    return {
        "pid": pid,
        "n_t1": len(fp1),
        "n_t2": len(fp2),
        "same_length": len(fp1) == len(fp2),
        "same_order": fp1 == fp2,
        "n_mismatch": sum(a != b for a, b in zip(fp1, fp2)) if len(fp1) == len(fp2) else None,
    }


def compare_all_t1_t2_structure() -> pd.DataFrame:
    pids = sorted(pid_from_filename(p) for p in ANSWER_BLOCK_DIR.glob("pid_*_wave4_Q_wave1_3_A.json"))
    return pd.DataFrame([compare_t1_t2_structure(pid) for pid in pids])


def find_example_question(
    path: Path, question_type: str, *, require_answers: bool = True
) -> Optional[dict]:
    data = load_json(path)
    for _, question in iter_questions(data):
        if question.get("QuestionType") != question_type:
            continue
        if require_answers and not question.get("Answers"):
            continue
        return question
    return None


def sanitize_question_example(question: dict, *, max_text: int = 240) -> dict:
    """Small, display-safe subset of a question dict."""
    if not question:
        return {"error": "no example found"}
    answers = question.get("Answers") or {}
    text = str(question.get("QuestionText") or "")
    example = {
        "QuestionID": question.get("QuestionID"),
        "QuestionType": question.get("QuestionType"),
        "QuestionText": text[:max_text] + ("…" if len(text) > max_text else ""),
        "Settings.Selector": (question.get("Settings") or {}).get("Selector"),
        "n_Options": len(question.get("Options") or []),
        "n_Rows": len(question.get("Rows") or []),
        "n_Columns": len(question.get("Columns") or []),
        "n_Statements": len(question.get("Statements") or []),
        "AnswerKeys": list(answers.keys()),
        "AnswersPreview": answers,
    }
    return example
