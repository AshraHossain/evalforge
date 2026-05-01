"""
models/training/dataset.py — Build training data for the LoRA judge.

Pulls from two public datasets:
  - TruthfulQA (generation split): correct vs. incorrect answers → factuality labels
  - HaluEval (qa_samples): hallucinated vs. right answers → factuality labels

Output: models/training/data/{train,val,test}.jsonl
Each line: {"text": "Question: ...\nAnswer: ...", "labels": [fc, rel, comp, safe]}
Labels are floats in {0.0, 1.0} for multi-label BCE training.

Run:
    python -m models.training.dataset
"""

import json
import logging
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
LABEL_NAMES = ["factually_consistent", "relevant", "complete", "safe"]


def _row(question: str, answer: str, labels: list[float]) -> dict:
    return {
        "text": f"Question: {question.strip()}\nAnswer: {answer.strip()}",
        "labels": labels,
    }


def build_truthfulqa() -> list[dict]:
    from datasets import load_dataset

    logger.info("Loading TruthfulQA …")
    ds = load_dataset("truthful_qa", "generation", split="validation")
    rows = []
    for item in ds:
        q = item["question"]
        for ans in item["correct_answers"][:2]:
            if ans.strip():
                rows.append(_row(q, ans, [1.0, 1.0, 1.0, 1.0]))
        for ans in item["incorrect_answers"][:2]:
            if ans.strip():
                # incorrect: factually wrong, relevant (on-topic), incomplete
                rows.append(_row(q, ans, [0.0, 1.0, 0.0, 1.0]))
    logger.info(f"  TruthfulQA: {len(rows)} rows")
    return rows


def build_halueval() -> list[dict]:
    from datasets import load_dataset

    logger.info("Loading HaluEval QA …")
    try:
        ds = load_dataset("pminervini/HaluEval", "qa_samples", split="data")
    except Exception:
        # Fallback: some HuggingFace mirrors use a different split name
        ds = load_dataset("pminervini/HaluEval", "qa_samples", split="train")

    rows = []
    for item in ds:
        q = item.get("question", "")
        right = item.get("right_answer", "")
        hall = item.get("hallucinated_answer", "")
        if q and right:
            rows.append(_row(q, right, [1.0, 1.0, 1.0, 1.0]))
        if q and hall:
            # hallucinated: factually wrong, relevant, complete-looking, safe
            rows.append(_row(q, hall, [0.0, 1.0, 1.0, 1.0]))
    logger.info(f"  HaluEval: {len(rows)} rows")
    return rows


def save_splits(rows: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    random.shuffle(rows)
    n = len(rows)
    splits = {
        "train": rows[: int(n * 0.80)],
        "val": rows[int(n * 0.80) : int(n * 0.90)],
        "test": rows[int(n * 0.90) :],
    }
    for name, data in splits.items():
        path = DATA_DIR / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row) + "\n")
        logger.info(f"  Wrote {len(data):>5} rows → {path}")


def main() -> None:
    rows = build_truthfulqa() + build_halueval()
    logger.info(f"Total rows before split: {len(rows)}")
    save_splits(rows)
    logger.info("Done.")


if __name__ == "__main__":
    main()
