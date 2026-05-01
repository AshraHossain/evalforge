"""
models/training/evaluate.py — Compare LoRA judge vs baselines on the test split.

Baselines:
  1. All-ones  — predict every label = 1 (optimistic baseline)
  2. Base model — DeBERTa-v3-base with no fine-tuning (random classifier head)
  3. LoRA judge — fine-tuned adapter from models/artifacts/lora_judge/

Run:
    python -m models.training.evaluate
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_MODEL = "microsoft/deberta-v3-base"
ARTIFACTS_DIR = Path("models/artifacts/lora_judge")
DATA_DIR = Path("models/training/data")
LABEL_NAMES = ["factually_consistent", "relevant", "complete", "safe"]
MAX_LENGTH = 512
BATCH_SIZE = 32


def load_test_data() -> tuple[list[str], np.ndarray]:
    path = DATA_DIR / "test.jsonl"
    if not path.exists():
        raise FileNotFoundError("Run python -m models.training.dataset first.")
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    texts = [r["text"] for r in rows]
    labels = np.array([r["labels"] for r in rows], dtype=int)
    return texts, labels


def run_inference(model, tokenizer, texts: list[str], device: torch.device) -> np.ndarray:
    model.eval()
    all_probs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        enc = tokenizer(
            batch,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
    return np.vstack(all_probs)


def probs_to_preds(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (probs >= threshold).astype(int)


def print_results(name: str, preds: np.ndarray, labels: np.ndarray) -> dict:
    f1_micro = f1_score(labels, preds, average="micro", zero_division=0)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    per_label = f1_score(labels, preds, average=None, zero_division=0)
    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")
    for lname, f1 in zip(LABEL_NAMES, per_label):
        bar = "█" * int(f1 * 20)
        print(f"  {lname:<25} F1={f1:.3f}  {bar}")
    print(f"  {'micro F1':<25} F1={f1_micro:.3f}")
    print(f"  {'macro F1':<25} F1={f1_macro:.3f}")
    return {"f1_micro": f1_micro, "f1_macro": f1_macro, "f1_per_label": per_label.tolist()}


def main() -> None:
    texts, labels = load_test_data()
    logger.info(f"Test set: {len(texts)} samples")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    results = {}

    # ── Baseline 1: all-ones ──────────────────────────────────────────────
    all_ones = np.ones_like(labels)
    results["All-ones baseline"] = print_results("All-ones baseline", all_ones, labels)

    # ── Baseline 2: base DeBERTa (no fine-tuning) ────────────────────────
    logger.info("Loading base model …")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=4,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    ).to(device)
    base_probs = run_inference(base_model, tokenizer, texts, device)
    results["Base DeBERTa (no fine-tune)"] = print_results(
        "Base DeBERTa (no fine-tune)", probs_to_preds(base_probs), labels
    )
    del base_model

    # ── LoRA fine-tuned ───────────────────────────────────────────────────
    if not ARTIFACTS_DIR.exists():
        logger.warning(f"Artifacts not found at {ARTIFACTS_DIR}. Run train.py first.")
    else:
        logger.info("Loading LoRA adapter …")
        base_for_lora = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL,
            num_labels=4,
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True,
        )
        lora_model = PeftModel.from_pretrained(base_for_lora, str(ARTIFACTS_DIR)).to(device)
        lora_probs = run_inference(lora_model, tokenizer, texts, device)
        results["LoRA fine-tuned"] = print_results(
            "LoRA fine-tuned", probs_to_preds(lora_probs), labels
        )

        # Summary comparison
        print(f"\n{'═'*55}")
        print("  SUMMARY — micro F1")
        print(f"{'═'*55}")
        for name, r in results.items():
            delta = ""
            if name == "LoRA fine-tuned":
                base_f1 = results.get("Base DeBERTa (no fine-tune)", {}).get("f1_micro", 0)
                diff = r["f1_micro"] - base_f1
                delta = f"  (+{diff:+.3f} vs base)"
            print(f"  {name:<30} {r['f1_micro']:.3f}{delta}")
        print()


if __name__ == "__main__":
    main()
