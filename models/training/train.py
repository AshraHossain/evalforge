"""
models/training/train.py — Fine-tune DeBERTa-v3-base with LoRA for 4-label scoring.

Labels: [factually_consistent, relevant, complete, safe]
Method: PEFT LoRA on query_proj + value_proj (< 1% of total params)
Output: models/artifacts/lora_judge/  (adapter weights + tokenizer)

Prerequisites:
    pip install -e ".[ml]"
    python -m models.training.dataset   # build train/val/test splits first

Run:
    python -m models.training.train
"""

import json
import logging
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_MODEL = "microsoft/deberta-v3-base"
ARTIFACTS_DIR = Path("models/artifacts/lora_judge")
DATA_DIR = Path("models/training/data")
NUM_LABELS = 4
MAX_LENGTH = 512


def load_jsonl(path: Path) -> Dataset:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return Dataset.from_list(rows)


def make_tokenize_fn(tokenizer):
    def tokenize(batch):
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )
        enc["labels"] = [list(map(float, lbl)) for lbl in batch["labels"]]
        return enc
    return tokenize


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= 0.5).astype(int)
    labels_int = labels.astype(int)
    return {
        "f1_micro": f1_score(labels_int, preds, average="micro", zero_division=0),
        "f1_macro": f1_score(labels_int, preds, average="macro", zero_division=0),
        "f1_per_label": f1_score(labels_int, preds, average=None, zero_division=0).tolist(),
    }


def main() -> None:
    if not (DATA_DIR / "train.jsonl").exists():
        raise FileNotFoundError(
            "Training data not found. Run: python -m models.training.dataset"
        )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenize_fn = make_tokenize_fn(tokenizer)

    logger.info("Tokenizing datasets …")
    train_ds = load_jsonl(DATA_DIR / "train.jsonl").map(tokenize_fn, batched=True, remove_columns=["text"])
    val_ds = load_jsonl(DATA_DIR / "val.jsonl").map(tokenize_fn, batched=True, remove_columns=["text"])

    logger.info(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    base = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    )

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query_proj", "value_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()

    use_fp16 = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=str(ARTIFACTS_DIR),
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        greater_is_better=True,
        logging_steps=50,
        fp16=use_fp16,
        report_to="none",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("Starting training …")
    trainer.train()

    logger.info(f"Saving adapter weights → {ARTIFACTS_DIR}")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ARTIFACTS_DIR))
    tokenizer.save_pretrained(str(ARTIFACTS_DIR))

    # Final val metrics
    metrics = trainer.evaluate()
    logger.info("Val metrics:")
    label_names = ["factually_consistent", "relevant", "complete", "safe"]
    for name, f1 in zip(label_names, metrics.get("eval_f1_per_label", [])):
        logger.info(f"  {name:<25} F1={f1:.3f}")
    logger.info(f"  {'micro F1':<25} F1={metrics.get('eval_f1_micro', 0):.3f}")
    logger.info(f"  {'macro F1':<25} F1={metrics.get('eval_f1_macro', 0):.3f}")
    logger.info("\nDone. Set LORA_MODEL_AVAILABLE=true in .env to activate.")


if __name__ == "__main__":
    main()
