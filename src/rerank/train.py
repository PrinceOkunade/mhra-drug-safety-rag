"""Fine-tune the cross-encoder reranker on synthetic corpus pairs (Exp 5 Step 2).

Full fine-tune of `cross-encoder/ms-marco-MiniLM-L-6-v2` (the Step-1 off-the-shelf
control) on the LLM-synthesised (question, passage) pairs from gen_pairs.py. The
point of the experiment: measure whether task-specific fine-tuning lifts ranking
OVER the generic off-the-shelf reranker - the delta is what the fine-tuning bought.

Discipline (PHASE3_SPEC): the gold set is already held out (gen_pairs excluded
every gold-passage chunk). Here we further split the synthetic pairs into
train/val, log a val metric curve (CEBinaryClassificationEvaluator) to check
learning + overfitting, and report the full training setup.

Saves the fine-tuned model to models/reranker-ft/ (point config.retriever.
reranker_model at it and re-run the eval to measure the lift).
"""
from __future__ import annotations

import argparse
import json
import random

from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator

from src.config import EVAL_DIR, ROOT

PAIRS_PATH = EVAL_DIR / "reranker_train_pairs.jsonl"
BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OUT_DIR = ROOT / "models" / "reranker-ft"


def _load_pairs() -> list[dict]:
    with open(PAIRS_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _examples(rows: list[dict]) -> list[InputExample]:
    return [InputExample(texts=[r["question"], r["passage"]], label=float(r["label"])) for r in rows]


def _score(evaluator, model) -> float:
    """v5 evaluators return a dict of metrics - pull out average_precision."""
    r = evaluator(model)
    if isinstance(r, dict):
        ap = next((v for k, v in r.items() if k.endswith("average_precision")), None)
        return float(ap if ap is not None else next(iter(r.values()), 0.0))
    return float(r)


def main(epochs: int, batch_size: int, lr: float, val_frac: float) -> None:
    rows = _load_pairs()
    random.seed(42)
    random.shuffle(rows)
    n_val = max(1, int(len(rows) * val_frac))
    val_rows, train_rows = rows[:n_val], rows[n_val:]

    train_ex = _examples(train_rows)
    val_ex = _examples(val_rows)
    train_dl = DataLoader(train_ex, shuffle=True, batch_size=batch_size)
    val_eval = CEBinaryClassificationEvaluator.from_input_examples(val_ex, name="val")
    # a held-in TRAIN sample, same size as val, to spot overfitting (train vs val gap)
    train_probe = CEBinaryClassificationEvaluator.from_input_examples(
        _examples(train_rows[:n_val]), name="train_probe"
    )

    n_pos = sum(1 for r in train_rows if r["label"] == 1)
    print(
        f"base={BASE_MODEL}\n"
        f"train={len(train_rows)} ({n_pos} pos / {len(train_rows)-n_pos} neg)  val={len(val_rows)}\n"
        f"epochs={epochs} batch_size={batch_size} lr={lr}  loss=BCEWithLogitsLoss (num_labels=1)"
    )

    model = CrossEncoder(BASE_MODEL, num_labels=1)
    print("val AP BEFORE fine-tuning:", round(_score(val_eval, model), 4))

    warmup = int(len(train_dl) * epochs * 0.1)
    model.fit(
        train_dataloader=train_dl,
        evaluator=val_eval,
        epochs=epochs,
        warmup_steps=warmup,
        optimizer_params={"lr": lr},
        evaluation_steps=max(1, len(train_dl) // 2),
        output_path=str(OUT_DIR),
        save_best_model=True,
    )

    val_after = round(_score(val_eval, model), 4)
    train_after = round(_score(train_probe, model), 4)
    print(f"val AP AFTER fine-tuning:  {val_after}")
    print(f"train-probe AP AFTER:      {train_after}  (gap {round(train_after-val_after,4)} - large gap = overfit)")
    model.save(str(OUT_DIR))
    print(f"saved fine-tuned reranker -> {OUT_DIR}")
    print(f"curve CSV -> {OUT_DIR / 'eval' / 'CEBinaryClassificationEvaluator_val_results.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()
    main(args.epochs, args.batch_size, args.lr, args.val_frac)
