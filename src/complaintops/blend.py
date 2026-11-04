"""Blend LoRA DistilBERT (Colab run) and TF-IDF + logistic regression by averaging probabilities.

p_blend = w * p_lr + (1 - w) * p_lora. The one number w is chosen on the VALIDATION split.
Test probabilities are computed and saved for the later evaluation gate, but NOT scored here,
so the test split is only looked at once, at the end.
Run from the project root:  python -m complaintops.blend
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from complaintops.tracking import log_run

LORA = Path("data/lora_colab")


def lr_proba(pipe, split):
    """Baseline probabilities for a split; computed once, then cached on disk."""
    path = Path(f"data/proba_lr_{split}.npy")
    if path.exists():
        return np.load(path)
    df = pd.read_parquet(f"data/processed/{split}.parquet")
    proba = pipe.predict_proba(df["narrative"]).astype("float32")
    np.save(path, proba)
    return proba


def scores(y, proba):
    pred = proba.argmax(1)
    return (f1_score(y, pred, average="macro"), f1_score(y, pred, average="weighted"),
            accuracy_score(y, pred))


def main():
    classes = json.load(open(LORA / "classes.json"))
    pipe = joblib.load("data/baseline.joblib")
    assert list(pipe.classes_) == classes, "class order differs between the two models"

    va = pd.read_parquet("data/processed/val.parquet")
    y = va["label"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    p_lr, p_lora = lr_proba(pipe, "val"), np.load(LORA / "proba_val.npy")

    m_lr, w_lr, _ = scores(y, p_lr)
    m_lo, w_lo, _ = scores(y, p_lora)
    print(f"LR    macro {m_lr:.4f}  weighted {w_lr:.4f}   (expected 0.4717 / 0.6047)")
    print(f"LoRA  macro {m_lo:.4f}  weighted {w_lo:.4f}   (expected 0.5241 / 0.6433)")
    assert abs(m_lr - 0.4717) < 0.002 and abs(m_lo - 0.5241) < 0.002, "inputs do not reproduce"

    grid = np.round(np.linspace(0, 1, 21), 2)
    rows = [(w, *scores(y, w * p_lr + (1 - w) * p_lora)) for w in grid]
    print("\n  w_lr   macro-F1  weighted-F1  accuracy")
    for w, m, wf, a in rows:
        print(f"  {w:.2f}   {m:.4f}    {wf:.4f}       {a:.4f}")
    best = max(rows, key=lambda r: r[1])  # chosen by macro-F1
    w, m, wf, a = best
    print(f"\nbest w_lr = {w}: macro {m:.4f}  weighted {wf:.4f}  accuracy {a:.4f}  "
          f"(LoRA alone: {m_lo:.4f} / {w_lo:.4f})", flush=True)

    lr_proba(pipe, "test")  # cache for the gate; not scored here
    log_run("blend_lora_lr",
            {"model": "blend_lora_lr", "w_lr": w, "w_grid_step": 0.05, "n_classes": 26,
             "selected_by": "val_macro_f1"},
            {"val_macro_f1": m, "val_weighted_f1": wf, "val_accuracy": a,
             "val_macro_f1_lora": m_lo, "val_macro_f1_lr": m_lr},
            tags={"stage": "experiment", "note": "w chosen on validation; test not scored"})


if __name__ == "__main__":
    main()
