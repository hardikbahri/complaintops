"""Per-class bias (logit adjustment) tuning for the LoRA model, judged honestly.

Prediction becomes argmax_c( log p_c + b_c ). The 26 offsets b_c are tuned to maximise macro-F1.
Tuning many numbers on validation can overfit, so the honest score is cross-fitted: each validation
row is predicted with offsets tuned on OTHER rows. A second check tunes on the older half of the
validation split and tests on the newer half (a mini drift test). The test split is never scored.
Run from the project root:  python -m complaintops.tune_bias
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold

from complaintops.tracking import log_run

LORA = Path("data/lora_colab")
GRID = np.round(np.linspace(-2.0, 2.0, 17), 2)  # candidate offsets for each class


def macro_f1(y, pred, k):
    """Fast macro-F1: F1_c = 2*TP / (n_true + n_pred)."""
    tp = np.bincount(y[pred == y], minlength=k)
    denom = np.bincount(y, minlength=k) + np.bincount(pred, minlength=k)
    return float(np.mean(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)))


def fit_bias(logp, y, passes=3):
    """Coordinate ascent: tune one class offset at a time, keep any change that raises macro-F1."""
    k = logp.shape[1]
    b = np.zeros(k)
    best = macro_f1(y, (logp + b).argmax(1), k)
    for _ in range(passes):
        for c in range(k):
            keep = b[c]
            for v in GRID:
                b[c] = v
                s = macro_f1(y, (logp + b).argmax(1), k)
                if s > best + 1e-6:
                    best, keep = s, v
            b[c] = keep
    return b


def summary(y, pred):
    return (f1_score(y, pred, average="macro"), f1_score(y, pred, average="weighted"),
            accuracy_score(y, pred))


def main():
    classes = json.load(open(LORA / "classes.json"))
    k = len(classes)
    va = pd.read_parquet("data/processed/val.parquet")
    y = va["label"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    logp = np.log(np.clip(np.load(LORA / "proba_val.npy"), 1e-9, 1.0))
    base_pred = logp.argmax(1)
    base = summary(y, base_pred)
    print(f"LoRA, no bias:                macro {base[0]:.4f}  weighted {base[1]:.4f}  acc {base[2]:.4f}")

    # 1. random 5-fold cross-fit: every row is predicted with offsets tuned on the other 4/5
    oof = np.zeros(len(y), dtype=int)
    for tr, te in KFold(5, shuffle=True, random_state=42).split(logp):
        oof[te] = (logp[te] + fit_bias(logp[tr], y[tr])).argmax(1)
    cf = summary(y, oof)
    print(f"cross-fitted (5-fold):        macro {cf[0]:.4f}  weighted {cf[1]:.4f}  acc {cf[2]:.4f}")

    # 2. chronological check: tune on the older half, test on the newer half (val is sorted by date)
    h = len(y) // 2
    b_old = fit_bias(logp[:h], y[:h])
    new_base = summary(y[h:], base_pred[h:])
    new_tuned = summary(y[h:], (logp[h:] + b_old).argmax(1))
    print(f"newer half, no bias:          macro {new_base[0]:.4f}  weighted {new_base[1]:.4f}  acc {new_base[2]:.4f}")
    print(f"newer half, tuned on older:   macro {new_tuned[0]:.4f}  weighted {new_tuned[1]:.4f}  acc {new_tuned[2]:.4f}")

    # Final offsets: tuned on all of validation, saved for the evaluation gate (applied to test later).
    b = fit_bias(logp, y)
    Path("results").mkdir(exist_ok=True)
    json.dump(dict(classes=classes, bias=[float(x) for x in b]), open("results/lora_bias.json", "w"), indent=2)
    print("\nlargest offsets (class, b):")
    for i in np.argsort(-np.abs(b))[:8]:
        print(f"  {b[i]:+.2f}  {classes[i]}")

    log_run("lora_bias_tuned",
            {"model": "lora_bias_tuned", "n_classes": k, "grid": "-2..2 step 0.25", "passes": 3,
             "evaluation": "5-fold cross-fit on validation"},
            {"val_macro_f1_crossfit": cf[0], "val_weighted_f1_crossfit": cf[1], "val_accuracy_crossfit": cf[2],
             "val_macro_f1_untuned": base[0], "newer_half_macro_tuned_on_older": new_tuned[0],
             "newer_half_macro_untuned": new_base[0]},
            artifacts=["results/lora_bias.json"],
            tags={"stage": "experiment", "note": "test split not scored"})


if __name__ == "__main__":
    main()
