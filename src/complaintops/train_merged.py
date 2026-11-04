"""Experiment: TF-IDF + logistic regression on merged labels (26 -> 19 classes).

Also scores the ORIGINAL 26-class baseline on the merged task (merge its predictions
and truths), which is the fair comparison: same validation rows, same task.
Run:  python -m complaintops.train_merged
"""
import time

import joblib
import pandas as pd
from sklearn.metrics import classification_report, f1_score

from complaintops.data import merge_labels
from complaintops.train_baseline import load, make_pipeline


def main():
    X_tr, y_tr = load("train")
    X_va, y_va = load("val")
    y_tr_m, y_va_m = merge_labels(y_tr), merge_labels(y_va)
    print(f"classes: {y_tr.nunique()} -> {y_tr_m.nunique()}", flush=True)

    # Fair comparison: old 26-class baseline predictions, merged afterwards.
    old = pd.read_parquet("data/baseline_val_preds.parquet")
    old_f1 = f1_score(merge_labels(old["y"]), merge_labels(old["pred"]), average="macro")
    print(f"old baseline scored on merged task: macro-F1 {old_f1:.4f}", flush=True)

    pipe = make_pipeline()
    t0 = time.time()
    pipe.fit(X_tr, y_tr_m)
    train_time = time.time() - t0
    pred = pipe.predict(X_va)
    f1 = f1_score(y_va_m, pred, average="macro")
    print(f"train time {train_time:.0f}s  merged-label val macro-F1 {f1:.4f}")
    print(classification_report(y_va_m, pred, zero_division=0))

    joblib.dump(pipe, "data/baseline_merged.joblib")
    pd.DataFrame({"y": y_va_m, "pred": pred, "text": X_va}).to_parquet(
        "data/baseline_merged_val_preds.parquet")


if __name__ == "__main__":
    main()
