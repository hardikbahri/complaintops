"""TF-IDF + logistic regression predicting only the PRODUCT (easier, better-posed task).

Target = the `product` column already in the processed splits. The 9 most common products in
TRAIN are kept; the rest (retired or tiny products, and products that never appear in train)
become "Other". Scored on the validation split; logged to MLflow.
Run from the project root:  python -m complaintops.train_product
"""
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from complaintops.tracking import log_run
from complaintops.train_baseline import make_pipeline

N_PRODUCTS = 9


def main():
    tr = pd.read_parquet("data/processed/train.parquet")
    va = pd.read_parquet("data/processed/val.parquet")
    keep = set(tr["product"].value_counts().head(N_PRODUCTS).index)  # chosen from train only
    y_tr = tr["product"].where(tr["product"].isin(keep), "Other")
    y_va = va["product"].where(va["product"].isin(keep), "Other")
    print(f"classes: {y_tr.nunique()}  (train: {sorted(y_tr.unique())})", flush=True)

    pipe = make_pipeline()
    t0 = time.time()
    pipe.fit(tr["narrative"], y_tr)
    train_time = time.time() - t0
    pred = pipe.predict(va["narrative"])
    f1, acc = f1_score(y_va, pred, average="macro"), accuracy_score(y_va, pred)
    wf1 = f1_score(y_va, pred, average="weighted")
    print(f"train time {train_time:.0f}s  product-level val macro-F1 {f1:.4f}  accuracy {acc:.4f}")
    report = classification_report(y_va, pred, zero_division=0)
    print(report)

    Path("results").mkdir(exist_ok=True)
    Path("results/product_report.txt").write_text(report)
    log_run("product_tfidf_lr",
            {"model": "tfidf_logreg", "task": "product", "n_classes": y_tr.nunique(),
             "class_weight": "balanced", "ngram_range": "1-2", "min_df": 3,
             "n_train": len(tr)},
            {"val_weighted_f1": wf1, "val_macro_f1": f1, "val_accuracy": acc,
             "train_time_s": train_time},
            artifacts=["results/product_report.txt"], model=pipe,
            tags={"stage": "experiment", "note": "product-only target; easier task, not comparable with 26-class"})


if __name__ == "__main__":
    main()
