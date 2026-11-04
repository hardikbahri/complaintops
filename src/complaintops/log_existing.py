"""Back-fill MLflow with experiments that were already run (no retraining). Step 5.

Run once from the project root:  python -m complaintops.log_existing
Every run is tagged logged_after_the_fact=true: the numbers come from saved files,
not from a training script that logged them live.
"""
import ast
import sys

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from complaintops.tracking import EXPERIMENT, TRACKING_URI, log_run
from complaintops.train_xgb import CHOSEN

LATE = {"logged_after_the_fact": "true"}
LR_PARAMS = dict(model="tfidf_logreg", ngram_range="1-2", min_df=3, sublinear_tf=True,
                 class_weight="balanced")


def val_scores(preds_path):
    p = pd.read_parquet(preds_path)
    return {"val_macro_f1": f1_score(p["y"], p["pred"], average="macro"),
            "val_accuracy": accuracy_score(p["y"], p["pred"])}


def already_has_runs():
    mlflow.set_tracking_uri(TRACKING_URI)
    exp = mlflow.get_experiment_by_name(EXPERIMENT)
    return exp is not None and len(mlflow.search_runs([exp.experiment_id])) > 0


def main():
    if already_has_runs() and "--force" not in sys.argv:
        sys.exit("The experiment already has runs; refusing to add duplicates (use --force).")

    # 1. Baseline, 26 classes
    log_run("baseline_tfidf_lr",
            {**LR_PARAMS, "n_classes": 26, "n_train": 234_849},
            {**val_scores("data/baseline_val_preds.parquet"), "train_time_s": 945},
            artifacts=["data/baseline_val_preds.parquet", "data/confusion_baseline.png"],
            model=joblib.load("data/baseline.joblib"), tags={**LATE, "stage": "baseline"})

    # 2. 5-fold CV of the baseline (50k-row subsample); numbers copied from the README
    log_run("baseline_tfidf_lr_cv5",
            {**LR_PARAMS, "n_classes": 26, "n_train": 50_000, "cv_folds": 5},
            {"cv_macro_f1_mean": 0.4635, "cv_macro_f1_std": 0.0044},
            tags={**LATE, "stage": "baseline"})

    # 3. Merged-label experiment, 19 classes
    log_run("merged_labels_tfidf_lr",
            {**LR_PARAMS, "n_classes": 19, "n_train": 234_849},
            {**val_scores("data/baseline_merged_val_preds.parquet"), "train_time_s": 981},
            artifacts=["data/baseline_merged_val_preds.parquet"],
            model=joblib.load("data/baseline_merged.joblib"),
            tags={**LATE, "stage": "experiment", "note": "labels merged 26->19; easier task"})

    # 4. Final XGBoost (trained on Colab T4)
    log_run("xgboost_final",
            {"model": "xgboost", **CHOSEN, "n_estimators": 800, "max_tfidf": 20_000,
             "n_classes": 26, "n_train": 211_364, "device": "cuda"},
            {"val_macro_f1": 0.4543, "val_accuracy": 0.59, "train_time_s": 1994,
             "best_iteration": 799},
            artifacts=["results/xgb_report.txt", "results/xgb_model.json"],
            tags={**LATE, "stage": "final"})

    # 5. Every finished Optuna trial (tuned on a 36k-row subsample)
    trials = pd.read_csv("results/optuna_xgb_trials.csv")
    trials = trials[trials["state"] == "COMPLETE"]
    for _, t in trials.iterrows():
        info = ast.literal_eval(t["user_attrs_info"])
        log_run(f"xgb_optuna_trial_{int(t['number'])}",
                {"model": "xgboost", "n_train": info["n_train"], "max_tfidf": 20_000,
                 "max_depth": int(t["params_max_depth"]),
                 "learning_rate": t["params_learning_rate"],
                 "subsample": t["params_subsample"],
                 "colsample_bytree": t["params_colsample_bytree"],
                 "min_child_weight": int(t["params_min_child_weight"])},
                {"val_macro_f1": t["value"], "train_time_s": info["train_time"],
                 "best_iteration": info["best_iteration"]},
                tags={**LATE, "stage": "tuning"})

    print(f"logged {4 + len(trials)} runs")


if __name__ == "__main__":
    main()
