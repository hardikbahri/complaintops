"""Train a deliberately weak model (the baseline pipeline on 5% of train) and register it. Step 7.

Its only purpose is to prove the evaluation gate blocks a bad candidate.
Run from the project root:  python -m complaintops.make_weak_model
"""
import mlflow
import pandas as pd
from sklearn.metrics import f1_score

from complaintops.tracking import EXPERIMENT, TRACKING_URI, git_commit
from complaintops.train_baseline import make_pipeline

NAME = "complaint-classifier"
FRACTION = 0.05


def main():
    tr = pd.read_parquet("data/processed/train.parquet").sample(frac=FRACTION, random_state=42)
    va = pd.read_parquet("data/processed/val.parquet")
    pipe = make_pipeline()
    pipe.fit(tr["narrative"], tr["label"])
    wf1 = f1_score(va["label"], pipe.predict(va["narrative"]), average="weighted")
    print(f"weak model: {len(tr)} train rows, val weighted F1 {wf1:.4f}")

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="weak_tfidf_lr_5pct"):
        mlflow.set_tags({"git_commit": git_commit(), "stage": "candidate",
                         "note": "deliberately weak: 5% of train, to prove the gate blocks it"})
        mlflow.log_params({"model": "tfidf_logreg", "train_fraction": FRACTION, "n_train": len(tr),
                           "n_classes": pipe.classes_.size})
        mlflow.log_metric("val_weighted_f1", wf1)
        info = mlflow.sklearn.log_model(pipe, name="model")
        version = mlflow.register_model(info.model_uri, NAME)
    print(f"registered {NAME} v{version.version}")


if __name__ == "__main__":
    main()
