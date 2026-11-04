"""Small MLflow helpers so every training script logs runs the same way. Step 5."""
import subprocess

import mlflow

TRACKING_URI = "sqlite:///mlflow.db"  # one local file holds runs + the model registry
EXPERIMENT = "complaintops"


def git_commit():
    """Short commit hash, with '-dirty' if there are uncommitted changes."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def log_run(name, params, metrics, artifacts=(), model=None, tags=None):
    """Record one experiment run. Returns the MLflow run id.

    params    - settings you chose (dict), e.g. {"model": "tfidf_lr", "class_weight": "balanced"}
    metrics   - numbers you measured (dict), e.g. {"val_macro_f1": 0.4717}
    artifacts - local files to attach (reports, plots, prediction files)
    model     - a fitted scikit-learn model/pipeline to store (needed for the registry)
    tags      - free labels, e.g. {"logged_after_the_fact": "true"}
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tag("git_commit", git_commit())
        if tags:
            mlflow.set_tags(tags)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        for path in artifacts:
            mlflow.log_artifact(str(path))
        if model is not None:
            mlflow.sklearn.log_model(model, name="model")
        return run.info.run_id
