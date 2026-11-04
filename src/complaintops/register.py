"""Register the best logged model as `complaint-classifier` and point the `champion` alias at it. Step 5.

Candidates are the 26-class runs, best validation macro-F1 first; the first one that has a
registrable model attached wins (the XGBoost run logged files, not a model, so it is skipped).
Run from the project root:  python -m complaintops.register
"""
import mlflow
from mlflow import MlflowClient

from complaintops.tracking import EXPERIMENT, TRACKING_URI

NAME = "complaint-classifier"


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    exp = mlflow.get_experiment_by_name(EXPERIMENT)
    runs = mlflow.search_runs(
        [exp.experiment_id], filter_string="params.n_classes = '26'",
        order_by=["metrics.val_macro_f1 DESC"])

    for _, run in runs.iterrows():
        try:
            version = mlflow.register_model(f"runs:/{run.run_id}/model", NAME)
        except Exception as e:  # this run has no model attached
            print(f"skip {run['tags.mlflow.runName']}: {type(e).__name__}")
            continue
        client = MlflowClient()
        client.set_registered_model_alias(NAME, "champion", version.version)
        client.set_model_version_tag(NAME, version.version, "git_commit", run["tags.git_commit"])
        print(f"registered {NAME} v{version.version} from run '{run['tags.mlflow.runName']}' "
              f"(val macro-F1 {run['metrics.val_macro_f1']:.4f}) -> alias 'champion'")
        return
    raise SystemExit("no run with a registrable model found")


if __name__ == "__main__":
    main()
