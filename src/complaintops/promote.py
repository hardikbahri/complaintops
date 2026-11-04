"""Run the evaluation gate on a candidate model version and promote it only if it passes. Step 7.

    python -m complaintops.promote --candidate-version 3 [--dry-run]

Loads the champion (alias `champion`) and the candidate from the MLflow registry, scores both on the
TEST split and on the golden set, applies the gate rules from evaluate.py, prints the report, and moves
the `champion` alias to the candidate only when every check passes. Exit code 0 = promoted, 1 = blocked.
Test-set probabilities saved at training time are reused when a version has a `test_proba_path` tag
(after a spot check against the live model); otherwise the model is run on the test set.
"""
import argparse
import json
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from complaintops.evaluate import format_report, run_gate
from complaintops.tracking import EXPERIMENT, TRACKING_URI, git_commit

NAME = "complaint-classifier"
CLASSES = json.load(open("data/lora_colab/classes.json"))  # alphabetical, same order as every model's output
SPOT_CHECK = 32


def load_scorer(uri):
    """Return f(texts) -> probability matrix (rows = texts, columns = CLASSES)."""
    if "sklearn" in mlflow.models.Model.load(uri).flavors:
        model = mlflow.sklearn.load_model(uri)
        assert list(model.classes_) == CLASSES, "model does not have all 26 classes in the expected order"
        return lambda texts: model.predict_proba(list(texts))
    model = mlflow.pyfunc.load_model(uri)

    def score(texts):
        out = model.predict(pd.DataFrame({"text": list(texts)}))
        assert list(out.columns) == CLASSES
        return out.to_numpy()
    return score


def test_probabilities(version, scorer, texts):
    """Saved probabilities if the version has them (spot-checked against the live model), else live."""
    path = version.tags.get("test_proba_path")
    if path and Path(path).exists():
        saved = np.load(path)
        live = scorer(texts[:SPOT_CHECK])
        agree = float((live.argmax(1) == saved[:SPOT_CHECK].argmax(1)).mean())
        print(f"  v{version.version}: using saved test probabilities ({path}); "
              f"spot check vs live model: {agree:.0%} agreement on {SPOT_CHECK} rows")
        assert agree >= 0.9, "saved probabilities do not match the registered model"
        return saved
    print(f"  v{version.version}: no saved probabilities, scoring {len(texts)} test rows with the model...")
    return scorer(texts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-version", type=int, required=True)
    ap.add_argument("--dry-run", action="store_true", help="run the gate but never move the alias")
    ap.add_argument("--require-improvement", action="store_true",
                    help="also require the candidate to be clearly better (lower bound of the difference > 0)")
    args = ap.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    champ = client.get_model_version_by_alias(NAME, "champion")
    cand = client.get_model_version(NAME, str(args.candidate_version))
    if champ.version == cand.version:
        sys.exit(f"v{cand.version} is already the champion")
    print(f"champion: v{champ.version}   candidate: v{cand.version}")

    te = pd.read_parquet("data/processed/test.parquet")
    y = te["label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    golden = pd.read_csv("tests/golden_set.csv")
    g_y = golden["label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()

    scorers = {v.version: load_scorer(f"models:/{NAME}/{v.version}") for v in (champ, cand)}
    texts = te["narrative"].tolist()
    p_champ = test_probabilities(champ, scorers[champ.version], texts)
    p_cand = test_probabilities(cand, scorers[cand.version], texts)
    g_champ = scorers[champ.version](golden["text"]).argmax(1)
    g_cand = scorers[cand.version](golden["text"]).argmax(1)

    report = run_gate(y, p_champ.argmax(1), p_cand.argmax(1), CLASSES, golden=(g_y, g_champ, g_cand),
                      require_improvement=args.require_improvement)
    text = format_report(report)
    print("\n" + text)

    Path("results").mkdir(exist_ok=True)
    stem = f"results/gate_v{cand.version}_vs_v{champ.version}"
    Path(stem + ".txt").write_text(text)
    json.dump(report, open(stem + ".json", "w"), indent=2)

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"gate_v{cand.version}_vs_v{champ.version}"):
        mlflow.set_tags({"git_commit": git_commit(), "stage": "gate",
                         "result": "passed" if report["passed"] else "blocked"})
        mlflow.log_params({"champion_version": champ.version, "candidate_version": cand.version})
        mlflow.log_metrics({"champion_test_weighted_f1": report["champion"]["weighted_f1"],
                            "candidate_test_weighted_f1": report["candidate"]["weighted_f1"],
                            "difference_lower_bound": report["difference"]["ci95"][0]})
        mlflow.log_artifact(stem + ".txt")
        mlflow.log_artifact(stem + ".json")

    client.set_model_version_tag(NAME, cand.version, "gate", "passed" if report["passed"] else "blocked")
    if report["passed"] and not args.dry_run:
        client.set_registered_model_alias(NAME, "champion", cand.version)
        print(f"alias 'champion' now points to v{cand.version}")
    elif report["passed"]:
        print("dry run: alias not moved")
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
