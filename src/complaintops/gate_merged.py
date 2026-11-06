"""The evaluation gate, run on the 24 served classes. Step 7, redone for the served taxonomy.

Same rules, same registered models and saved test probabilities as promote.py; the only change is that each model's
26 class probabilities are added into the 24 served classes (taxonomy.merge_proba) before predicting. Nothing is
promoted or demoted: this only writes results/gate24_*.json/.txt for the page.
    python -m complaintops.gate_merged
"""
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from complaintops.evaluate import format_report, run_gate
from complaintops.promote import CLASSES, NAME, load_scorer, test_probabilities
from complaintops.taxonomy import merge_label, merge_proba, merged_classes
from complaintops.tracking import TRACKING_URI

PAIRS = [(2, 1, False), (3, 1, False), (4, 3, True)]  # (candidate, champion, require_improvement), as originally run


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    new = merged_classes(CLASSES)
    te = pd.read_parquet("data/processed/test.parquet")
    golden = pd.read_csv("tests/golden_set.csv")
    y = te["label"].map(lambda s: new.index(merge_label(s))).to_numpy()
    g_y = golden["label"].map(lambda s: new.index(merge_label(s))).to_numpy()
    texts = te["narrative"].tolist()
    M = lambda p: merge_proba(np.asarray(p), CLASSES)  # noqa: E731
    cache = {}

    def get(version):
        if version not in cache:
            v = client.get_model_version(NAME, str(version))
            scorer = load_scorer(f"models:/{NAME}/{version}")
            cache[version] = (M(test_probabilities(v, scorer, texts)).argmax(1), M(scorer(golden["text"])).argmax(1))
        return cache[version]

    for cand, champ, improve in PAIRS:
        (tc, gc), (tn, gn) = get(champ), get(cand)
        report = run_gate(y, tc, tn, new, golden=(g_y, gc, gn), require_improvement=improve)
        text = format_report(report)
        print(f"v{cand} vs v{champ}\n{text}\n")
        stem = f"results/gate24_v{cand}_vs_v{champ}"
        Path(stem + ".txt").write_text(text)
        json.dump(report, open(stem + ".json", "w"), indent=2)


if __name__ == "__main__":
    main()
