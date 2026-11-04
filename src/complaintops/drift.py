"""Drift monitoring with Evidently, PSI and a text domain classifier. Step 8.

Reference = a random 70% of the VALIDATION split. Batches are compared with it:
  normal      5,000 other validation rows (same period, disjoint from the reference)
  shift_time  the newest month of the test period (real, later data; no labels are used)
  shift_mix   complaints from rarely seen products, deliberately over-sampled (the class mix changes)
Signals: Evidently's share of drifted columns, PSI on the class mix / prediction confidence / text statistics,
and a domain classifier (TF-IDF + logistic regression) that tries to tell reference from current text:
AUC near 0.5 = indistinguishable, high AUC = drift. Thresholds follow the usual rules of thumb.
Run from the project root:  python -m complaintops.drift
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from complaintops.features import FEATURE_COLUMNS, build_features

PSI_WARN, PSI_ALERT = 0.10, 0.25       # PSI: < 0.1 stable, 0.1-0.25 watch, > 0.25 significant
AUC_WARN, AUC_ALERT = 0.60, 0.70       # domain-classifier AUC: 0.5 = same, higher = more different
SHARE_ALERT = 0.5                      # Evidently: alert when at least half of the columns drifted
PSI_COLUMNS = ["n_chars", "n_words", "n_xxxx", "n_dollar", "conf", "pred"]
RARE_PRODUCTS = ["Student loan", "Vehicle loan or lease", "Payday/personal loan",
                 "Money transfer, virtual currency, or money service"]
OUT = Path("results/drift")


def psi(ref, cur, bins=10):
    """Population Stability Index of `cur` against `ref` (quantile bins for numbers, categories otherwise)."""
    ref, cur = np.asarray(ref), np.asarray(cur)
    if ref.dtype == object or ref.dtype.kind in "US":
        cats = sorted(set(ref) | set(cur))
        r = np.array([(ref == c).mean() for c in cats])
        c_ = np.array([(cur == c).mean() for c in cats])
    else:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        r = np.histogram(ref, edges)[0] / len(ref)
        c_ = np.histogram(cur, edges)[0] / len(cur)
    r, c_ = np.clip(r, 1e-4, None), np.clip(c_, 1e-4, None)
    return float(((c_ - r) * np.log(c_ / r)).sum())


def text_auc(ref_texts, cur_texts, n=3000, seed=0):
    """Cross-validated AUC of a classifier that separates reference text from current text."""
    m = min(n, len(ref_texts), len(cur_texts))
    a = pd.Series(ref_texts).sample(m, random_state=seed).tolist()
    b = pd.Series(cur_texts).sample(m, random_state=seed).tolist()
    X = TfidfVectorizer(min_df=3, max_features=20000, sublinear_tf=True).fit_transform(a + b)
    y = np.r_[np.zeros(m), np.ones(m)]
    p = cross_val_predict(LogisticRegression(max_iter=500), X, y, cv=5, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def check(ref, cur, evidently_share=None):
    """Compare a current batch with the reference. Returns the metrics and the alerts/warnings raised."""
    metrics = {f"psi_{c}": psi(ref[c], cur[c]) for c in PSI_COLUMNS}
    metrics["mean_confidence_ref"] = float(ref["conf"].mean())
    metrics["mean_confidence_cur"] = float(cur["conf"].mean())
    metrics["text_auc"] = text_auc(ref["narrative"].tolist(), cur["narrative"].tolist())
    if evidently_share is not None:
        metrics["evidently_drifted_share"] = float(evidently_share)

    alerts, warnings = [], []
    for c in PSI_COLUMNS:
        v = metrics[f"psi_{c}"]
        what = {"pred": "class mix (predicted labels)", "conf": "prediction confidence"}.get(c, f"text statistic {c}")
        if v >= PSI_ALERT:
            alerts.append(f"PSI {v:.2f} on {what} (alert at {PSI_ALERT})")
        elif v >= PSI_WARN:
            warnings.append(f"PSI {v:.2f} on {what} (watch at {PSI_WARN})")
    auc = metrics["text_auc"]
    if auc >= AUC_ALERT:
        alerts.append(f"text domain-classifier AUC {auc:.2f} (alert at {AUC_ALERT})")
    elif auc >= AUC_WARN:
        warnings.append(f"text domain-classifier AUC {auc:.2f} (watch at {AUC_WARN})")
    if evidently_share is not None and evidently_share >= SHARE_ALERT:
        alerts.append(f"Evidently: {evidently_share:.0%} of columns drifted (alert at {SHARE_ALERT:.0%})")
    status = "alert" if alerts else "warn" if warnings else "ok"
    return dict(status=status, alerts=alerts, warnings=warnings, metrics=metrics)


def scored_frames():
    """Validation and test rows with text statistics and the champion's confidence / predicted class."""
    from complaintops.lora_model import apply_bias

    classes = np.array(json.load(open("data/lora_colab/classes.json")))
    bias = json.load(open("results/lora_bias_weighted.json"))["bias"]  # offsets baked into champion v4
    val = pd.read_parquet("data/processed/val.parquet").reset_index(drop=True)
    test = pd.read_parquet("data/processed/test.parquet").reset_index(drop=True)
    proba = {"val": apply_bias(np.load("data/lora_colab/proba_val.npy"), bias),
             "test": np.load("data/lora_colab/proba_test_bias.npy")}
    out = {}
    for name, df in (("val", val), ("test", test)):
        p = proba[name]
        df["conf"] = p.max(1)
        df["pred"] = classes[p.argmax(1)]
        df[FEATURE_COLUMNS] = build_features(df).to_numpy()
        out[name] = df
    return out["val"], out["test"]


def make_batches(val, test, seed=42, size=5000):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(val))
    cut = int(0.7 * len(val))
    ref, held = val.iloc[idx[:cut]], val.iloc[idx[cut:]]
    newest = test[test["date"] >= test["date"].max() - pd.Timedelta(days=30)]
    pool = pd.concat([val, test])
    rare = pool["product"].isin(RARE_PRODUCTS)
    return dict(
        reference=ref,
        normal=held.sample(size, random_state=1),
        shift_time=newest.sample(min(size, len(newest)), random_state=1),
        shift_mix=pd.concat([pool[rare].sample(int(0.7 * size), random_state=1),
                             pool[~rare].sample(int(0.3 * size), random_state=1)]))


def evidently_report(ref, cur, html_path):
    """Evidently data-drift report (HTML). Returns the share of drifted columns."""
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    num = ["n_chars", "n_words", "n_xxxx", "n_dollar", "conf"]
    cols = num + ["pred", "narrative"]
    dd = DataDefinition(numerical_columns=num, categorical_columns=["pred"], text_columns=["narrative"])
    snap = Report([DataDriftPreset()]).run(
        current_data=Dataset.from_pandas(cur[cols].reset_index(drop=True), data_definition=dd),
        reference_data=Dataset.from_pandas(ref[cols].reset_index(drop=True), data_definition=dd))
    Path(html_path).parent.mkdir(parents=True, exist_ok=True)
    snap.save_html(str(html_path))
    return snap.dict()["metrics"][0]["value"]["share"]


def main():
    import mlflow

    from complaintops.tracking import EXPERIMENT, TRACKING_URI, git_commit

    val, test = scored_frames()
    batches = make_batches(val, test)
    ref = batches.pop("reference")
    print(f"reference: {len(ref)} validation rows, mean confidence {ref['conf'].mean():.3f}\n")

    summary = {}
    for name, cur in batches.items():
        share = evidently_report(ref, cur, OUT / f"{name}.html")
        res = check(ref, cur, share)
        res["n_rows"] = len(cur)
        summary[name] = res
        print(f"== {name} ({len(cur)} rows): {res['status'].upper()}")
        for line in res["alerts"]:
            print("   ALERT:", line)
        for line in res["warnings"]:
            print("   watch:", line)
        m = res["metrics"]
        print(f"   text AUC {m['text_auc']:.3f}, class-mix PSI {m['psi_pred']:.3f}, mean confidence "
              f"{m['mean_confidence_cur']:.3f} (ref {m['mean_confidence_ref']:.3f}), "
              f"Evidently drifted share {m['evidently_drifted_share']:.0%}\n")
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="drift_monitoring"):
        mlflow.set_tags({"git_commit": git_commit(), "stage": "monitoring"})
        for name, res in summary.items():
            mlflow.log_metrics({f"{name}_{k}": v for k, v in res["metrics"].items()})
            mlflow.set_tag(f"{name}_status", res["status"])
        mlflow.log_artifacts(str(OUT))


if __name__ == "__main__":
    main()
