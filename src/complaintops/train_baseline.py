"""TF-IDF + logistic regression baseline, then XGBoost. Steps 3 and 4."""
"""TF-IDF + logistic regression baseline. Step 3."""
import time

import joblib
import matplotlib

matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


def make_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def load(split):
    df = pd.read_parquet(f"data/processed/{split}.parquet")
    return df["narrative"], df["label"]


def run_cv(n=50_000, folds=5, seed=42):
    """5-fold CV macro-F1 on a random subsample of train (full train would take >1 hour)."""
    X, y = load("train")
    idx = X.sample(n, random_state=seed).index
    t0 = time.time()
    scores = cross_val_score(make_pipeline(), X.loc[idx], y.loc[idx],
                             cv=folds, scoring="f1_macro", n_jobs=2)
    print(f"{folds}-fold macro-F1 on {n} train rows: {scores.mean():.4f} +/- {scores.std():.4f}")
    print("folds:", [round(s, 4) for s in scores], f"({time.time() - t0:.0f}s)")


XGB_DEFAULTS = dict(max_depth=6, learning_rate=0.1, subsample=0.8,
                    colsample_bytree=0.5, min_child_weight=1)


def load_full(split):
    return pd.read_parquet(f"data/processed/{split}.parquet")


def prepare_xgb_data(n_train=40_000, max_tfidf=20_000, seed=42):
    """Build the (expensive) feature matrices once, so tuning trials can reuse them.

    Early stopping uses a 10% hold-out carved from the TRAIN subsample, so the
    validation split stays untouched for the final comparison with the baseline.
    """
    import scipy.sparse as sp
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.class_weight import compute_sample_weight

    from complaintops.features import build_features

    tr = load_full("train")
    if n_train < len(tr):
        tr = tr.sample(n_train, random_state=seed)
    va = load_full("val")
    es = tr.sample(frac=0.1, random_state=seed)
    tr = tr.drop(es.index)

    t0 = time.time()
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True,
                            max_features=max_tfidf)
    tfidf.fit(tr["narrative"])  # fitted on the training rows only

    def matrix(df):
        num = sp.csr_matrix(build_features(df).to_numpy(dtype="float32"))
        return sp.hstack([tfidf.transform(df["narrative"]), num], format="csr")

    le = LabelEncoder().fit(tr["label"])
    X_tr, X_es, X_va = matrix(tr), matrix(es), matrix(va)
    y_tr, y_es, y_va = (le.transform(d["label"]) for d in (tr, es, va))
    return dict(
        X_tr=X_tr, X_es=X_es, X_va=X_va, y_tr=y_tr, y_es=y_es, y_va=y_va,
        w_tr=compute_sample_weight("balanced", y_tr),  # like class_weight="balanced"
        tfidf=tfidf, le=le, prep_time=time.time() - t0)


def train_xgb(data, params=None, n_estimators=400, device="cuda", seed=42):
    """Train XGBoost on prepared data. Returns (val macro-F1, info dict, classifier)."""
    import xgboost as xgb

    params = {**XGB_DEFAULTS, **(params or {})}
    t0 = time.time()
    clf = xgb.XGBClassifier(
        tree_method="hist", device=device, n_estimators=n_estimators,
        early_stopping_rounds=20, eval_metric="mlogloss", random_state=seed, **params)
    clf.fit(data["X_tr"], data["y_tr"], sample_weight=data["w_tr"],
            eval_set=[(data["X_es"], data["y_es"])], verbose=False)
    train_time = time.time() - t0
    val_f1 = f1_score(data["y_va"], clf.predict(data["X_va"]), average="macro")
    info = dict(train_time=round(train_time), best_iteration=clf.best_iteration,
                n_train=data["X_tr"].shape[0], n_features=data["X_tr"].shape[1])
    return val_f1, info, clf


def fit_xgb(n_train=40_000, params=None, max_tfidf=20_000, n_estimators=400,
            device="cuda", seed=42):
    data = prepare_xgb_data(n_train, max_tfidf, seed)
    f1, info, clf = train_xgb(data, params, n_estimators, device, seed)
    return f1, info, (clf, data["tfidf"], data["le"])


if __name__ == "__main__":
    import sys

    if "cv" in sys.argv:
        run_cv()
        sys.exit()

    if "xgb" in sys.argv:
        n = int(sys.argv[sys.argv.index("xgb") + 1]) if len(sys.argv) > sys.argv.index("xgb") + 1 else 40_000
        f1, info, _ = fit_xgb(n_train=n)
        print(f"XGBoost val macro-F1 {f1:.4f}  {info}")
        sys.exit()

    X_tr, y_tr = load("train")
    X_va, y_va = load("val")

    pipe = make_pipeline()
    t0 = time.time()
    pipe.fit(X_tr, y_tr)
    train_time = time.time() - t0

    pred = pipe.predict(X_va)
    val_f1 = f1_score(y_va, pred, average="macro")
    print(f"train time {train_time:.0f}s  val macro-F1 {val_f1:.4f}")

    # Save the model and the validation predictions so analysis doesn't need retraining.
    joblib.dump(pipe, "data/baseline.joblib")  # data/ is gitignored
    pd.DataFrame({"y": y_va, "pred": pred, "text": X_va}).to_parquet(
        "data/baseline_val_preds.parquet")

    # Per-class precision / recall / F1: shows which classes fail.
    print(classification_report(y_va, pred, zero_division=0))

    # Confusion matrix, each row normalized to sum to 1 (row = true class).
    fig, ax = plt.subplots(figsize=(14, 12))
    ConfusionMatrixDisplay.from_predictions(
        y_va, pred, normalize="true", xticks_rotation=90, include_values=False, ax=ax)
    plt.tight_layout()
    plt.savefig("data/confusion_baseline.png", dpi=120)
