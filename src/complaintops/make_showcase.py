"""Collect the project's real numbers into web/showcase.json, which the web page reads.

Nothing on the page is typed in by hand: validation scores come from the MLflow runs, held-out test scores from
the evaluation-gate reports, and speed / drift / agent results from their result files.
    python -m complaintops.make_showcase            # numbers only
    python -m complaintops.make_showcase --record   # also run the real agent on example complaints (uses the LLM)
"""
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("web/showcase.json")
R = Path("results")
EXAMPLES = [
    ("A collector keeps calling me about a debt that is not mine.", "Confident case"),
    ("I was charged an overdraft fee even though I had enough money in my account.",
     "Unsure case"),
    ("Please help. Something is wrong with my account and nobody is answering me.",
     "Vague case"),
]


def load(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def mlflow_runs():
    import mlflow

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    exp = mlflow.get_experiment_by_name("complaintops")
    df = mlflow.search_runs([exp.experiment_id], max_results=500)
    return {r["tags.mlflow.runName"]: r for _, r in df.iterrows()}


def calibration(merged=True):
    from complaintops.lora_model import apply_bias
    from complaintops.taxonomy import merge_label, merge_proba, merged_classes

    full = json.load(open("data/lora_colab/classes.json"))
    bias = json.load(open("results/lora_bias_weighted.json"))["bias"]
    va = pd.read_parquet("data/processed/val.parquet")
    p = apply_bias(np.load("data/lora_colab/proba_val.npy"), bias)
    if merged:
        classes, p = merged_classes(full), merge_proba(p, full)
        y = va["label"].map(lambda s: classes.index(merge_label(s))).to_numpy()
    else:
        y = va["label"].map({c: i for i, c in enumerate(full)}).to_numpy()
    conf, ok = p.max(1), p.argmax(1) == y
    bins = []
    for lo, hi in [(0, .3), (.3, .4), (.4, .5), (.5, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.01)]:
        m = (conf >= lo) & (conf < hi)
        bins.append(dict(lo=lo, hi=min(hi, 1.0), share=float(m.mean()), accuracy=float(ok[m].mean())))
    thresholds = [dict(t=t, handled=float((conf >= t).mean()), accuracy=float(ok[conf >= t].mean()),
                       rest_accuracy=float(ok[conf < t].mean())) for t in (0.5, 0.6, 0.7, 0.8, 0.9)]
    unsure = conf < 0.8
    top3 = np.argsort(-p, axis=1)[:, :3]
    in_top3 = np.array([y[i] in top3[i] for i in range(len(y))])
    big3 = va["label"].str.startswith("Credit reporting | ").mean()
    return dict(bins=bins, thresholds=thresholds, unsure_share=float(unsure.mean()),
                unsure_top1=float(ok[unsure].mean()), unsure_top3=float(in_top3[unsure].mean()),
                credit_reporting_share=float(big3))


def taxonomy_block():
    """Scores on the served 24-class taxonomy: validation for each model, held-out test with bootstrap intervals."""
    from sklearn.metrics import f1_score

    from complaintops.evaluate import bootstrap, interval, weighted_f1
    from complaintops.lora_model import apply_bias
    from complaintops.taxonomy import MERGED_LABEL, merge_label, merge_proba, merged_classes

    full = json.load(open("data/lora_colab/classes.json"))
    classes = merged_classes(full)
    k = len(classes)
    bias = json.load(open("results/lora_bias_weighted.json"))["bias"]
    va = pd.read_parquet("data/processed/val.parquet")
    te = pd.read_parquet("data/processed/test.parquet")
    y_va = va["label"].map(lambda s: classes.index(merge_label(s))).to_numpy()
    y_te = te["label"].map(lambda s: classes.index(merge_label(s))).to_numpy()
    M = lambda a: merge_proba(a, full)  # noqa: E731
    val = {"TF-IDF + logistic regression": (M(np.load("data/proba_lr_val.npy")), "the baseline"),
           "DistilBERT + LoRA, 60k rows": (M(np.load("data/lora_run/proba_val.npy")), "a quarter of the data"),
           "DistilBERT + LoRA, all rows": (M(np.load("data/lora_colab/proba_val.npy")), "rank 16, 3 epochs"),
           "LoRA + per-class offsets (champion)": (M(apply_bias(np.load("data/lora_colab/proba_val.npy"), bias)),
                                                   "the offsets were tuned on these validation rows")}
    validation = []
    for name, (p, note) in val.items():
        pred = p.argmax(1)
        validation.append(dict(name=name, weighted_f1=float(weighted_f1(y_va, pred, k)),
                               macro_f1=float(f1_score(y_va, pred, average="macro")),
                               accuracy=float((pred == y_va).mean()), note=note))
    test_probs = {"TF-IDF baseline (champion v1)": M(np.load("data/proba_lr_test.npy")),
                  "LoRA, all training rows": M(np.load("data/lora_colab/proba_test.npy")),
                  "LoRA + class offsets (champion v4)": M(np.load("data/lora_colab/proba_test_bias.npy"))}
    preds = {n: p.argmax(1) for n, p in test_probs.items()}
    boots = bootstrap(y_te, preds, k, n=1000)
    test = []
    for n, pred in preds.items():
        lo, hi = interval(boots[n])
        test.append(dict(name=n, weighted_f1=float(weighted_f1(y_te, pred, k)), ci95=[lo, hi],
                         macro_f1=float(f1_score(y_te, pred, average="macro")), accuracy=float((pred == y_te).mean())))
    d_lo, d_hi = interval(boots["LoRA + class offsets (champion v4)"] - boots["TF-IDF baseline (champion v1)"])
    return dict(n_classes=k, merged_label=MERGED_LABEL, validation=validation, test=test,
                champion_gain_over_baseline=dict(point=test[2]["weighted_f1"] - test[0]["weighted_f1"], ci95=[d_lo, d_hi]),
                trio=(load(R / "taxonomy_eval.json") or {}).get("trio_confusion_val"))


def agent_eval():
    """The LLM evaluation summary plus the paired comparison against the classifier, from the saved per-complaint rows."""
    summary = load(R / "agent_llm_eval.json")
    if summary is None or not (R / "agent_llm_eval.csv").exists():
        return summary
    from math import comb

    d = pd.read_csv(R / "agent_llm_eval.csv")
    a = d[d["llm_ok"]]
    llm_right, clf_right = a["llm_label"] == a["true"], a["classifier"] == a["true"]
    b, c = int((llm_right & ~clf_right).sum()), int((~llm_right & clf_right).sum())
    n, k = b + c, min(b, c)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0  # two-sided sign test
    summary.update(agree_with_classifier=float((a["llm_label"] == a["classifier"]).mean()),
                   llm_only_right=b, classifier_only_right=c, sign_test_p=p,
                   llm_says_high=float((a["llm_conf"] == "high").mean()), answered=int(len(a)))
    return summary


def count_tests():
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], capture_output=True, text=True).stdout
    last = [ln for ln in out.splitlines() if "test" in ln and ("collected" in ln or " tests" in ln)]
    return int(last[-1].split()[0]) if last else None


def gate(path):
    g = load(path)
    return g and dict(champion=g["champion"], candidate=g["candidate"], difference=g["difference"],
                      checks=[dict(name=c["name"], passed=c["passed"], detail=c["detail"]) for c in g["checks"]],
                      passed=g["passed"])


async def record_runs():
    from complaintops.agent import ComplaintAgent
    from complaintops.config import load_env

    load_env()
    runs = []
    async with ComplaintAgent() as agent:
        for text, label in EXAMPLES:
            res = await agent.run(text)
            runs.append(dict(label=label, text=text, mode="recorded", llm_provider=agent.llm.name,
                             threshold=agent.threshold, **res))
            await asyncio.sleep(2)
    return runs


def main():
    runs = mlflow_runs()

    def m(name, key):
        v = runs[name].get(f"metrics.{key}")
        return None if v is None or pd.isna(v) else float(v)

    models = [
        dict(name="TF-IDF + logistic regression", weighted_f1=m("baseline_tfidf_lr", "val_weighted_f1"),
             macro_f1=m("baseline_tfidf_lr", "val_macro_f1"), note="the baseline"),
        dict(name="XGBoost (Optuna-tuned)", weighted_f1=m("xgboost_final", "val_weighted_f1"),
             macro_f1=m("xgboost_final", "val_macro_f1"), note="did not beat the baseline"),
        dict(name="DistilBERT + LoRA, 60k rows", weighted_f1=m("lora_distilbert_60k_local", "val_weighted_f1"),
             macro_f1=m("lora_distilbert_60k_local", "val_macro_f1"), note="a quarter of the data"),
        dict(name="DistilBERT + LoRA, all rows", weighted_f1=m("lora_distilbert_full_colab", "val_weighted_f1"),
             macro_f1=m("lora_distilbert_full_colab", "val_macro_f1"), note="rank 16, 3 epochs"),
        dict(name="LoRA + per-class offsets (champion)", weighted_f1=m("lora_bias_tuned_weighted", "val_weighted_f1_crossfit"),
             macro_f1=m("lora_bias_tuned_weighted", "val_macro_f1_crossfit"),
             note="offsets tuned on other validation rows (cross-fitted)"),
    ]
    data = dict(
        generated=time.strftime("%Y-%m-%d"),
        facts=dict(complaints=335_499, train=234_849, validation=50_325, test=50_325, classes=24, classes_original=26,
                   train_dates="2015-03 to 2023-02", validation_dates="2023-02 to 2023-09",
                   test_dates="2023-09 to 2024-05", tests=count_tests(), mlflow_runs=len(runs)),
        models_validation=models,
        product_task=dict(weighted_f1=m("product_tfidf_lr", "val_weighted_f1"), macro_f1=m("product_tfidf_lr", "val_macro_f1"),
                          accuracy=m("product_tfidf_lr", "val_accuracy"), classes=10),
        gates=dict(blocked=gate(R / "gate_v2_vs_v1.json"), promoted=gate(R / "gate_v3_vs_v1.json"),
                   refinement=gate(R / "gate_v4_vs_v3.json")),
        taxonomy=taxonomy_block(),
        calibration=calibration(True), calibration_26=calibration(False),
        serving=load(R / "serving_benchmark.json"),
        drift=load(R / "drift" / "summary.json"),
        agent_eval=agent_eval(),
        pipeline_eval=load(R / "pipeline_eval.json"),
        recorded_runs=[],
    )
    if "--record" in sys.argv:
        data["recorded_runs"] = asyncio.run(record_runs())
    elif OUT.exists():  # keep earlier recordings when only refreshing numbers
        data["recorded_runs"] = json.loads(OUT.read_text()).get("recorded_runs", [])
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} KB); recorded runs: {len(data['recorded_runs'])}")


if __name__ == "__main__":
    main()
