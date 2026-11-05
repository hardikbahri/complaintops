"""End-to-end quality of the triage pipeline on VALIDATION, under the served 24-class taxonomy.

CURRENT pipeline: the classifier answers every complaint at or above the confidence threshold; every other complaint
goes to a human reviewer (with suggestions and a note). Its numbers are exact on validation: no LLM chooses anything.
LLM-DECIDES variant (the earlier design, no longer used): for the unsure complaints an LLM chose among candidate
labels. Estimated from the LLM-evaluated random sample in results/agent_llm_eval.csv, each sampled row weighted up
to stand for the unsure complaints it was drawn from; bootstrap intervals show the sampling noise.
Complaints sent to a human have no automated answer, so the score depends on how they are counted; every reading is
reported: automated answers only (with the coverage), humans assumed correct (an assumption about the reviewers, not a
measurement), humans counted wrong (the worst case).
    python -m complaintops.pipeline_eval
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from complaintops.agent import THRESHOLD
from complaintops.lora_model import apply_bias

R = Path("results")
N_BOOT = 1000


def scores(true, pred, w, k):
    """Weighted F1 and accuracy with row weights; classes are integer codes 0..k-1 (k = 'human review')."""
    if w.sum() == 0:
        return dict(weighted_f1=float("nan"), accuracy=float("nan"))
    tp = np.bincount(true[pred == true], weights=w[pred == true], minlength=k + 1)
    support = np.bincount(true, weights=w, minlength=k + 1)
    predicted = np.bincount(pred, weights=w, minlength=k + 1)
    denom = support + predicted
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 0.0)
    return dict(weighted_f1=float((support * f1).sum() / w.sum()), accuracy=float(tp.sum() / w.sum()))


def current(true, clf, conf, k, rows=None):
    """The served pipeline: the classifier on the confident complaints, a human for the rest (exact; rows resamples)."""
    if rows is not None:
        true, clf, conf = true[rows], clf[rows], conf[rows]
    sure = conf >= THRESHOLD
    w = np.ones(len(true))
    answered = np.where(sure, 1.0, 0.0)
    out = {"classifier alone, every complaint": scores(true, clf, w, k),
           "pipeline, automated answers only": scores(true, clf, answered, k),
           "pipeline, humans assumed correct": scores(true, np.where(sure, clf, true), w, k),
           "pipeline, humans counted wrong": scores(true, np.where(sure, clf, k), w, k)}
    out["pipeline, automated answers only"]["coverage"] = float(sure.mean())
    return out


def llm_decides(true, clf, conf, sample, llm_code, llm_ok, k, rng=None):
    sure = np.flatnonzero(conf >= THRESHOLD)
    n_unsure = int((conf < THRESHOLD).sum())
    s = np.arange(len(sample)) if rng is None else rng.integers(0, len(sample), len(sample))
    rows = np.r_[sure, sample[s]]
    w = np.r_[np.ones(len(sure)), np.full(len(s), n_unsure / len(s))]
    t, c = true[rows], clf[rows]
    is_unsure = np.r_[np.zeros(len(sure), bool), np.ones(len(s), bool)]
    ok = np.r_[np.ones(len(sure), bool), llm_ok[s]]
    llm = np.r_[np.zeros(len(sure), int), llm_code[s]]
    pipe = np.where(is_unsure & ok, llm, c)  # confident -> classifier; unsure & LLM answered -> LLM
    human = is_unsure & ~ok                  # unsure & the LLM could not decide -> human queue
    answered = np.where(human, 0.0, w)
    out = {"classifier alone": scores(t, c, w, k),
           "pipeline, automated answers only": scores(t, pipe, answered, k),
           "pipeline, humans assumed correct": scores(t, np.where(human, t, pipe), w, k),
           "pipeline, humans counted wrong": scores(t, np.where(human, k, pipe), w, k),
           "classifier alone, same answered complaints": scores(t, c, answered, k)}
    out["pipeline, automated answers only"]["coverage"] = float(answered.sum() / w.sum())
    return out


def with_intervals(point, boots):
    result = {}
    for name, m in point.items():
        result[name] = dict(m)
        for key in ("weighted_f1", "accuracy"):
            vals = np.array([b[name][key] for b in boots])
            result[name][key + "_ci95"] = [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]
    return result


def main():
    from complaintops.taxonomy import merge_label, merge_proba, merged_classes

    full = json.load(open("data/lora_colab/classes.json"))
    classes = merged_classes(full)  # the served 24-class taxonomy
    k = len(classes)
    code = {c: i for i, c in enumerate(classes)}
    bias = json.load(open("results/lora_bias_weighted.json"))["bias"]
    va = pd.read_parquet("data/processed/val.parquet").reset_index(drop=True)
    true = va["label"].map(lambda s: code[merge_label(s)]).to_numpy()
    p = merge_proba(apply_bias(np.load("data/lora_colab/proba_val.npy"), bias), full)
    conf, clf = p.max(1), p.argmax(1)
    rng = np.random.default_rng(0)

    # the served pipeline: exact on validation, bootstrap over validation rows for the intervals
    cur_point = current(true, clf, conf, k)
    cur_boots = [current(true, clf, conf, k, rng.integers(0, len(true), len(true))) for _ in range(N_BOOT)]
    result = {"current": with_intervals(cur_point, cur_boots)}

    # the same policy on the held-out TEST split: the threshold (0.8) and the offsets were fixed on validation
    te = pd.read_parquet("data/processed/test.parquet").reset_index(drop=True)
    t_true = te["label"].map(lambda s: code[merge_label(s)]).to_numpy()
    pt = merge_proba(np.load("data/lora_colab/proba_test_bias.npy"), full)
    t_conf, t_clf = pt.max(1), pt.argmax(1)
    t_point = current(t_true, t_clf, t_conf, k)
    t_boots = [current(t_true, t_clf, t_conf, k, rng.integers(0, len(t_true), len(t_true))) for _ in range(N_BOOT)]
    result["current_test"] = with_intervals(t_point, t_boots)

    # the earlier variant in which an LLM chose the label for unsure complaints (estimated from a sample)
    d = pd.read_csv(R / "agent_llm_eval.csv")
    sample = d["row"].to_numpy()
    llm_ok = d["llm_ok"].to_numpy(dtype=bool)
    llm_code = d["llm_label"].map(code).fillna(-1).astype(int).to_numpy()
    assert (conf[sample] < THRESHOLD).all(), "the sample must be unsure complaints"
    assert (llm_code[llm_ok] >= 0).all()
    point = llm_decides(true, clf, conf, sample, llm_code, llm_ok, k)
    boots = [llm_decides(true, clf, conf, sample, llm_code, llm_ok, k, rng) for _ in range(N_BOOT)]
    result["llm_decides"] = with_intervals(point, boots)
    diff = np.array([b["pipeline, automated answers only"]["weighted_f1"] - b["classifier alone, same answered complaints"]["weighted_f1"] for b in boots])
    result["llm_decides"]["_paired_automated_vs_classifier_same_complaints"] = dict(
        point=point["pipeline, automated answers only"]["weighted_f1"] - point["classifier alone, same answered complaints"]["weighted_f1"],
        ci95=[float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))])
    result["_meta"] = dict(split="validation", threshold=THRESHOLD, n_validation=len(va), n_classes=k,
                           n_unsure=int((conf < THRESHOLD).sum()), n_llm_sample=len(sample), n_bootstrap=N_BOOT)
    json.dump(result, open(R / "pipeline_eval.json", "w"), indent=2)

    def show(title, block):
        print(f"\n{title}\n{'reading':46s} {'weighted F1':>12s} {'95% CI':>18s} {'accuracy':>9s}")
        for name, m in block.items():
            if name.startswith("_"):
                continue
            lo, hi = m["weighted_f1_ci95"]
            extra = f"   coverage {m['coverage']:.1%}" if "coverage" in m else ""
            print(f"{name:46s} {m['weighted_f1']:12.4f}  [{lo:.4f}, {hi:.4f}] {m['accuracy']:9.4f}{extra}")

    show("CURRENT pipeline (classifier answers the confident complaints, a person gets the rest) - exact on validation",
         result["current"])
    show("CURRENT pipeline on the held-out TEST split (threshold and offsets fixed on validation)", result["current_test"])
    show("EARLIER variant: an LLM chose the label for unsure complaints (estimated from 150 sampled complaints)",
         result["llm_decides"])


if __name__ == "__main__":
    main()
