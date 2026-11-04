"""Does the LLM step actually help? Measured on VALIDATION complaints the classifier is unsure about.

For a random sample of validation complaints with classifier confidence below the agent's threshold, this builds
the same candidates the agent would (classifier top 3 + labels agreed by similar past complaints), asks the LLM to
choose, and compares with the true label. Complaint text is sent to the LLM provider (public CFPB narratives).
    python -m complaintops.evaluate_agent --n 150 --model openai/gpt-oss-120b
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from complaintops.agent import THRESHOLD, build_candidates
from complaintops.config import load_env


def wilson(k, n, z=1.96):
    """95% interval for a proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def main():
    load_env()
    from complaintops.llm import OpenAICompatibleLLM, PRESETS, decide_label
    from complaintops.lora_model import apply_bias
    from complaintops.vector_store import connect, search

    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--pause", type=float, default=2.2, help="seconds between LLM calls (free-tier rate limit)")
    args = ap.parse_args()

    from complaintops.taxonomy import merge_label, merge_proba, merged_classes

    full = json.load(open("data/lora_colab/classes.json"))
    classes = merged_classes(full)  # the served 24-class taxonomy
    bias = json.load(open("results/lora_bias_weighted.json"))["bias"]
    va = pd.read_parquet("data/processed/val.parquet").reset_index(drop=True)
    p = merge_proba(apply_bias(np.load("data/lora_colab/proba_val.npy"), bias), full)
    conf, top1 = p.max(1), p.argmax(1)
    y = va["label"].map(lambda s: classes.index(merge_label(s))).to_numpy()
    unsure = np.flatnonzero(conf < THRESHOLD)
    rows = np.random.default_rng(args.seed).choice(unsure, min(args.n, len(unsure)), replace=False)
    print(f"{len(unsure)} of {len(va)} validation complaints are below confidence {THRESHOLD}; testing {len(rows)}")

    url, key_var, _ = PRESETS["groq"]
    llm = OpenAICompatibleLLM(url, os.environ[key_var], args.model)
    conn = connect()
    out = []
    for n, i in enumerate(rows, 1):
        top3 = np.argsort(-p[i])[:3]
        alts = [{"label": classes[j], "confidence": float(p[i, j])} for j in top3]
        similar = search(va["narrative"][i], k=5, conn=conn)
        cands = build_candidates(alts, similar)
        d = decide_label(llm, va["narrative"][i], cands, similar)
        out.append(dict(row=int(i), true=classes[y[i]], classifier=classes[top1[i]], conf=float(conf[i]),
                        in_candidates=classes[y[i]] in cands, llm_label=d["label"], llm_conf=d["confidence"],
                        llm_ok=d["ok"], reason=d["reason"]))
        if n % 25 == 0:
            print(f"  {n}/{len(rows)}", flush=True)
        time.sleep(args.pause)
    conn.close()

    df = pd.DataFrame(out)
    n = len(df)
    ok = df[df["llm_ok"]]
    clf_right = (df["classifier"] == df["true"]).sum()
    llm_right = (ok["llm_label"] == ok["true"]).sum()
    clf_on_answered = (ok["classifier"] == ok["true"]).sum()
    print(f"\nmodel {args.model}, {n} unsure validation complaints")
    print(f"  classifier top-1 right:                       {clf_right}/{n} = {clf_right / n:.3f}  "
          f"95% CI {tuple(round(v, 3) for v in wilson(clf_right, n))}")
    print(f"  true label among the LLM's candidates:        {df['in_candidates'].mean():.3f}")
    print(f"  LLM gave a usable answer (coverage):          {len(ok)}/{n} = {len(ok) / n:.3f}")
    print(f"  LLM right when it answered:                   {llm_right}/{len(ok)} = {llm_right / max(len(ok), 1):.3f}  "
          f"95% CI {tuple(round(v, 3) for v in wilson(llm_right, len(ok)))}")
    print(f"  classifier on those same complaints:          {clf_on_answered}/{len(ok)} = {clf_on_answered / max(len(ok), 1):.3f}")
    print(f"  sent on to a human:                           {n - len(ok)}/{n}")
    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/agent_llm_eval.csv", index=False)
    json.dump(dict(model=args.model, n=n, threshold=THRESHOLD, classifier_top1=clf_right / n,
                   in_candidates=float(df["in_candidates"].mean()), llm_coverage=len(ok) / n,
                   llm_accuracy_when_answered=llm_right / max(len(ok), 1),
                   classifier_accuracy_on_answered=clf_on_answered / max(len(ok), 1)),
              open("results/agent_llm_eval.json", "w"), indent=2)


if __name__ == "__main__":
    main()
