"""Build the golden set: ~40 tricky complaints with known labels that every model must keep getting right.

Chosen from the VALIDATION split (never the test split) among rows where BOTH the baseline and the
LoRA model are correct and fairly confident, so a failure means a real regression, not a noisy label.
"Tricky" = very short, heavily redacted, or mentioning more than one product area.
Run once from the project root:  python -m complaintops.make_golden   (then review tests/golden_set.csv)
"""
import json
import re

import numpy as np
import pandas as pd

OUT = "tests/golden_set.csv"
N_TARGET = 40
MAX_WORDS = 200

AREAS = {
    "credit reporting": ["credit report", "credit bureau", "equifax", "experian", "transunion", "credit score"],
    "debt collection": ["debt collector", "collection agency", "collections", "collector"],
    "mortgage": ["mortgage", "escrow", "foreclos", "refinanc"],
    "credit card": ["credit card", "apr", "annual fee", "card statement"],
    "bank account": ["checking account", "savings account", "overdraft", "debit card"],
    "student loan": ["student loan", "navient", "fedloan", "nelnet"],
    "vehicle loan": ["auto loan", "car loan", "vehicle", "lease"],
    "money transfer": ["zelle", "paypal", "venmo", "wire transfer", "cash app"],
}


def areas_hit(text):
    t = text.lower()
    return [a for a, words in AREAS.items() if any(w in t for w in words)]


def main():
    classes = json.load(open("data/lora_colab/classes.json"))
    va = pd.read_parquet("data/processed/val.parquet").reset_index(drop=True)
    y = va["label"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    p_lr, p_lo = np.load("data/proba_lr_val.npy"), np.load("data/lora_colab/proba_val.npy")
    rows = np.arange(len(y))
    true_lr, true_lo = p_lr[rows, y], p_lo[rows, y]
    ok = (p_lr.argmax(1) == y) & (p_lo.argmax(1) == y) & (true_lr >= 0.5) & (true_lo >= 0.6)

    words = va["narrative"].str.split().str.len()
    n_x = va["narrative"].str.count("XXXX")
    notes = []
    for i in range(len(va)):
        tags = []
        if words[i] <= 30:
            tags.append("short")
        if n_x[i] >= 6:
            tags.append("heavily redacted")
        if len(areas_hit(va["narrative"][i])) >= 2:
            tags.append("mentions several product areas")
        notes.append(", ".join(tags))
    va["note"] = notes
    cand = va[ok & (words <= MAX_WORDS)].copy()
    cand["tricky"] = cand["note"] != ""
    print(f"{len(cand)} rows where both models are correct and confident; {int(cand['tricky'].sum())} tricky")

    # Priority: short or multi-topic complaints are the most informative; redaction alone is common, so it ranks lower.
    cand["priority"] = (2 * cand["note"].str.contains("short|several").astype(int)
                        + cand["note"].str.contains("redacted").astype(int))
    cand = cand.sample(frac=1, random_state=42)  # shuffle so ties are broken randomly
    cand = cand.sort_values("priority", ascending=False, kind="stable")
    picked = []
    for _, grp in cand.groupby("label"):  # first pass: the best example of every class
        picked.append(grp.index[0])
    second = [grp.index[1] for _, grp in cand.groupby("label") if len(grp) > 1]  # second-best per class
    second = cand.loc[second].sort_values("priority", ascending=False, kind="stable")
    picked += list(second.index[:N_TARGET - len(picked)])

    out = cand.loc[picked, ["narrative", "label", "note"]].rename(columns={"narrative": "text"})
    out["val_row"] = out.index
    out["text"] = out["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    out.to_csv(OUT, index=False)
    print(f"wrote {len(out)} cases to {OUT}; classes covered: {out['label'].nunique()} of {len(classes)}")
    missing = sorted(set(classes) - set(out["label"]))
    if missing:
        print("classes with no confident case (left out):")
        for m in missing:
            print("  -", m)
    print("\nnotes:", out["note"].replace("", "(plain)").value_counts().to_dict())


if __name__ == "__main__":
    main()
