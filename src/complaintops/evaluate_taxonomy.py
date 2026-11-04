"""What does merging the three overlapping credit-reporting issues do? Measured with the existing models.

No retraining: the 26-class probabilities are summed into 24 classes (taxonomy.merge_proba). Validation is used to
look at the confusion and the confidence; the test split is scored once for the merged task, with nothing tuned.
    python -m complaintops.evaluate_taxonomy
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from complaintops.evaluate import bootstrap, interval, weighted_f1
from complaintops.lora_model import apply_bias
from complaintops.taxonomy import MERGE_MAP, merge_label, merge_proba, merged_classes

R = Path("results")


def main():
    classes = json.load(open("data/lora_colab/classes.json"))
    bias = json.load(open("results/lora_bias_weighted.json"))["bias"]
    new = merged_classes(classes)
    idx = {c: i for i, c in enumerate(new)}
    k = len(new)
    va = pd.read_parquet("data/processed/val.parquet")
    te = pd.read_parquet("data/processed/test.parquet")
    y26_va = va["label"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    y_va = va["label"].map(lambda s: idx[merge_label(s)]).to_numpy()
    y_te = te["label"].map(lambda s: idx[merge_label(s)]).to_numpy()
    print(f"{len(classes)} classes -> {k} classes; merged class = {len(MERGE_MAP)} labels into one")

    lora_va = merge_proba(apply_bias(np.load("data/lora_colab/proba_val.npy"), bias), classes)
    lora_te = merge_proba(np.load("data/lora_colab/proba_test_bias.npy"), classes)
    lr_va = merge_proba(np.load("data/proba_lr_val.npy"), classes)
    lr_te = merge_proba(np.load("data/proba_lr_test.npy"), classes)

    out = {"classes_before": len(classes), "classes_after": k, "merged_label": new[idx[merge_label(list(MERGE_MAP)[0])]]}

    # 1. the confusion the merge removes (validation, 26-class champion)
    trio = [classes.index(c) for c in MERGE_MAP]
    p26 = apply_bias(np.load("data/lora_colab/proba_val.npy"), bias).argmax(1)
    t = np.isin(y26_va, trio)
    exact = (p26[t] == y26_va[t]).mean()
    sibling = (np.isin(p26[t], trio) & (p26[t] != y26_va[t])).mean()
    outside = (~np.isin(p26[t], trio)).mean()
    out["trio_confusion_val"] = dict(rows=int(t.sum()), share_of_val=float(t.mean()), exactly_right=float(exact),
                                     swapped_with_a_sibling=float(sibling), predicted_outside_trio=float(outside))
    print(f"\nvalidation complaints whose true label is one of the three: {t.sum()} ({t.mean():.1%} of all)")
    print(f"  26-class model: exactly right {exact:.1%} | swapped with a sibling {sibling:.1%} | outside the three {outside:.1%}")

    # 2. quality on the merged task
    def row(name, y, p):
        pred = p.argmax(1)
        return dict(model=name, weighted_f1=float(weighted_f1(y, pred, k)), macro_f1=float(f1_score(y, pred, average="macro")),
                    accuracy=float((pred == y).mean()))
    out["validation"] = [row("TF-IDF + LR baseline", y_va, lr_va), row("LoRA + offsets (champion)", y_va, lora_va)]
    out["test"] = [row("TF-IDF + LR baseline", y_te, lr_te), row("LoRA + offsets (champion)", y_te, lora_te)]
    print("\nmerged 24-class task                   weighted F1  macro-F1  accuracy")
    for split in ("validation", "test"):
        for r in out[split]:
            print(f"  {split:10s} {r['model']:27s} {r['weighted_f1']:8.4f}  {r['macro_f1']:8.4f}  {r['accuracy']:8.4f}")
    boots = bootstrap(y_te, {"lr": lr_te.argmax(1), "lora": lora_te.argmax(1)}, k, n=1000)
    lo, hi = interval(boots["lora"] - boots["lr"])
    out["test_gain_lora_over_lr"] = dict(point=out["test"][1]["weighted_f1"] - out["test"][0]["weighted_f1"], ci95=[lo, hi])
    print(f"  test gain of LoRA over the baseline: {out['test_gain_lora_over_lr']['point']:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")

    # 3. confidence: the merged class collects the split probability, so more complaints become confident
    conf, ok = lora_va.max(1), lora_va.argmax(1) == y_va
    conf26, ok26 = apply_bias(np.load("data/lora_colab/proba_val.npy"), bias).max(1), p26 == y26_va
    out["threshold_val"] = []
    print("\nconfidence threshold   answered alone (26 -> 24 classes)   accuracy there (26 -> 24)")
    for th in (0.7, 0.8, 0.9):
        a, b = conf26 >= th, conf >= th
        out["threshold_val"].append(dict(t=th, handled_26=float(a.mean()), accuracy_26=float(ok26[a].mean()),
                                         handled_24=float(b.mean()), accuracy_24=float(ok[b].mean())))
        print(f"  >= {th}                {a.mean():6.1%} -> {b.mean():6.1%}                          {ok26[a].mean():.3f} -> {ok[b].mean():.3f}")
    top3 = np.argsort(-lora_va, axis=1)[:, :3]
    unsure = conf < 0.8
    out["unsure_val"] = dict(share=float(unsure.mean()), top1=float(ok[unsure].mean()),
                             in_top3=float(np.mean([y_va[i] in top3[i] for i in np.flatnonzero(unsure)])))
    print(f"\nunsure (< 0.8) after the merge: {unsure.mean():.1%} of complaints; top guess right {ok[unsure].mean():.1%}; "
          f"right label in the top three {out['unsure_val']['in_top3']:.1%}")
    json.dump(out, open(R / "taxonomy_eval.json", "w"), indent=2)


if __name__ == "__main__":
    main()
