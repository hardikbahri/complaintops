"""The label taxonomy: 26 original classes, or 24 with the three overlapping credit-reporting issues merged.

"Incorrect information on your report", "Problem with a credit reporting company's investigation" and "Improper use
of your report" are chosen by consumers from a form and describe nearly the same story, so the text cannot tell
them apart. Merging them is a task change, not a model improvement: every result on the merged taxonomy must be
labelled as such. The merge is applied after the model, by adding the three classes' probabilities, so the same
trained model serves both taxonomies.
"""
import numpy as np

_P = "Credit reporting | "
MERGED_LABEL = _P + "Dispute, accuracy or use of report"
MERGE_MAP = {
    _P + "Incorrect information on your report": MERGED_LABEL,
    _P + "Problem with a credit reporting company's investigation into an existing problem": MERGED_LABEL,
    _P + "Improper use of your report": MERGED_LABEL,
}


def merge_label(label):
    return MERGE_MAP.get(label, label)


def merged_classes(classes):
    """The merged class list, in a stable order: each class keeps its place; the merged one takes the first slot."""
    out = []
    for c in classes:
        m = merge_label(c)
        if m not in out:
            out.append(m)
    return out


def merge_proba(proba, classes):
    """Add the probabilities of merged classes. proba: (n, len(classes)); returns (n, len(merged_classes))."""
    new = merged_classes(classes)
    index = {c: i for i, c in enumerate(new)}
    out = np.zeros((proba.shape[0], len(new)), dtype=proba.dtype)
    for j, c in enumerate(classes):
        out[:, index[merge_label(c)]] += proba[:, j]
    return out
