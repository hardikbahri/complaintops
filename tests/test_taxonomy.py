"""The merged taxonomy: three labels become one, probabilities add up, nothing else changes."""
import json
from pathlib import Path

import numpy as np
import pytest

from complaintops.taxonomy import MERGE_MAP, MERGED_LABEL, merge_label, merge_proba, merged_classes

CLASSES = ["A | one", *MERGE_MAP, "Z | last"]


def test_three_labels_become_one_and_the_rest_are_untouched():
    new = merged_classes(CLASSES)
    assert len(new) == len(CLASSES) - 2
    assert new.count(MERGED_LABEL) == 1 and "A | one" in new and "Z | last" in new
    assert all(merge_label(c) == MERGED_LABEL for c in MERGE_MAP)
    assert merge_label("A | one") == "A | one"


def test_merged_probabilities_add_up_and_still_sum_to_one():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(len(CLASSES)), size=50)
    m = merge_proba(p, CLASSES)
    new = merged_classes(CLASSES)
    assert m.shape == (50, len(new)) and np.allclose(m.sum(axis=1), 1.0)
    trio = [CLASSES.index(c) for c in MERGE_MAP]
    assert np.allclose(m[:, new.index(MERGED_LABEL)], p[:, trio].sum(axis=1))
    assert np.allclose(m[:, new.index("A | one")], p[:, 0])


@pytest.mark.skipif(not Path("data/lora_colab/classes.json").exists(), reason="model outputs not available")
def test_the_real_class_list_goes_from_26_to_24():
    classes = json.load(open("data/lora_colab/classes.json"))
    assert len(classes) == 26 and len(merged_classes(classes)) == 24
    assert set(MERGE_MAP) <= set(classes)
