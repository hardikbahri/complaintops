"""The ONNX serving function: right output format, same tokens as the library tokenizer, golden set kept."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from complaintops.serve_onnx import Classifier

pytestmark = pytest.mark.skipif(not Path("data/onnx/model_int8.onnx").exists(), reason="run export_onnx first")


def test_predict_returns_a_label_and_a_confidence_for_every_text():
    clf = Classifier()
    out = clf.predict(["A collector keeps calling me about a debt that is not mine.",
                       "There is a wrong account on my credit report."])
    assert len(out) == 2
    for r in out:
        assert r["label"] in clf.classes and 0.0 < r["confidence"] <= 1.0
    assert clf.predict([]) == []


TEXTS = ["short one", "I have a much longer complaint about my mortgage servicer " * 10, "another"]


def batched_vs_single(clf):
    p_all = clf.probabilities(TEXTS, batch_size=16)
    p_one = np.concatenate([clf.probabilities([t], batch_size=1) for t in TEXTS])
    return p_all, p_one


def test_fp32_answers_do_not_depend_on_batching_or_padding():
    p_all, p_one = batched_vs_single(Classifier(model="model_fp32.onnx"))
    assert np.allclose(p_all.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(p_all, p_one, atol=1e-5)  # padding must not change the answer at all


def test_int8_answers_barely_depend_on_batching():
    # Dynamic int8 quantization computes activation scales per batch, so batch contents change the
    # result slightly (about 0.04 in probability here). The predicted classes must still agree.
    p_all, p_one = batched_vs_single(Classifier())
    wobble = np.abs(p_all - p_one).max(axis=1)
    assert wobble.max() < 0.1
    # A class can only flip when the top two were closer than the wobble (a near-tie); clear winners must not change.
    top2 = np.sort(p_all, axis=1)[:, -2:]
    clear = (top2[:, 1] - top2[:, 0]) > 2 * wobble
    assert (p_all.argmax(1)[clear] == p_one.argmax(1)[clear]).all()


def test_tokenizer_matches_the_library_tokenizer():
    from transformers import AutoTokenizer

    clf = Classifier()
    hf = AutoTokenizer.from_pretrained("data/onnx/tokenizer")
    for t in ["I never received notification of the debt.", "XXXX XXXX charged me $50 on XX/XX/2023!"]:
        assert clf.tok.encode(t).ids == hf(t, truncation=True, max_length=256)["input_ids"]


def test_int8_model_keeps_most_of_the_golden_set():
    from complaintops.taxonomy import merge_label

    golden = pd.read_csv("tests/golden_set.csv")  # labelled with the 26 original classes
    pred = [r["label"] for r in Classifier().predict(golden["text"].tolist())]
    kept = float(np.mean(np.array(pred) == golden["label"].map(merge_label).to_numpy()))
    assert kept >= 0.9, f"int8 model keeps only {kept:.0%} of the golden set"
