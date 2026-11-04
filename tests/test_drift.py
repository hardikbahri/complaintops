"""Drift monitor: the normal batch must pass quietly and shifted batches must fire an alert."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from complaintops.drift import check, make_batches, psi, scored_frames

CLASSES = np.array(["a", "b", "c", "d"])
VOCAB_A = [f"alpha{i}" for i in range(300)]
VOCAB_B = [f"beta{i}" for i in range(300)]


def synthetic(n, rng, vocab, p_class, conf):
    texts = [" ".join(rng.choice(vocab, 40)) for _ in range(n)]
    return pd.DataFrame({
        "n_chars": [len(t) for t in texts], "n_words": 40, "n_xxxx": rng.poisson(2, n),
        "n_dollar": rng.poisson(0.5, n), "conf": np.clip(rng.normal(conf, 0.1, n), 0, 1),
        "pred": rng.choice(CLASSES, n, p=p_class), "narrative": texts})


def test_psi_is_zero_for_identical_data_and_large_for_a_shifted_mix():
    rng = np.random.default_rng(0)
    a = rng.choice(CLASSES, 5000, p=[0.5, 0.3, 0.1, 0.1])
    assert psi(a, a) < 1e-9
    assert psi(a, rng.choice(CLASSES, 5000, p=[0.05, 0.05, 0.1, 0.8])) > 0.25


def test_synthetic_normal_batch_passes_and_shifted_batches_fire():
    rng = np.random.default_rng(1)
    mix = [0.5, 0.3, 0.1, 0.1]
    ref = synthetic(3000, rng, VOCAB_A, mix, 0.72)
    normal = check(ref, synthetic(1000, rng, VOCAB_A, mix, 0.72))
    assert normal["status"] == "ok", normal

    new_topics = check(ref, synthetic(1000, rng, VOCAB_B, mix, 0.72))   # different wording
    assert new_topics["status"] == "alert" and new_topics["metrics"]["text_auc"] > 0.9
    new_mix = check(ref, synthetic(1000, rng, VOCAB_A, [0.05, 0.05, 0.1, 0.8], 0.72))  # different class mix
    assert new_mix["status"] == "alert" and new_mix["metrics"]["psi_pred"] > 0.25
    lost_confidence = check(ref, synthetic(1000, rng, VOCAB_A, mix, 0.45))  # model much less sure
    assert lost_confidence["status"] == "alert" and lost_confidence["metrics"]["psi_conf"] > 0.25


DATA_FILES = ["data/processed/val.parquet", "data/processed/test.parquet", "data/lora_colab/proba_val.npy",
              "data/lora_colab/proba_test_bias.npy", "results/lora_bias_weighted.json"]


@pytest.mark.skipif(not all(Path(f).exists() for f in DATA_FILES), reason="model outputs not available")
def test_real_batches_normal_passes_shifted_fire():
    val, test = scored_frames()
    batches = make_batches(val, test)
    ref = batches.pop("reference")
    assert check(ref, batches["normal"])["status"] == "ok"
    assert check(ref, batches["shift_mix"])["status"] == "alert"
    assert check(ref, batches["shift_time"])["status"] == "alert"
