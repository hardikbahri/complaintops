"""Unit tests for the evaluation gate (synthetic predictions, no MLflow, no model files)."""
import numpy as np
from sklearn.metrics import f1_score

from complaintops.evaluate import (RECALL_FLOOR, golden_regressions, interval, per_class_recall,
                                   run_gate, weighted_f1)

K = 5
CLASSES = [f"c{i}" for i in range(K)]


def make_data(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.choice(K, size=n, p=[0.5, 0.2, 0.15, 0.1, 0.05])
    return y, rng


def noisy(y, rng, acc):
    """Predictions that are right with probability `acc`, otherwise a random wrong class."""
    pred = y.copy()
    wrong = rng.random(len(y)) > acc
    pred[wrong] = (y[wrong] + rng.integers(1, K, wrong.sum())) % K
    return pred


def test_weighted_f1_matches_sklearn():
    y, rng = make_data()
    pred = noisy(y, rng, 0.7)
    assert abs(weighted_f1(y, pred, K) - f1_score(y, pred, average="weighted")) < 1e-9


def test_equal_models_pass_and_bad_model_is_blocked():
    y, rng = make_data()
    champ = noisy(y, rng, 0.75)
    same = champ.copy()
    bad = noisy(y, rng, 0.45)
    assert run_gate(y, champ, same, CLASSES, n_boot=200)["passed"]
    report = run_gate(y, champ, bad, CLASSES, n_boot=200)
    assert not report["passed"]
    assert not report["checks"][0]["passed"]  # failed on the lower bound


def test_clearly_better_model_passes():
    y, rng = make_data()
    champ = noisy(y, rng, 0.6)
    better = noisy(y, rng, 0.85)
    report = run_gate(y, champ, better, CLASSES, n_boot=200)
    assert report["passed"]
    assert report["difference"]["ci95"][0] > 0  # the improvement is real, not noise


def test_collapsed_class_is_blocked_even_if_overall_score_is_fine():
    y, rng = make_data()
    champ = noisy(y, rng, 0.8)
    cand = champ.copy()
    cand[y == 4] = 0  # never predicts the smallest class (5% of rows): overall F1 barely moves
    assert per_class_recall(y, cand, K)[4] < RECALL_FLOOR
    report = run_gate(y, champ, cand, CLASSES, n_boot=200)
    floor = [c for c in report["checks"] if c["name"] == "class recall floor"][0]
    assert not floor["passed"] and not report["passed"]


def test_golden_set_regression_blocks():
    y, rng = make_data()
    champ = noisy(y, rng, 0.8)
    cand = champ.copy()
    g_y = np.array([0, 1, 2, 3])
    g_champ = np.array([0, 1, 2, 3])   # champion gets all four right
    g_cand = np.array([0, 1, 4, 3])    # candidate breaks case 2
    assert golden_regressions(g_y, g_champ, g_cand).tolist() == [2]
    assert not run_gate(y, champ, cand, CLASSES, golden=(g_y, g_champ, g_cand), n_boot=100)["passed"]
    assert run_gate(y, champ, cand, CLASSES, golden=(g_y, g_champ, g_champ), n_boot=100)["passed"]


def test_interval_brackets_the_estimate():
    y, rng = make_data()
    pred = noisy(y, rng, 0.7)
    from complaintops.evaluate import bootstrap
    lo, hi = interval(bootstrap(y, {"m": pred}, K, n=300)["m"])
    assert lo < weighted_f1(y, pred, K) < hi
