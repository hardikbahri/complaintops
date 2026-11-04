"""Metrics, bootstrap confidence intervals and the evaluation gate. Step 7.

Headline metric: weighted F1 (per-class F1 weighted by class size). The gate blocks a candidate when
  1. the 95% lower bound of (candidate - champion) weighted F1 is below -MARGIN. Both models are scored on
     the same bootstrap resamples, so test-set noise cancels and an identical model always passes,
  2. any class has recall under RECALL_FLOOR (catches a class silently collapsing), or
  3. any golden-set complaint the champion gets right is now wrong (regression).
All functions here are pure (numpy in, dict out) so they can be unit-tested without MLflow.
"""
import numpy as np

MARGIN = 0.005       # candidate's lower bound may be at most 0.5 points below the champion
RECALL_FLOOR = 0.10  # every class must be found at least this often
N_BOOT = 1000


def weighted_f1(y, pred, k):
    """Weighted F1 with F1_c = 2*TP / (n_true + n_pred); weights = n_true. Equals sklearn's 'weighted'."""
    tp = np.bincount(y[pred == y], minlength=k)
    n_true = np.bincount(y, minlength=k)
    denom = n_true + np.bincount(pred, minlength=k)
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return float((n_true * f1).sum() / len(y))


def per_class_recall(y, pred, k):
    tp = np.bincount(y[pred == y], minlength=k)
    n_true = np.bincount(y, minlength=k)
    return np.where(n_true > 0, tp / np.maximum(n_true, 1), np.nan)


def bootstrap(y, preds, k, n=N_BOOT, seed=0):
    """Resample the test rows n times; score every model on the SAME resample (paired).

    preds: {name: predicted-class array}. Returns {name: array of n weighted-F1 scores}.
    """
    rng = np.random.default_rng(seed)
    out = {name: np.empty(n) for name in preds}
    for i in range(n):
        idx = rng.integers(0, len(y), len(y))
        for name, p in preds.items():
            out[name][i] = weighted_f1(y[idx], p[idx], k)
    return out


def interval(scores, level=95):
    lo, hi = np.percentile(scores, [(100 - level) / 2, 100 - (100 - level) / 2])
    return float(lo), float(hi)


def golden_regressions(golden_y, champ_pred, cand_pred):
    """Indices of golden cases the champion gets right but the candidate gets wrong."""
    return np.flatnonzero((champ_pred == golden_y) & (cand_pred != golden_y))


def run_gate(y, champ_pred, cand_pred, classes, golden=None, n_boot=N_BOOT, seed=0):
    """Apply the three gate rules. golden = (golden_y, champ_golden_pred, cand_golden_pred) or None."""
    k = len(classes)
    boots = bootstrap(y, {"champion": champ_pred, "candidate": cand_pred}, k, n_boot, seed)
    champ_f1, cand_f1 = weighted_f1(y, champ_pred, k), weighted_f1(y, cand_pred, k)
    cand_lo, cand_hi = interval(boots["candidate"])
    champ_lo, champ_hi = interval(boots["champion"])
    diff_lo, diff_hi = interval(boots["candidate"] - boots["champion"])

    checks = []
    checks.append(dict(
        name="lower bound of the difference", passed=bool(diff_lo >= -MARGIN),
        detail=f"95% lower bound of (candidate - champion) = {diff_lo:+.4f} must be >= -{MARGIN}"))

    recall = per_class_recall(y, cand_pred, k)
    low = [(classes[i], float(recall[i])) for i in range(k) if recall[i] < RECALL_FLOOR]
    checks.append(dict(
        name="class recall floor", passed=not low,
        detail=f"lowest class recall {np.nanmin(recall):.3f} (floor {RECALL_FLOOR}); "
               + (f"below floor: {low}" if low else "no class below floor")))

    if golden is not None:
        g_y, g_champ, g_cand = golden
        bad = golden_regressions(g_y, g_champ, g_cand)
        checks.append(dict(
            name="golden set", passed=len(bad) == 0,
            detail=f"{len(g_y) - len(bad)}/{len(g_y)} golden cases kept; regressions at rows {bad.tolist()}"))

    return dict(
        champion=dict(weighted_f1=champ_f1, ci95=[champ_lo, champ_hi]),
        candidate=dict(weighted_f1=cand_f1, ci95=[cand_lo, cand_hi]),
        difference=dict(point=cand_f1 - champ_f1, ci95=[diff_lo, diff_hi]),
        checks=checks, passed=all(c["passed"] for c in checks), n_boot=n_boot)


def format_report(report):
    """Human-readable gate output (this is what goes in the README)."""
    c, d = report["champion"], report["candidate"]
    lines = [
        f"champion  weighted F1 {c['weighted_f1']:.4f}  95% CI [{c['ci95'][0]:.4f}, {c['ci95'][1]:.4f}]",
        f"candidate weighted F1 {d['weighted_f1']:.4f}  95% CI [{d['ci95'][0]:.4f}, {d['ci95'][1]:.4f}]",
        f"difference {report['difference']['point']:+.4f}  95% CI "
        f"[{report['difference']['ci95'][0]:+.4f}, {report['difference']['ci95'][1]:+.4f}]  "
        f"({report['n_boot']} paired bootstrap resamples)",
    ]
    for ch in report["checks"]:
        lines.append(f"  [{'PASS' if ch['passed'] else 'FAIL'}] {ch['name']}: {ch['detail']}")
    lines.append("GATE: " + ("PASSED, candidate may be promoted" if report["passed"] else "BLOCKED"))
    return "\n".join(lines)
