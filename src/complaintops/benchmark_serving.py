"""Benchmark PyTorch fp32 vs ONNX fp32 vs ONNX int8 for serving. Step 9.

Latency: 500 validation complaints, batch size 1 and 16, end to end (tokenizing + model), CPU only, with 2
threads (what a small free host gives) and with the default thread count. Quality: weighted F1 and agreement
with PyTorch fp32 on 3,000 validation complaints, with the champion's per-class offsets applied to every
variant. Validation only. Needs data/onnx/ from export_onnx.py.
Run from the project root:  python -m complaintops.benchmark_serving
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from complaintops.evaluate import weighted_f1
from complaintops.taxonomy import merge_label, merge_proba, merged_classes

ONNX = Path("data/onnx")
MAX_LEN = 256
N_LATENCY, N_QUALITY = 500, 3000


def build_runners(threads):
    """Return {variant: f(texts) -> logits} and the file size (MB) of each variant."""
    if threads:
        torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(ONNX / "tokenizer")
    pt = AutoModelForSequenceClassification.from_pretrained(ONNX / "merged", attn_implementation="eager").eval()

    def run_pt(texts):
        enc = tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        with torch.no_grad():
            return pt(**enc).logits.numpy()

    def make_ort(path):
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

        def run(texts):
            enc = tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="np")
            return sess.run(None, {"input_ids": enc["input_ids"].astype(np.int64),
                                   "attention_mask": enc["attention_mask"].astype(np.int64)})[0]
        return run

    runners = {"PyTorch fp32": run_pt, "ONNX fp32": make_ort(ONNX / "model_fp32.onnx"),
               "ONNX int8": make_ort(ONNX / "model_int8.onnx")}
    sizes = {"PyTorch fp32": (ONNX / "merged" / "model.safetensors").stat().st_size / 1e6,
             "ONNX fp32": (ONNX / "model_fp32.onnx").stat().st_size / 1e6,
             "ONNX int8": (ONNX / "model_int8.onnx").stat().st_size / 1e6}
    return runners, sizes


def latency(run, texts, batch):
    for i in range(3):  # warm-up
        run(texts[i * batch:(i + 1) * batch])
    times = []
    for i in range(0, len(texts) - batch + 1, batch):
        t0 = time.perf_counter()
        run(texts[i:i + batch])
        times.append((time.perf_counter() - t0) * 1000)
    return dict(p50_ms=float(np.percentile(times, 50)), p95_ms=float(np.percentile(times, 95)),
                n_batches=len(times))


def quality(run, texts, y, bias, k, full_classes):
    """Weighted F1 (offsets applied) and the predictions, in the original order."""
    order = np.argsort([len(t) for t in texts])  # sorted by length: far less padding, same results
    logits = np.zeros((len(texts), k), dtype="float32")
    for i in range(0, len(texts), 32):
        j = order[i:i + 32]
        logits[j] = run([texts[x] for x in j])
    z = np.exp((logits + bias) - (logits + bias).max(1, keepdims=True))
    pred = merge_proba(z / z.sum(1, keepdims=True), full_classes).argmax(1)  # the 24 served classes
    return weighted_f1(y, pred, len(merged_classes(full_classes))), pred


def main():
    import mlflow

    from complaintops.tracking import EXPERIMENT, TRACKING_URI, git_commit

    classes = json.load(open(ONNX / "classes.json"))
    bias = np.array(json.load(open(ONNX / "bias.json"))["bias"], dtype="float32")
    va = pd.read_parquet("data/processed/val.parquet")
    served = merged_classes(classes)
    y_all = va["label"].map(lambda s: served.index(merge_label(s))).to_numpy()
    rng = np.random.default_rng(7)
    lat_idx = rng.choice(len(va), N_LATENCY, replace=False)
    qual_idx = rng.choice(len(va), N_QUALITY, replace=False)
    lat_texts = va["narrative"].iloc[lat_idx].tolist()
    q_texts, q_y = va["narrative"].iloc[qual_idx].tolist(), y_all[qual_idx]

    quality_only = "--quality-only" in sys.argv  # keep the saved timings, redo only the F1 comparison
    results = json.load(open("results/serving_benchmark.json"))["latency"] if quality_only else {}
    for label, threads in (() if quality_only else (("2 threads", 2), ("default threads", None))):
        runners, sizes = build_runners(threads)
        print(f"\n=== latency, {label} ===", flush=True)
        for name, run in runners.items():
            r = {"size_mb": sizes[name]}
            for b in (1, 16):
                r[f"bs{b}"] = latency(run, lat_texts, b)
            results[f"{label} | {name}"] = r
            print(f"{name:13s} size {sizes[name]:6.1f} MB | batch 1: p50 {r['bs1']['p50_ms']:.0f} ms, "
                  f"p95 {r['bs1']['p95_ms']:.0f} ms | batch 16: p50 {r['bs16']['p50_ms']:.0f} ms, "
                  f"p95 {r['bs16']['p95_ms']:.0f} ms", flush=True)

    print(f"\n=== quality on {N_QUALITY} validation complaints ===", flush=True)
    runners, _ = build_runners(None)
    quality_res, preds = {}, {}
    for name, run in runners.items():
        wf1, preds[name] = quality(run, q_texts, q_y, bias, len(classes), classes)
        quality_res[name] = wf1
    for name in runners:
        agree = float((preds[name] == preds["PyTorch fp32"]).mean())
        quality_res[name] = dict(weighted_f1=quality_res[name], agreement_with_pytorch=agree)
        print(f"{name:13s} weighted F1 {quality_res[name]['weighted_f1']:.4f}   same class as PyTorch fp32: {agree:.1%}")

    out = dict(latency=results, quality=quality_res, n_latency=N_LATENCY, n_quality=N_QUALITY)
    Path("results").mkdir(exist_ok=True)
    json.dump(out, open("results/serving_benchmark.json", "w"), indent=2)
    if quality_only:
        return

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="serving_benchmark"):
        mlflow.set_tags({"git_commit": git_commit(), "stage": "serving"})
        for key, r in results.items():
            tag = key.replace(" | ", "_").replace(" ", "_")
            mlflow.log_metrics({f"{tag}_size_mb": r["size_mb"], f"{tag}_bs1_p50_ms": r["bs1"]["p50_ms"],
                                f"{tag}_bs1_p95_ms": r["bs1"]["p95_ms"], f"{tag}_bs16_p50_ms": r["bs16"]["p50_ms"],
                                f"{tag}_bs16_p95_ms": r["bs16"]["p95_ms"]})
        for name, q in quality_res.items():
            tag = name.replace(" ", "_")
            mlflow.log_metrics({f"{tag}_val_weighted_f1": q["weighted_f1"],
                                f"{tag}_agreement": q["agreement_with_pytorch"]})
        mlflow.log_artifact("results/serving_benchmark.json")


if __name__ == "__main__":
    main()
