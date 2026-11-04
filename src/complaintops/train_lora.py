"""LoRA fine-tune of DistilBERT. Step 6. Run on a GPU (Colab T4 recommended).

Colab (all defaults = the planned run):
    python -m complaintops.train_lora --out /content/drive/MyDrive/lora_run
If Colab disconnects, run the same command again with --resume (continues from the last checkpoint).
Smoke test on CPU:
    python -m complaintops.train_lora --n-train 200 --n-val 64 --epochs 1 --batch-size 16 --max-len 64 --out data/lora_smoke
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

MODEL = "distilbert-base-uncased"


def batches(texts, y, tok, batch_size, max_len, order):
    """Yield (tokenized batch, labels) following `order`. Padding is per batch, to the longest text in it."""
    for i in range(0, len(order), batch_size):
        j = order[i:i + batch_size]
        enc = tok([texts[k] for k in j], truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt")
        yield enc, (torch.tensor(y[j]) if y is not None else None)


@torch.no_grad()
def predict_proba(model, texts, tok, batch_size, max_len, device, amp=True):
    """Class probabilities for every text, in the original order."""
    model.eval()
    # Sort by length so each batch has little padding (much faster); undo the sort at the end.
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), model.config.num_labels), dtype="float32")
    pos = 0
    for enc, _ in batches(texts, None, tok, batch_size, max_len, order):
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp and device == "cuda"):
            logits = model(**enc).logits
        n = logits.shape[0]
        out[order[pos:pos + n]] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        pos += n
    return out


def save_ckpt(path, model, opt, sched, scaler, epoch, batch, step, elapsed):
    """Everything needed to continue training: trainable weights, optimizer, scheduler, position."""
    state = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    tmp = path.with_suffix(".tmp")
    torch.save(dict(model=state, opt=opt.state_dict(), sched=sched.state_dict(),
                    scaler=scaler.state_dict(), epoch=epoch, batch=batch, step=step,
                    elapsed=elapsed), tmp)
    tmp.replace(path)  # replace only after a complete write, so a crash never leaves a broken file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=235_000, help="rows of train to use (more than exist = all)")
    ap.add_argument("--n-val", type=int, default=None, help="use only the first N validation rows (smoke test)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--targets", default="q_lin,k_lin,v_lin,out_lin,lin1,lin2",
                    help="DistilBERT layers that get LoRA adapters (all attention + feed-forward by default)")
    ap.add_argument("--weight-power", type=float, default=0.5,
                    help="class weight = balanced_weight ** power (0 = none, 1 = fully balanced)")
    ap.add_argument("--no-amp", action="store_true", help="train in full fp32 instead of mixed fp16")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/ckpt.pt if it exists")
    ap.add_argument("--ckpt-steps", type=int, default=1000, help="save a checkpoint every N steps")
    ap.add_argument("--test-proba", action="store_true",
                    help="also save probabilities for the test split (not scored here; for later blending/gating)")
    ap.add_argument("--out", default="data/lora_run")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda" and not args.no_amp
    print("device:", device, " mixed precision:", use_amp, flush=True)

    # ---- data: the same processed splits as every other model ----
    tr = pd.read_parquet("data/processed/train.parquet")
    va = pd.read_parquet("data/processed/val.parquet")
    le = LabelEncoder().fit(tr["label"])  # all classes, even if a small sample misses one
    if args.n_train < len(tr):
        tr = tr.sample(args.n_train, random_state=args.seed)
    if args.n_val:
        va = va.iloc[:args.n_val]
    X_tr, y_tr = tr["narrative"].tolist(), le.transform(tr["label"])
    X_va, y_va = va["narrative"].tolist(), le.transform(va["label"])
    print(f"train {len(X_tr)}  val {len(X_va)}  classes {len(le.classes_)}", flush=True)

    # ---- model: DistilBERT + a fresh classification head, then LoRA on top ----
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(le.classes_))
    lora = LoraConfig(task_type=TaskType.SEQ_CLS, r=args.rank, lora_alpha=args.alpha,
                      lora_dropout=args.dropout, target_modules=args.targets.split(","))
    model = get_peft_model(base, lora).to(device)
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total_params = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {n_train_params:,} of {n_total_params:,} "
          f"({100 * n_train_params / n_total_params:.2f}%)", flush=True)

    # ---- loss with softened class weights (fully balanced weights hurt precision earlier) ----
    present = np.unique(y_tr)
    w = np.ones(len(le.classes_), dtype="float32")
    w[present] = compute_class_weight("balanced", classes=present, y=y_tr) ** args.weight_power
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, device=device))

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    steps_per_epoch = -(-len(X_tr) // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "ckpt.pt"

    # ---- resume, if asked and a checkpoint exists ----
    start_epoch, start_batch, step, elapsed0 = 0, 0, 0, 0.0
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        params = dict(model.named_parameters())
        for n, v in ck["model"].items():
            params[n].data.copy_(v)
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, start_batch, step, elapsed0 = ck["epoch"], ck["batch"], ck["step"], ck["elapsed"]
        print(f"RESUMED at epoch {start_epoch + 1}, batch {start_batch}, step {step}", flush=True)
    t0 = time.time() - elapsed0
    check = rng.choice(len(X_va), size=min(3000, len(X_va)), replace=False)  # quick per-epoch progress check

    for epoch in range(start_epoch, args.epochs):
        model.train()
        # Same shuffle for the same epoch number, so a resumed run sees exactly the data it would have.
        order = np.random.default_rng(args.seed + epoch).permutation(len(X_tr))
        skip = start_batch if epoch == start_epoch else 0
        b = skip
        for enc, y in batches(X_tr, y_tr, tok, args.batch_size, args.max_len, order[skip * args.batch_size:]):
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                logits = model(**enc).logits
            loss = loss_fn(logits.float(), y.to(device))
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            b += 1
            if step % 100 == 0:
                print(f"epoch {epoch + 1}  step {step}/{total_steps}  loss {loss.item():.3f}  "
                      f"{time.time() - t0:.0f}s", flush=True)
            if step % args.ckpt_steps == 0:
                save_ckpt(ckpt_path, model, opt, sched, scaler, epoch, b, step, time.time() - t0)
        p = predict_proba(model, [X_va[i] for i in check], tok, args.eval_batch_size, args.max_len, device, use_amp)
        print(f"== epoch {epoch + 1} done; quick check on {len(check)} val rows: "
              f"macro-F1 {f1_score(y_va[check], p.argmax(1), average='macro'):.4f}  "
              f"weighted-F1 {f1_score(y_va[check], p.argmax(1), average='weighted'):.4f}", flush=True)
        model.save_pretrained(out / "adapter")
        save_ckpt(ckpt_path, model, opt, sched, scaler, epoch + 1, 0, step, time.time() - t0)
    train_time = time.time() - t0

    # ---- final evaluation on the whole validation split (test is never scored here) ----
    proba = predict_proba(model, X_va, tok, args.eval_batch_size, args.max_len, device, use_amp)
    pred = proba.argmax(1)
    f1m, f1w, acc = (f1_score(y_va, pred, average="macro"), f1_score(y_va, pred, average="weighted"),
                     accuracy_score(y_va, pred))
    report = classification_report(le.inverse_transform(y_va), le.inverse_transform(pred), zero_division=0)
    print(f"train time {train_time:.0f}s  val weighted-F1 {f1w:.4f}  macro-F1 {f1m:.4f}  accuracy {acc:.4f}")
    print(report)

    adapter_mb = sum(f.stat().st_size for f in (out / "adapter").rglob("*") if f.is_file()) / 1e6
    np.save(out / "proba_val.npy", proba)  # saved so blends / threshold tuning never need retraining
    if args.test_proba:
        te = pd.read_parquet("data/processed/test.parquet")
        np.save(out / "proba_test.npy",
                predict_proba(model, te["narrative"].tolist(), tok, args.eval_batch_size, args.max_len, device, use_amp))
    (out / "classes.json").write_text(json.dumps(list(le.classes_)))
    (out / "lora_report.txt").write_text(report)
    json.dump(dict(val_weighted_f1=f1w, val_macro_f1=f1m, val_accuracy=acc, train_time_s=round(train_time),
                   trainable_params=n_train_params, total_params=n_total_params,
                   adapter_mb=round(adapter_mb, 2), args=vars(args)),
              open(out / "lora_metrics.json", "w"), indent=2)
    print(f"trainable params: {n_train_params:,}   adapter size: {adapter_mb:.2f} MB")


if __name__ == "__main__":
    main()
