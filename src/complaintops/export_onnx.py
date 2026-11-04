"""Merge the LoRA adapter into DistilBERT, export to ONNX, quantize to int8. Step 9.

Writes data/onnx/: merged/ (plain PyTorch model), tokenizer/, model_fp32.onnx, model_int8.onnx,
classes.json, bias.json. The champion's per-class offsets are NOT baked into the network: serve_onnx.py adds
them to the logits, exactly as the registered model does.
Run from the project root:  python -m complaintops.export_onnx
"""
import json
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE = "distilbert-base-uncased"
OUT = Path("data/onnx")
MAX_LEN = 256


class LogitsOnly(torch.nn.Module):
    """The same forward pass as DistilBertForSequenceClassification, written out for ONNX export.

    The library builds the padding mask with Python control flow that the exporter freezes at the sample
    length, which gives wrong answers for other lengths. Here the mask is plain tensor arithmetic:
    0 for real tokens and a very large negative number for padding, added to the attention scores.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        m = self.model
        x = m.distilbert.embeddings(input_ids)
        mask = (1.0 - attention_mask[:, None, None, :].to(x.dtype)) * torch.finfo(x.dtype).min
        h = m.distilbert.transformer(hidden_states=x, attention_mask=mask).last_hidden_state
        pooled = torch.relu(m.pre_classifier(h[:, 0]))
        return m.classifier(pooled)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    classes = json.load(open("data/lora_colab/classes.json"))

    # 1. Merge: W <- W + (adapter correction). The result is one ordinary network with no adapter layers.
    base = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=len(classes), attn_implementation="eager")
    model = PeftModel.from_pretrained(base, "data/lora_colab/adapter").merge_and_unload().eval()
    model.save_pretrained(OUT / "merged")
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.save_pretrained(OUT / "tokenizer")
    json.dump(classes, open(OUT / "classes.json", "w"))
    shutil.copy("results/lora_bias_weighted.json", OUT / "bias.json")

    # 2. Export with dynamic batch and sequence length.
    texts = pd.read_csv("tests/golden_set.csv")["text"].tolist()
    enc = tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    wrapper = LogitsOnly(model).eval()
    with torch.no_grad():
        ref = model(**enc).logits.numpy()
        mine = wrapper(enc["input_ids"], enc["attention_mask"]).numpy()
    assert np.abs(mine - ref).max() < 1e-4, "hand-written forward pass differs from the library model"
    print(f"hand-written forward pass vs library model: max |logit difference| {np.abs(mine - ref).max():.2e}")

    sample = tok(["I keep getting calls about a debt that is not mine.", "Short one."],
                 padding="max_length", max_length=32, truncation=True, return_tensors="pt")
    fp32 = OUT / "model_fp32.onnx"
    torch.onnx.export(
        wrapper, (sample["input_ids"], sample["attention_mask"]), str(fp32),
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                      "logits": {0: "batch"}},
        opset_version=17, dynamo=False)

    # 3. Dynamic int8 quantization of the weights.
    int8 = OUT / "model_int8.onnx"
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    print(f"fp32 {fp32.stat().st_size / 1e6:.1f} MB   int8 {int8.stat().st_size / 1e6:.1f} MB")

    # 4. Check the exports against PyTorch on 40 real complaints of different lengths.
    for name, path in (("onnx fp32", fp32), ("onnx int8", int8)):
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        out = sess.run(None, {"input_ids": enc["input_ids"].numpy(), "attention_mask": enc["attention_mask"].numpy()})[0]
        agree = float((out.argmax(1) == ref.argmax(1)).mean())
        print(f"{name}: max |logit difference| vs PyTorch {np.abs(out - ref).max():.5f}, "
              f"same predicted class on {agree:.0%} of 40 complaints")
        if name == "onnx fp32":
            assert np.abs(out - ref).max() < 1e-3, "fp32 ONNX export does not match PyTorch"


if __name__ == "__main__":
    main()
