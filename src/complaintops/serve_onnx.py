"""Serve the champion with ONNX Runtime only (no PyTorch): predict(texts) -> label and confidence. Step 9.

Needs data/onnx/ (from export_onnx.py): model_int8.onnx, tokenizer/tokenizer.json, classes.json, bias.json.
The per-class offsets in bias.json are added to the logits before the softmax, exactly as in registered model v4.
    from complaintops.serve_onnx import predict
    predict(["A collector keeps calling me about a debt that is not mine."])
    -> [{"label": "Debt collection | Attempts to collect debt not owed", "confidence": 0.71}]
"""
import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from complaintops.taxonomy import merge_proba, merged_classes

ONNX_DIR = Path("data/onnx")
MAX_LEN = 256


class Classifier:
    def __init__(self, model="model_int8.onnx", onnx_dir=ONNX_DIR, threads=None, taxonomy=None):
        """taxonomy: "merged" (24 classes, the default) or "full" (the 26 original classes); env TAXONOMY sets it."""
        onnx_dir = Path(onnx_dir)
        self.merged = (taxonomy or os.environ.get("TAXONOMY", "merged")) == "merged"
        self.tok = Tokenizer.from_file(str(onnx_dir / "tokenizer" / "tokenizer.json"))
        self.tok.enable_truncation(max_length=MAX_LEN)
        self.tok.enable_padding(pad_id=self.tok.token_to_id("[PAD]"), pad_token="[PAD]")  # to the longest in a batch
        self.full_classes = json.load(open(onnx_dir / "classes.json"))
        self.classes = merged_classes(self.full_classes) if self.merged else self.full_classes
        self.bias = np.array(json.load(open(onnx_dir / "bias.json"))["bias"], dtype="float32")
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(onnx_dir / model), opts, providers=["CPUExecutionProvider"])

    def probabilities(self, texts, batch_size=16):
        """Class probabilities, one row per text (rows sum to 1, columns follow self.classes)."""
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok.encode_batch(list(texts[i:i + batch_size]))
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            logits = self.session.run(None, {"input_ids": ids, "attention_mask": mask})[0] + self.bias
            z = np.exp(logits - logits.max(axis=1, keepdims=True))
            p = z / z.sum(axis=1, keepdims=True)
            out.append(merge_proba(p, self.full_classes) if self.merged else p)
        return np.concatenate(out) if out else np.zeros((0, len(self.classes)), dtype="float32")

    def predict(self, texts, batch_size=16, top_k=1):
        """[{"label": predicted class, "confidence": its probability}] for each text, in order.

        With top_k > 1 each result also has "alternatives": the top_k most likely classes (best first).
        """
        out = []
        for p in self.probabilities(texts, batch_size):
            best = np.argsort(-p)[:top_k]
            rec = {"label": self.classes[best[0]], "confidence": float(p[best[0]])}
            if top_k > 1:
                rec["alternatives"] = [{"label": self.classes[j], "confidence": float(p[j])} for j in best]
            out.append(rec)
        return out


_default = None


def predict(texts, top_k=1):
    """Module-level convenience: loads the int8 model once, on first use."""
    global _default
    if _default is None:
        _default = Classifier()
    return _default.predict(texts, top_k=top_k)
