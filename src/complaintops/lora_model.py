"""Package the LoRA adapter as an MLflow model that predicts from raw text, and register it. Step 7.

MLflow cannot load a bare adapter, so this wraps DistilBERT + adapter in a `pyfunc` model:
input = DataFrame with a `text` column, output = class probabilities (rows = texts, columns = classes).
Optionally applies per-class offsets (from tune_bias.py) to the scores before the softmax.
Register from the project root:
    python -m complaintops.lora_model                                      # plain adapter
    python -m complaintops.lora_model --bias results/lora_bias_weighted.json   # adapter + offsets
"""
import argparse
import json

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd

from complaintops.tracking import EXPERIMENT, TRACKING_URI, git_commit

BASE = "distilbert-base-uncased"
NAME = "complaint-classifier"
MAX_LEN = 256


class LoraClassifier(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.classes = json.load(open(context.artifacts["classes"]))
        self.tok = AutoTokenizer.from_pretrained(BASE)
        base = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=len(self.classes))
        self.model = PeftModel.from_pretrained(base, context.artifacts["adapter"]).eval()
        self.torch = torch
        self.bias = None
        if "bias" in context.artifacts:  # per-class offsets, added to the scores before the softmax
            b = json.load(open(context.artifacts["bias"]))
            assert b["classes"] == self.classes
            self.bias = torch.tensor(b["bias"], dtype=torch.float32)

    def predict(self, context, model_input, params=None):
        texts = model_input["text"].tolist()
        order = np.argsort([len(t) for t in texts])  # similar lengths together = less padding
        out = np.zeros((len(texts), len(self.classes)), dtype="float32")
        with self.torch.no_grad():
            for i in range(0, len(texts), 32):
                j = order[i:i + 32]
                enc = self.tok([texts[k] for k in j], truncation=True, max_length=MAX_LEN,
                               padding=True, return_tensors="pt")
                logits = self.model(**enc).logits
                if self.bias is not None:
                    logits = logits + self.bias
                out[j] = self.torch.softmax(logits, dim=-1).numpy()
        return pd.DataFrame(out, columns=self.classes)


def apply_bias(proba, bias):
    """softmax(log p + b): what the model with offsets outputs, computed from saved probabilities."""
    z = np.log(np.clip(proba, 1e-9, 1.0)) + np.asarray(bias)
    z = np.exp(z - z.max(axis=1, keepdims=True))
    return (z / z.sum(axis=1, keepdims=True)).astype("float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bias", default=None, help="JSON file of per-class offsets to bake into the model")
    args = ap.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    m = json.load(open("data/lora_colab/lora_metrics.json"))
    artifacts = {"adapter": "data/lora_colab/adapter", "classes": "data/lora_colab/classes.json"}
    proba_path = "data/lora_colab/proba_test.npy"
    run_name = "lora_distilbert_full_colab_model"
    if args.bias:
        artifacts["bias"] = args.bias
        run_name = "lora_distilbert_full_colab_bias_model"
        # saved test probabilities for this version, derived from the plain model's saved ones
        proba_path = "data/lora_colab/proba_test_bias.npy"
        bias = json.load(open(args.bias))["bias"]
        np.save(proba_path, apply_bias(np.load("data/lora_colab/proba_test.npy"), bias))

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({"git_commit": git_commit(), "stage": "candidate"})
        mlflow.log_params({"model": "distilbert-base-uncased+lora", "rank": m["args"]["rank"],
                           "epochs": m["args"]["epochs"], "n_classes": 26,
                           "bias_offsets": args.bias or "none"})
        mlflow.log_metrics({"val_weighted_f1": m["val_weighted_f1"], "val_accuracy": m["val_accuracy"]})
        info = mlflow.pyfunc.log_model(
            name="model", python_model=LoraClassifier(), artifacts=artifacts,
            input_example=pd.DataFrame({"text": ["I keep getting calls about a debt that is not mine."]}),
            pip_requirements=["torch", "transformers", "peft", "pandas", "numpy"])
        version = mlflow.register_model(info.model_uri, NAME)
    client = mlflow.MlflowClient()
    client.set_model_version_tag(NAME, version.version, "test_proba_path", proba_path)
    print(f"registered {NAME} v{version.version} (run {run.info.run_id})")


if __name__ == "__main__":
    main()
