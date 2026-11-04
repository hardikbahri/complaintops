"""Package the LoRA adapter as an MLflow model that predicts from raw text, and register it. Step 7.

MLflow cannot load a bare adapter, so this wraps DistilBERT + adapter in a `pyfunc` model:
input = DataFrame with a `text` column, output = class probabilities (rows = texts, columns = classes).
Register from the project root:  python -m complaintops.lora_model
"""
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

    def predict(self, context, model_input, params=None):
        texts = model_input["text"].tolist()
        order = np.argsort([len(t) for t in texts])  # similar lengths together = less padding
        out = np.zeros((len(texts), len(self.classes)), dtype="float32")
        with self.torch.no_grad():
            for i in range(0, len(texts), 32):
                j = order[i:i + 32]
                enc = self.tok([texts[k] for k in j], truncation=True, max_length=MAX_LEN,
                               padding=True, return_tensors="pt")
                out[j] = self.torch.softmax(self.model(**enc).logits, dim=-1).numpy()
        return pd.DataFrame(out, columns=self.classes)


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    m = json.load(open("data/lora_colab/lora_metrics.json"))
    with mlflow.start_run(run_name="lora_distilbert_full_colab_model") as run:
        mlflow.set_tags({"git_commit": git_commit(), "stage": "candidate"})
        mlflow.log_params({"model": "distilbert-base-uncased+lora", "rank": m["args"]["rank"],
                           "epochs": m["args"]["epochs"], "n_classes": 26})
        mlflow.log_metrics({"val_weighted_f1": m["val_weighted_f1"], "val_accuracy": m["val_accuracy"]})
        info = mlflow.pyfunc.log_model(
            name="model", python_model=LoraClassifier(),
            artifacts={"adapter": "data/lora_colab/adapter", "classes": "data/lora_colab/classes.json"},
            input_example=pd.DataFrame({"text": ["I keep getting calls about a debt that is not mine."]}),
            pip_requirements=["torch", "transformers", "peft", "pandas", "numpy"])
        version = mlflow.register_model(info.model_uri, NAME)
    client = mlflow.MlflowClient()
    client.set_model_version_tag(NAME, version.version, "test_proba_path", "data/lora_colab/proba_test.npy")
    print(f"registered {NAME} v{version.version} (run {run.info.run_id})")


if __name__ == "__main__":
    main()
