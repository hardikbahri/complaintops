# ComplaintOps

Complaint triage platform: train, track, gate, monitor and serve models, then put an agent in front of them.

Status: Milestone 1 in progress (model side).

## Quick start
```
uv sync
uv run pytest
```

## Results (fill in as you go)

### Splits (Step 2)
| Split | Rows | Date range | Classes |
|---|---|---|---|
| train | | | |
| validation | | | |
| test | | | |

### Model comparison (Steps 3, 4, 6)
| Model | Val macro-F1 | Train time | Notes |
|---|---|---|---|
| TF-IDF + logistic regression | | | |
| XGBoost | | | |
| DistilBERT + LoRA | | | trainable params: ; adapter size: |

### Error analysis (Step 3)
(five lines on why the baseline fails)

### Evaluation gate (Step 7)
(one blocked run, one promoted run)

### Drift (Step 8)
(normal batch vs shifted batch)

### Serving benchmark (Step 9)
| Variant | Size (MB) | p50 ms | p95 ms | Macro-F1 |
|---|---|---|---|---|
| PyTorch fp32 | | | | |
| ONNX fp32 | | | | |
| ONNX int8 | | | | |
