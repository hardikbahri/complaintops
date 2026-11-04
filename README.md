# ComplaintOps

Complaint triage platform: train, track, gate, monitor and serve models, then put an agent in front of them.

Status: Milestone 1 in progress (model side).

## Quick start
```
uv sync
uv run pytest
```

## Results (fill in as you go)

### Data (Step 2)
The official CFPB bulk file and API no longer include the complaint narrative, so this project uses
a third-party copy of the CFPB Consumer Complaint Database: Hugging Face `sovai/cfpb_complaints`
(not an official CFPB release). A random 400,000-row sample of rows with a narrative was taken,
keeping only `date`, `product`, `issue`, `submitted_via` and `narrative`; leakage-prone columns
(company response, timely response, ...) are never used.

Cleaning: narratives under 10 characters dropped; exact duplicates dropped after lowercasing and
collapsing whitespace, keeping the earliest (400,000 -> 335,499 rows). Label = `product | issue`.
Renamed products and issues (the CFPB changed its menus around 2017-04-24) were merged to their
modern names; the mortgage and checking-account mappings are approximate. The 25 most common
labels in train are kept, the rest become "Other". Split is by date: oldest 70% / next 15% / newest 15%.

### Splits (Step 2)
| Split | Rows | Date range | Classes |
|---|---|---|---|
| train | 234,849 | 2015-03-19 to 2023-02-13 | 26 (25 + Other) |
| validation | 50,325 | 2023-02-13 to 2023-09-08 | 26 |
| test | 50,325 | 2023-09-08 to 2024-05-01 | 26 |

The biggest class is 28-31% of every split; "Other" is 13.5% of train but 6.8% (val) and 8.5% (test).
The smallest classes have only 113 (val) and 93 (test) rows, so their recall is noisy.
`tests/test_data.py` checks that no narrative appears in two splits.

### Model comparison (Steps 3, 4, 6)
Headline metric: **weighted F1** (each class's F1 weighted by its number of validation rows). Macro-F1
(every class counts equally, so small classes weigh as much as large ones) is reported next to it and is
the stricter measure. The headline was switched from macro to weighted F1 after the first results, so the
two columns are always shown together; all models are scored on both on the same validation split.

| Model | Val weighted F1 | Val macro-F1 | Train time | Notes |
|---|---|---|---|---|
| TF-IDF + logistic regression | 0.6047 | 0.4717 | ~16 min (235k rows) | 5-fold CV macro-F1 0.4635 ± 0.0044 (50k subsample); val accuracy 0.60 |
| XGBoost (Optuna-tuned) | 0.59 | 0.4543 | ~33 min (T4 GPU, 211k rows) | val accuracy 0.59; did not beat TF-IDF + LR; hit the 800-tree cap without early stopping; weighted F1 read from the 2-decimal report |
| TF-IDF + LR, merged labels (19 classes) | 0.8028 | 0.4777 | ~16 min | easier task: one merged class is 74% of validation, so the weighted score is inflated by it; not comparable |
| TF-IDF + LR, product-level only (10 classes) | 0.89 | 0.6296 (0.70 without the 6-row "Other") | ~8 min | easier task, not comparable with the first rows; val accuracy 0.876 |
| DistilBERT + LoRA (first run: 60k of 235k train rows, 2 epochs, rank 8) | 0.6108 | 0.4589 | ~75 min (GTX 1650, fp32) | val accuracy 0.6107; trainable params: 758,042 of 67,731,508 (1.12%); adapter size: 3.04 MB |
| DistilBERT + LoRA (all train rows, 3 epochs, rank 16 on all linear layers) | 0.6433 | 0.5241 | ~92 min (Colab T4, fp16) | val accuracy 0.6428; trainable params: 1,937,690 of 68,911,156 (2.81%); adapter size: 7.77 MB; resumed once after a Colab disconnect |
| LoRA + per-class bias tuning (26 offsets on the log-probabilities) | 0.6455 | **0.5432** | minutes, no training | scores are 5-fold cross-fitted on validation (offsets tuned on other rows); val accuracy 0.6481; same gain on a chronological check (newer half 0.5164 -> 0.5366) |
| LoRA + TF-IDF LR blend (not adopted) | 0.6445 | 0.5287 | minutes | +0.005 macro-F1 over LoRA alone; not worth serving two models |

### XGBoost notes (Step 4)
Features: TF-IDF (20,000 word/bigram features, fitted on train only) plus five numeric features
(`n_chars`, `n_words`, `n_xxxx`, `n_dollar`, `day_of_week`). `submitted_via` was dropped because it is
constant ("Web"); company response and timely response were excluded as leakage. Settings came from a
12-trial Optuna search on a 40k-row subsample, scored on validation (never on test).
XGBoost scored below the linear baseline: on sparse text, linear models are hard to beat.
None of the 15 most important features is a numeric feature, so the engineered features added almost
nothing beyond the text. Precision is low and recall high on most small classes, because balanced
sample weights make the model over-predict rare classes.

### Label-merge experiment (not adopted)
Merging 12 confusable labels into 5 (26 -> 19 classes: three credit-reporting issues, three
debt-collection misconduct issues, two mortgage payment issues, opening/managing a checking account,
two credit-card terms issues) did not help. The original 26-class model scored on the merged task:
macro-F1 0.4801. A TF-IDF + LR model retrained on merged labels: 0.4777. The merged task is easier and
dominated by one class (74% of validation), so those numbers are not comparable with 0.4717.
The weak classes are a long tail of small ones, not the merged ones. The 26-class task stays the main one.

### Product-level task (easier version)
Predicting only the product (9 most common products in train + "Other" = 10 classes) reuses the same
data, splits and leakage checks; only the target changes. TF-IDF + LR reaches macro-F1 0.6296 and
accuracy 0.876. "Other" has 6 validation rows and scores 0, so over the 9 real products macro-F1 is about
0.70. Credit reporting is 59% of train but 75% of validation, and balanced class weights make the model
over-predict small products (e.g. debt collection precision 0.45, recall 0.72). Ambiguity exists at the
product level too: credit-report complaints often mention collectors, auto loans or cards.
The 26-class issue task stays as the "hard version".

### Experiment tracking (Step 5)
All runs are logged to MLflow (`sqlite:///mlflow.db`, experiment `complaintops`) with parameters, metrics,
artifacts and the git commit. View them with
`uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`. The baseline, merged-label, XGBoost and Optuna
runs were back-filled from saved results (`python -m complaintops.log_existing`, tagged
`logged_after_the_fact=true`); later runs are logged live by the training scripts through
`complaintops.tracking.log_run`. The best 26-class model with a registrable artifact is registered as
`complaint-classifier` with alias `champion` (v1: TF-IDF + LR, val macro-F1 0.4717; `complaintops.register`).
A blend of LR and XGBoost was attempted and dropped: the XGBoost model was trained on Colab and its fitted
TF-IDF vectorizer was not saved, and a local refit does not reproduce the same vocabulary (ties at the
`max_features` cutoff), so the model could not be reused.

### Error analysis (Step 3)
Read from 20 random validation errors (12 stay within the right product, 8 cross products)
and the per-class report. Weakest classes: Debt collection | False statements (F1 0.18),
Written notification (0.22), Took or threatened legal action (0.25), and Other (0.28).

1. **The three big credit-reporting issues overlap.** "Incorrect information", "Improper use" and "Problem with the investigation" describe nearly the same story (disputing items on a report), so the model often picks a sibling; some labels are arguable even to a human.
2. **Words from other products appear inside credit-reporting complaints.** Mentions of charge-offs, collections, auto or credit-card accounts pushed predictions to debt collection or vehicle loan.
3. **Heavy redaction removes the signal.** Complaints that are mostly `XXXX` tokens leave almost no vocabulary for TF-IDF.
4. **Adjacent classes within a product** (mortgage "Struggling to pay" vs "Trouble during payment", checking "Opening" vs "Managing") differ by intent, not distinctive words, and bag-of-words cannot tell intent.
5. **"Other" is a grab-bag** with no consistent vocabulary, so its recall is only 0.19.

### Evaluation gate (Step 7)
A candidate model version replaces the `champion` alias only if `python -m complaintops.promote
--candidate-version N` passes three checks, all on the **test** split (scored here for the first time; every
earlier choice was made on validation) plus the golden set. The metric is **weighted F1**.

1. The 95% lower bound of (candidate - champion) weighted F1 must be at least -0.005. Both models are scored
   on the same 1,000 bootstrap resamples (paired), so test-set noise cancels. Comparing the candidate's own
   lower bound with the champion's score blocked even an identical model on a small sample, which a unit test
   caught.
2. No class may have recall under 0.10 (a class silently collapsing). This replaces macro-F1 as the guard for
   small classes.
3. A 40-case golden set (`tests/golden_set.csv`: complaints from the validation split that the baseline and
   LoRA both get right with confidence; all 26 classes, mostly short, heavily redacted or multi-topic) must
   keep passing: any case the champion gets right and the candidate gets wrong blocks the candidate.

`tests/test_gate.py` checks the rules on synthetic predictions (identical model passes, clearly worse model
blocked, collapsed small class blocked, golden regression blocked, better model passes).

**Blocked run**: v2, the baseline pipeline trained on 5% of train (11,742 rows), against champion v1
(TF-IDF + LR):
```
champion  weighted F1 0.5310  95% CI [0.5266, 0.5356]
candidate weighted F1 0.5016  95% CI [0.4970, 0.5063]
difference -0.0294  95% CI [-0.0325, -0.0262]  (1000 paired bootstrap resamples)
  [FAIL] lower bound of the difference: 95% lower bound of (candidate - champion) = -0.0325 must be >= -0.005
  [FAIL] class recall floor: lowest class recall 0.087 (floor 0.1); below floor: [('Debt collection | Written notification about debt', 0.087)]
  [FAIL] golden set: 34/40 golden cases kept; regressions at rows [6, 13, 16, 23, 37, 39]
GATE: BLOCKED
```
**Promoted run**: v3, DistilBERT + LoRA (all train rows, 3 epochs, rank 16), against champion v1:
```
champion  weighted F1 0.5310  95% CI [0.5266, 0.5356]
candidate weighted F1 0.5771  95% CI [0.5724, 0.5817]
difference +0.0460  95% CI [+0.0424, +0.0498]  (1000 paired bootstrap resamples)
  [PASS] lower bound of the difference: 95% lower bound of (candidate - champion) = +0.0424 must be >= -0.005
  [PASS] class recall floor: lowest class recall 0.136 (floor 0.1); no class below floor
  [PASS] golden set: 40/40 golden cases kept; regressions at rows []
GATE: PASSED, candidate may be promoted
alias 'champion' now points to v3
```
Notes: the test split is harder than validation for both models (baseline weighted F1 0.6047 on validation vs
0.5310 on test; LoRA 0.6433 vs 0.5771). Both drop by a similar amount, which points to the later time period
(data drift) and not to overfitting to validation; Step 8 looks at this. The weak 5%-data model is only 2.5
points below the baseline on validation, so the gate is catching a small regression, not an obvious failure.
Saved test probabilities are reused after a spot check against the live registered model (32 rows each,
100% agreement). The blend and the per-class bias tuning were not taken forward: the blend added +0.005 macro-F1,
and the bias tuning was tuned for macro-F1 and adds only +0.002 weighted F1.

**Refinement run**: v4 = v3 plus 26 per-class offsets on the scores, tuned for weighted F1 on the
**validation** split only (5-fold cross-fitted validation gain +0.008; a chronological check, tuned on the older
half and scored on the newer half, +0.010). Gated against champion v3 with `--require-improvement`, which
adds a rule that the lower bound of the difference must be above zero:
```
champion  weighted F1 0.5771  95% CI [0.5724, 0.5817]
candidate weighted F1 0.5869  95% CI [0.5824, 0.5914]
difference +0.0099  95% CI [+0.0079, +0.0117]  (1000 paired bootstrap resamples)
  [PASS] lower bound of the difference: 95% lower bound of (candidate - champion) = +0.0079 must be >= -0.005
  [PASS] must clearly improve: 95% lower bound of the difference +0.0079 must be > 0
  [PASS] class recall floor: lowest class recall 0.101 (floor 0.1); no class below floor
  [PASS] golden set: 40/40 golden cases kept; regressions at rows []
GATE: PASSED, candidate may be promoted
alias 'champion' now points to v4
```
The offsets were tuned without looking at the test split, and the gain carried over to it (+0.010 on test).
The lowest class recall (0.101) is right at the floor, so the next change that lowers it will be blocked.

Where the remaining error is: about 17-23% of validation complaints have a templated near-duplicate in train
(99% credit reporting) and LoRA already gets 0.83-0.86 accuracy on them, the same as copying the sibling's
label. On the other complaints accuracy is about 0.59-0.60, which is where any further gain has to come from.

### Drift (Step 8)
`python -m complaintops.drift` compares batches of complaints with a reference (a random 70% of the validation
split, 35,227 rows) using the champion's outputs and writes an Evidently HTML report per batch
(`results/drift/*.html`, not in git because each is about 4 MB) plus `results/drift/summary.json`. The text
shows no labels, only the complaint text, the model's confidence and its predicted class, which is what is
available in production.

Signals and thresholds (the usual rules of thumb, not tuned on these batches):
- **PSI** on the predicted class mix, the prediction confidence and the text statistics (length, words, XXXX
  redactions, dollar amounts): watch at 0.10, alert at 0.25.
- **Text domain classifier**: TF-IDF + logistic regression tries to tell reference text from current text
  (5-fold AUC): 0.5 means indistinguishable; watch at 0.60, alert at 0.70.
- **Evidently**: share of drifted columns, alert at 50%.

| Batch | What it is | Status | Text AUC | Class-mix PSI | Mean confidence | Evidently drifted columns |
|---|---|---|---|---|---|---|
| normal | 5,000 other validation rows (same period) | **OK**, no alerts | 0.51 | 0.004 | 0.725 (ref 0.719) | 0% |
| shift_time | newest month of the test period (2,630 rows, real later data) | **ALERT** | 0.75 | 0.036 | 0.662 | 43% |
| shift_mix | rare products over-sampled to 70% (5,000 rows) | **ALERT** | 0.80 | 1.41 | 0.645 | 57% |

Reading it: the normal batch raises nothing. The real later month is caught by the text classifier (the
wording has changed, AUC 0.75) and the model is visibly less sure (confidence 0.72 -> 0.66), even though the
class mix and the simple text statistics barely moved; Evidently's tabular columns alone (43%) would not have
fired. This is the drift behind the test split being harder than validation (weighted F1 0.64 vs 0.59 for the
champion). The artificially shifted class mix fires on several signals at once.
`tests/test_drift.py` checks all of this: on synthetic data (new wording, new class mix and lost confidence each
fire; same-distribution data stays quiet) and on the real batches (skipped if the model outputs are missing).
Evidently 0.7.x does not import under pandas 3, so pandas is pinned below 3.

### Serving benchmark (Step 9)
The champion's LoRA adapter is merged into DistilBERT (`merge_and_unload()`), exported to ONNX and quantized to
int8 (`python -m complaintops.export_onnx`); `python -m complaintops.benchmark_serving` measures it. Latency is
end to end (tokenizing + model) on CPU for 500 validation complaints, one complaint per request and batches of
16, with 2 threads (about what a small free host gives). Quality is weighted F1 on 3,000 validation complaints,
with the champion's per-class offsets applied to every variant.

| Variant | Size (MB) | Batch 1 p50 / p95 (ms) | Batch 16 p50 / p95 (ms) | Weighted F1 | Same class as PyTorch fp32 |
|---|---|---|---|---|---|
| PyTorch fp32 | 268 | 167 / 218 | 3073 / 3159 | 0.6689 | 100% |
| ONNX fp32 | 268 | 97 / 134 | 2260 / 2322 | 0.6689 | 100% |
| **ONNX int8** | **67** | **91 / 128** | **2183 / 2580** | 0.6662 | 92.4% |

With the default thread count (6 cores) one complaint takes p50 / p95: PyTorch 177 / 256 ms, ONNX fp32
73 / 111 ms, ONNX int8 60 / 80 ms.

**Recommendation: serve ONNX int8.** It is 4x smaller and 1.8x (2 threads) to 3x (6 cores) faster than PyTorch
at a cost of 0.003 weighted F1, which is within the noise of a 3,000-row sample, though it changes 7.6% of
individual predictions. The F1 column is only for comparing variants: the offsets were tuned on validation, so
it is not the model's score (test weighted F1 0.587, see the gate).

Notes: the first export was wrong (logits off by up to 1.98 even though the classes matched) because the
library builds its padding mask with Python control flow that the exporter freezes at the sample length; the
export uses a hand-written forward pass, checked against the library model (identical) and the ONNX fp32 model
(difference 0.00001). Dynamic int8 quantization computes activation scales per batch, so int8 probabilities move
by up to about 0.04 depending on what else is in the batch (fp32 does not change at all); predicted classes stayed
the same in the test. `complaintops.serve_onnx.predict(texts)` returns a label and a confidence per text and
needs only ONNX Runtime and the tokenizer, not PyTorch. Models are in `data/onnx/` (not in git).

