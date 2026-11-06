# ComplaintOps

Complaint triage platform: train, track, gate, monitor and serve models, then put an agent in front of them.

Status: Milestone 1 (model side) done; Milestone 2 (agent) in progress: classifier API, similar-complaint search,
MCP tools, LangGraph agent, tracing, the web page, agent checks in CI and a free static hosted copy are built; a live hosted
agent, a second agent framework and Milestone 3 (kind, Terraform) are not.

## Quick start
```
uv sync
uv run pytest
docker compose up -d                                  # Postgres + pgvector for similar-complaint search
uv run python -m complaintops.vector_store index      # embed 50,000 training complaints (needs data/processed)
uv run uvicorn complaintops.api:app --port 8000       # then open http://127.0.0.1:8000
```
Models and data are not in git (`data/` is ignored); `export_onnx` and the training scripts rebuild them. Keys go in
a git-ignored `.env` (see `.env.example`); with no key the agent runs against a mock LLM.

## Continuous integration
`.github/workflows/ci.yml` runs on every push and pull request: it installs the locked dependencies, then runs the agent's
behaviour checks (`tests/test_agent_rules.py`) as their own step, then the full suite. The agent checks run the real
agent (LangGraph, the MCP client and protocol, the real review queue) against a stub tool server
(`tests/stub_mcp_server.py`) with canned classifier and search answers and a scripted LLM, so they need no model, database
or API key. They protect the rules the page claims: confident complaints never call the LLM, unsure ones are flagged for a
person, the LLM never sets the label (including when the complaint carries a planted instruction), and a failing LLM falls
back to a template note. Forcing the router to accept every complaint makes 10 of the 11 fail. Tests that need the 67 MB model,
the search index or Postgres (none are in git) skip themselves on a fresh checkout (70 pass and 40 skip there), and the
gate's rules are tested on synthetic predictions; the gate itself runs on the model's real predictions, locally.

## Hosting
`deploy/` builds two bundles (details in `deploy/README.md`). A **static copy** (five files, about 120 KB) works on any
static host: the page detects that there is no backend and replays four real recorded agent runs, saying so, with
no trace links or keys in it (the builder refuses to finish otherwise). A **live copy** is a slim container for any host
that runs Docker: the API, the page, the int8 model and an in-memory copy of the similar-complaint index (the same
neighbours came back on 25 of 25 test lookups; query embeddings come from fastembed, which matches the stored vectors).
It needs no PyTorch and no database driver (a test checks that importing the server pulls in neither); run locally from
the built image it was 310 MB, ready in about 8 seconds, answered a confident complaint in 59 ms and an unsure one in
326 ms, and used about 580 MB of memory. Hugging Face now charges for Docker Spaces, so the free route is the static copy.

## Label taxonomy: 24 classes
The task is to predict `product | issue`. Three credit-reporting issues that consumers choose among on a form
("Incorrect information on your report", "Problem with a credit reporting company's investigation", "Improper use
of your report") are one class, "Credit reporting | Dispute, accuracy or use of report"; together they were 73.8%
of validation complaints. The merge is applied after the model by adding the three classes' probabilities
(`complaintops.taxonomy`), so every score in this README is for the 24 resulting classes, and weighted F1 depends on
the class count, so compare scores only with the same task.

On the held-out test set the baseline scores 0.776 weighted F1 and the champion (LoRA + class offsets) 0.817 (95%
interval of the gain over the baseline +0.038 to +0.044). The champion's test macro-F1 is 0.533 and its accuracy
0.829. At confidence 0.8 the classifier alone answers 74% of validation complaints at 96.8% accuracy. Reproduce with
`python -m complaintops.evaluate_taxonomy`.

## The web page and the agent
`web/` is a single static page (plain HTML, CSS and JavaScript, no build step) served by the FastAPI app. It has a
live demo that runs the real agent (`POST /triage`) and shows the decision, the classifier's confidence against the
agent's threshold, the similar past complaints and a step-by-step trace, followed by the evidence: model
comparison, the held-out test intervals, the evaluation gate, the confidence check, drift and serving speed.
Every number is read from `web/showcase.json`, which `python -m complaintops.make_showcase` regenerates from the
MLflow runs and result files (`--record` also re-records the example runs against the real LLM). Each chart has a
table view, and light and dark themes follow the system setting. If the live service cannot be reached the page
falls back to the recorded runs and says so. `/triage` is rate-limited (per client and per day) so a public demo
cannot exhaust a free LLM quota.

The agent (`complaintops.agent`) calls three tools, only through an MCP server (`complaintops.mcp_server`):
`classify_complaint`, `search_similar_complaints` (pgvector) and `flag_for_review`. It trusts the classifier at
confidence 0.8 or more. Every other complaint is searched for similar past ones, an LLM writes a short note for the
reviewer (what the complaint is about, why the candidate labels are hard to tell apart, what would settle it), and
the case is queued for a human with the classifier's top guesses, the labels that at least two of the five similar
complaints share, and the note. The LLM never chooses a label, so text planted in a complaint cannot make the
pipeline invent one; if the LLM is unavailable a template note is used and the case is queued all the same.

**Why the LLM does not decide.** It was tried as the decider first and measured (below): it did not beat the
classifier's own guess, so its job was changed to writing notes.

**Tracing (Langfuse).** With `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and `LANGFUSE_BASE_URL` set in `.env`,
each `/triage` run is recorded as one Langfuse trace (`complaintops.tracing`): a root span `triage`, a span per tool
call made through MCP (`tool:classify_complaint`, `tool:search_similar_complaints`, `tool:flag_for_review`) and a
generation `llm:summarise-case` holding the model name, the exact prompt and the raw reply, all with timings. The
response carries `trace_id` and `trace_url`, and the page links to it. Tracing does nothing without keys and cannot
break a request (every call swallows its own errors). Because the run's text is sent to Langfuse's cloud, the page
shows an opt-out checkbox whenever the server traces (the API takes `trace=false`), and a public deployment should
leave the keys unset. A trace is private to the Langfuse project, so a reader needs access or a shared link.
The first real trace paid for itself: it showed the similarity search taking 18 s on the first request after a start
(the embedding model loading), so the tool server now loads its models in the background at start-up; the same
request takes about 3 s end to end. `tests/test_tracing.py` checks the recorded structure with a fake client.

**Experiment: should the LLM choose the label?** `python -m complaintops.evaluate_agent` asks the LLM
(`openai/gpt-oss-120b`) to choose among at most five candidate labels for 150 random validation complaints below the
threshold, with the true label hidden. These are the classifier's low-confidence complaints, the hardest 26% of the
data, so every number in this table is low by design and is not comparable with the overall scores:

| | 24 classes |
|---|---|
| Unsure complaints (below 0.8) | 26% of validation |
| Classifier's top guess right | 47% |
| True label among the candidates | 85% |
| LLM returned a usable answer | 51% (76 of 150) |
| LLM right on those answers | 39% (30 of 76) |
| Classifier right on the same complaints | 43% (33 of 76) |
| Disagreements: classifier right / LLM right | 12 / 9 (sign test p = 0.66) |

The other 74 calls returned nothing usable: 61 hit the free tier's rate limit and 13 replies could not be parsed as JSON; the
LLM never declined to answer. So this is a smaller test than planned (I chose not to rerun it with retries), and the 13 unreadable
replies could be biased toward harder complaints. The LLM called itself "high confidence" on 99% of its answers, so its confidence says nothing about when to trust
it, and it did not beat the classifier's own guess. On validation the LLM-decides pipeline scored -0.010
weighted F1 against the classifier alone on the same complaints (95% interval -0.028 to +0.008), which is why it was
replaced.

**End to end, the pipeline that runs now** (`python -m complaintops.pipeline_eval`, 24 classes). The threshold (0.8)
and the per-class offsets were fixed on validation, so the test column is the clean one:

| | Test weighted F1 [95% interval] | Test accuracy | Validation weighted F1 |
|---|---|---|---|
| Classifier alone, every complaint | 0.817 [0.814, 0.821] | 82.9% | 0.854 |
| Pipeline, answered with no human (72.6% of test complaints) | 0.922 [0.919, 0.925] | 93.9% | 0.963 |

The no-human row scores only what the classifier answered itself, a part selected by the model's own confidence, so
read it with its coverage. The
point of the pipeline is the trade between accuracy and automation: about 73% of complaints with no human at 94%
accuracy on test, the rest to a person. One model and one zero-shot prompt were tried, so this is a
first measurement, not a verdict on LLMs.

## Results

Every score in this section is for the 24 served classes (see the label taxonomy above).

### Data (Step 2)
The official CFPB bulk file and API no longer include the complaint narrative, so this project uses
a third-party copy of the CFPB Consumer Complaint Database: Hugging Face `sovai/cfpb_complaints`
(not an official CFPB release). A random 400,000-row sample of rows with a narrative was taken,
keeping only `date`, `product`, `issue`, `submitted_via` and `narrative`; leakage-prone columns
(company response, timely response, ...) are never used.

Cleaning: narratives under 10 characters dropped; exact duplicates dropped after lowercasing and
collapsing whitespace, keeping the earliest (400,000 -> 335,499 rows). Label = `product | issue`.
Renamed products and issues (the CFPB changed its menus around 2017-04-24) were merged to their
modern names; the mortgage and checking-account mappings are approximate. The most common labels in
train are kept and the rest become "Other"; the three overlapping credit-reporting issues are then merged, giving the
24 served classes. Split is by date: oldest 70% / next 15% / newest 15%.

### Splits (Step 2)
| Split | Rows | Date range | Classes |
|---|---|---|---|
| train | 234,849 | 2015-03-19 to 2023-02-13 | 24 |
| validation | 50,325 | 2023-02-13 to 2023-09-08 | 24 |
| test | 50,325 | 2023-09-08 to 2024-05-01 | 24 |

The biggest class is 28-31% of every split; "Other" is 13.5% of train but 6.8% (val) and 8.5% (test).
The smallest classes have only 113 (val) and 93 (test) rows, so their recall is noisy.
`tests/test_data.py` checks that no narrative appears in two splits.

### Model comparison (Steps 3, 4, 6), 24 classes
Headline metric: **weighted F1** (each class's F1 weighted by its number of validation rows). Macro-F1
(every class counts equally, so small classes weigh as much as large ones) is reported next to it and is
the stricter measure. The headline was switched from macro to weighted F1 after the first results, so the
two columns are always shown together; all models are scored on both on the same validation split.

| Model | Val weighted F1 | Val macro-F1 | Val accuracy | Train time | Notes |
|---|---|---|---|---|---|
| TF-IDF + logistic regression | 0.8124 | 0.4826 | 0.809 | ~16 min (235k rows) | the baseline |
| XGBoost (Optuna-tuned) | not scored | not scored | not scored | ~33 min (T4 GPU, 211k rows) | trained and did not beat TF-IDF + LR on the original labels; its vectorizer was not saved, so it cannot be scored on the 24 served classes without retraining |
| DistilBERT + LoRA (first run: 60k of 235k train rows, 2 epochs, rank 8) | 0.8244 | 0.4605 | 0.827 | ~75 min (GTX 1650, fp32) | trainable params: 758,042 of 67,731,508 (1.12%); adapter size: 3.04 MB |
| DistilBERT + LoRA (all train rows, 3 epochs, rank 16 on all linear layers) | 0.8423 | 0.5289 | 0.843 | ~92 min (Colab T4, fp16) | trainable params: 1,937,690 of 68,911,156 (2.81%); adapter size: 7.77 MB; resumed once after a Colab disconnect |
| LoRA + per-class offsets (champion) | 0.8536 | 0.5456 | 0.858 | minutes, no training | the offsets were tuned on these validation rows, so this row is slightly optimistic; the held-out test is the clean comparison |
| LoRA + TF-IDF LR blend (not adopted) | | | | minutes | under one point of macro-F1 over LoRA alone; not worth serving two models |

Two easier side tasks were also tried and are not comparable with these rows: merging 12 confusable labels into 5
(a 19-class task, weighted F1 0.8028, macro-F1 0.4777, dominated by one class that is 74% of validation) and
predicting only the product (10 classes, weighted F1 0.89, macro-F1 0.6296).

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
An earlier attempt merged 12 confusable labels into 5 (19 classes: three credit-reporting issues, three
debt-collection misconduct issues, two mortgage payment issues, opening/managing a checking account, two
credit-card terms issues). It did not help: the original model scored on the merged labels reached macro-F1 0.4801 and
a TF-IDF + LR model retrained on merged labels 0.4777, so training on them added nothing. The merged task is
easier and dominated by one class (74% of validation), so its weighted F1 looks high without the model improving.
The weak classes are a long tail of small ones, not the merged ones. The only merge kept is the three
credit-reporting issues in the served taxonomy, as a labelled product decision.

### Product-level task (easier version)
Predicting only the product (9 most common products in train + "Other" = 10 classes) reuses the same
data, splits and leakage checks; only the target changes. TF-IDF + LR reaches macro-F1 0.6296 and
accuracy 0.876. "Other" has 6 validation rows and scores 0, so over the 9 real products macro-F1 is about
0.70. Credit reporting is 59% of train but 75% of validation, and balanced class weights make the model
over-predict small products (e.g. debt collection precision 0.45, recall 0.72). Ambiguity exists at the
product level too: credit-report complaints often mention collectors, auto loans or cards.
The issue-level task stays the main one.

### Experiment tracking (Step 5)
All runs are logged to MLflow (`sqlite:///mlflow.db`, experiment `complaintops`) with parameters, metrics,
artifacts and the git commit. View them with
`uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`. The baseline, merged-label, XGBoost and Optuna
runs were back-filled from saved results (`python -m complaintops.log_existing`, tagged
`logged_after_the_fact=true`); later runs are logged live by the training scripts through
`complaintops.tracking.log_run`. The best model with a registrable artifact is registered as
`complaint-classifier` with alias `champion` (v1: TF-IDF + LR; `complaintops.register`).
A blend of LR and XGBoost was attempted and dropped: the XGBoost model was trained on Colab and its fitted
TF-IDF vectorizer was not saved, and a local refit does not reproduce the same vocabulary (ties at the
`max_features` cutoff), so the model could not be reused.

### Error analysis (Step 3)
Read from 20 random validation errors (12 stay within the right product, 8 cross products)
and the per-class report. The weakest classes are in debt collection (false statements, written notification,
threatened legal action) and "Other".

1. **The three big credit-reporting issues overlap.** "Incorrect information", "Improper use" and "Problem with the investigation" describe nearly the same story (disputing items on a report), so the model often picked a sibling; some labels are arguable even to a human. This is why the served taxonomy merges them.
2. **Words from other products appear inside credit-reporting complaints.** Mentions of charge-offs, collections, auto or credit-card accounts pushed predictions to debt collection or vehicle loan.
3. **Heavy redaction removes the signal.** Complaints that are mostly `XXXX` tokens leave almost no vocabulary for TF-IDF.
4. **Adjacent classes within a product** (mortgage "Struggling to pay" vs "Trouble during payment", checking "Opening" vs "Managing") differ by intent, not distinctive words, and bag-of-words cannot tell intent.
5. **"Other" is a grab-bag** with no consistent vocabulary, so its recall is low.

### Evaluation gate (Step 7), on the 24 classes
A candidate model version replaces the `champion` alias only if `python -m complaintops.promote
--candidate-version N` passes three checks, all on the **test** split (scored here for the first time; every
earlier choice was made on validation) plus the golden set. The metric is **weighted F1**. The registered models
output the original label probabilities; `python -m complaintops.gate_merged` adds them into the 24 served classes
and applies the same rules (`results/gate24_*.json`), and that is what is reported here.

1. The 95% lower bound of (candidate - champion) weighted F1 must be at least -0.005. Both models are scored
   on the same 1,000 bootstrap resamples (paired), so test-set noise cancels. Comparing the candidate's own
   lower bound with the champion's score blocked even an identical model on a small sample, which a unit test
   caught.
2. No class may have recall under 0.10 (a class silently collapsing). This replaces macro-F1 as the guard for
   small classes.
3. A 40-case golden set (`tests/golden_set.csv`: complaints from the validation split that the baseline and
   LoRA both get right with confidence; mostly short, heavily redacted or multi-topic) must
   keep passing: any case the champion gets right and the candidate gets wrong blocks the candidate.

`tests/test_gate.py` checks the rules on synthetic predictions (identical model passes, clearly worse model
blocked, collapsed small class blocked, golden regression blocked, better model passes).

**Blocked run**: v2, the baseline pipeline trained on 5% of train (11,742 rows), against champion v1
(TF-IDF + LR):
```
champion  weighted F1 0.7763  95% CI [0.7724, 0.7799]
candidate weighted F1 0.7570  95% CI [0.7527, 0.7613]
difference -0.0193  95% CI [-0.0220, -0.0166]  (1000 paired bootstrap resamples)
  [FAIL] lower bound of the difference: 95% lower bound of (candidate - champion) = -0.0220 must be >= -0.005
  [FAIL] class recall floor: lowest class recall 0.062 (floor 0.1); below floor: [('Debt collection | Written notification about debt', 0.062)]
  [FAIL] golden set: 32/40 golden cases kept; regressions at rows [6, 12, 13, 16, 23, 35, 37, 39]
GATE: BLOCKED
```
**Promoted run**: v3, DistilBERT + LoRA (all train rows, 3 epochs, rank 16), against champion v1:
```
champion  weighted F1 0.7763  95% CI [0.7724, 0.7799]
candidate weighted F1 0.8088  95% CI [0.8053, 0.8126]
difference +0.0326  95% CI [+0.0298, +0.0351]  (1000 paired bootstrap resamples)
  [PASS] lower bound of the difference: 95% lower bound of (candidate - champion) = +0.0298 must be >= -0.005
  [PASS] class recall floor: lowest class recall 0.129 (floor 0.1); no class below floor
  [PASS] golden set: 40/40 golden cases kept; regressions at rows []
GATE: PASSED, candidate may be promoted
```
Notes: the test split is harder than validation for both models (baseline weighted F1 0.8124 on validation vs
0.7763 on test; LoRA 0.8423 vs 0.8088). Both drop by a similar amount, which points to the later time period
(data drift) and not to overfitting to validation; Step 8 looks at this. Saved test probabilities are reused after a
spot check against the live registered model (32 rows each, 100% agreement). The blend was not taken forward
because it added under one point of macro-F1.

**Refinement run**: v4 = v3 plus per-class offsets on the scores, tuned for weighted F1 on the
**validation** split only (a 5-fold cross-fitted check and a chronological check, tuned on the older half and scored
on the newer half, both showed a gain). Gated against champion v3 with `--require-improvement`, which
adds a rule that the lower bound of the difference must be above zero:
```
champion  weighted F1 0.8088  95% CI [0.8053, 0.8126]
candidate weighted F1 0.8173  95% CI [0.8138, 0.8210]
difference +0.0085  95% CI [+0.0064, +0.0105]  (1000 paired bootstrap resamples)
  [PASS] lower bound of the difference: 95% lower bound of (candidate - champion) = +0.0064 must be >= -0.005
  [PASS] must clearly improve: 95% lower bound of the difference +0.0064 must be > 0
  [FAIL] class recall floor: lowest class recall 0.095 (floor 0.1); below floor: [('Debt collection | Written notification about debt', 0.095)]
  [PASS] golden set: 40/40 golden cases kept; regressions at rows []
GATE: BLOCKED
```
On the 24 classes the gate blocks v4 by 0.005 of recall on one small class, even though v4 is clearly better overall
(it was promoted earlier, when the gate was run on the original label probabilities). The offsets are what is served
now, and the gate's verdict is reported as it came out rather than tuned away. The offsets were tuned without
looking at the test split, and the overall gain carried over to it.

Where the remaining error is: about 17-23% of validation complaints have a templated near-duplicate in train
(99% credit reporting) and LoRA already gets 0.83-0.86 accuracy on them, the same as copying the sibling's
label. Accuracy on the other complaints is much lower, which is where any further gain has to come from.

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
| normal | 5,000 other validation rows (same period) | **OK**, no alerts | 0.51 | 0.004 | 0.867 (ref 0.864) | 0% |
| shift_time | newest month of the test period (2,630 rows, real later data) | **ALERT** | 0.75 | 0.026 | 0.831 | 43% |
| shift_mix | rare products over-sampled to 70% (5,000 rows) | **ALERT** | 0.80 | 1.38 | 0.706 | 57% |

Reading it: the normal batch raises nothing. The real later month is caught by the text classifier (the
wording has changed, AUC 0.75) and the model is visibly less sure (confidence 0.86 -> 0.83), even though the
class mix and the simple text statistics barely moved; Evidently's tabular columns alone (43%) would not have
fired. This is the drift behind the test split being harder than validation (weighted F1 0.854 on validation vs 0.817 on test for the
champion). The artificially shifted class mix fires on several signals at once.
`tests/test_drift.py` checks all of this: on synthetic data (new wording, new class mix and lost confidence each
fire; same-distribution data stays quiet) and on the real batches (skipped if the model outputs are missing).
Evidently 0.7.x does not import under pandas 3, so pandas is pinned below 3.

### Serving benchmark (Step 9)
The champion's LoRA adapter is merged into DistilBERT (`merge_and_unload()`), exported to ONNX and quantized to
int8 (`python -m complaintops.export_onnx`); `python -m complaintops.benchmark_serving` measures it. Latency is
end to end (tokenizing + model) on CPU for 500 validation complaints, one complaint per request and batches of
16, with 2 threads (about what a small free host gives). Quality is weighted F1 (24 classes) on 3,000 validation complaints,
with the champion's per-class offsets applied to every variant.

| Variant | Size (MB) | Batch 1 p50 / p95 (ms) | Batch 16 p50 / p95 (ms) | Weighted F1 | Same class as PyTorch fp32 |
|---|---|---|---|---|---|
| PyTorch fp32 | 268 | 167 / 218 | 3073 / 3159 | 0.8605 | 100% |
| ONNX fp32 | 268 | 97 / 134 | 2260 / 2322 | 0.8605 | 100% |
| **ONNX int8** | **67** | **91 / 128** | **2183 / 2580** | 0.8563 | 96.7% |

With the default thread count (6 cores) one complaint takes p50 / p95: PyTorch 177 / 256 ms, ONNX fp32
73 / 111 ms, ONNX int8 60 / 80 ms.

**Recommendation: serve ONNX int8.** It is 4x smaller and 1.8x (2 threads) to 3x (6 cores) faster than PyTorch
at a cost of 0.004 weighted F1, which is within the noise of a 3,000-row sample, though it changes 3.3% of
predicted classes. The F1 column is only for comparing variants: the offsets were tuned on validation, so
it is not the model's score (held-out test weighted F1 is 0.817).

Notes: the first export was wrong (logits off by up to 1.98 even though the classes matched) because the
library builds its padding mask with Python control flow that the exporter freezes at the sample length; the
export uses a hand-written forward pass, checked against the library model (identical) and the ONNX fp32 model
(difference 0.00001). Dynamic int8 quantization computes activation scales per batch, so int8 probabilities move
by up to about 0.04 depending on what else is in the batch (fp32 does not change at all); predicted classes stayed
the same in the test. `complaintops.serve_onnx.predict(texts)` returns a label and a confidence per text and
needs only ONNX Runtime and the tokenizer, not PyTorch. Models are in `data/onnx/` (not in git).

