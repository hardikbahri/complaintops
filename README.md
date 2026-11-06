# ComplaintOps

[![CI](https://github.com/hardikbahri/complaintops/actions/workflows/ci.yml/badge.svg)](https://github.com/hardikbahri/complaintops/actions/workflows/ci.yml) [Live page (recorded demo)](https://hb-007-complaintops.static.hf.space/)

Consumer-complaint triage: a fine-tuned classifier, an evaluation gate, drift monitoring, and an agent that sends unsure cases to a human.

## What it does
- Sorts real CFPB complaints into 24 `product | issue` classes with DistilBERT + LoRA, served as a 67 MB int8 ONNX model.
- Answers confident complaints (confidence 0.8 or more) with no human.
- For the rest, an agent searches similar past complaints, has an LLM write a short reviewer note, and queues the case with suggested labels.
- The LLM never sets a label. It was tried as the decider and did not beat the classifier.
- Every model change must pass an automated evaluation gate.

## Results (all on the 24 served classes)
- Held-out test weighted F1: baseline 0.776, LoRA 0.809, champion (LoRA + class offsets) 0.817 [0.814, 0.821].
- Champion gain over the baseline: +0.041 (95% interval +0.038 to +0.044). Test macro-F1 0.533, accuracy 0.829.
- The test period is harder than validation for every model (baseline 0.812 to 0.776): drift, not overfitting.
- Pipeline on test: 72.6% of complaints answered with no human at 93.9% accuracy (weighted F1 0.922). That row only scores what the classifier chose to answer; the classifier alone scores 0.817 on everything.
- Three credit-reporting issues that consumers pick among on a form are one class, "Credit reporting | Dispute, accuracy or use of report" (73.8% of validation complaints together). Compare scores only with the same class count.

## How it is built
- **Data:** third-party Hugging Face copy of the CFPB database (`sovai/cfpb_complaints`), 335,499 complaints after cleaning and de-duplication. Split by date: train 234,849 (to 2023-02), validation 50,325, test 50,325 (to 2024-05). A test checks that no complaint is in two splits.
- **Models (validation weighted F1 / macro-F1):** TF-IDF + logistic regression 0.812 / 0.483; DistilBERT + LoRA on 60k rows 0.824 / 0.461; on all rows 0.842 / 0.529; with per-class offsets (champion) 0.854 / 0.546. XGBoost + Optuna was trained and did not beat the baseline.
- **Tracking:** MLflow runs and model registry (`complaint-classifier`, alias `champion`).
- **Serving:** the LoRA adapter is merged, exported to ONNX and quantized to int8. 67 MB against 268 MB for PyTorch; 91 ms median for one complaint on 2 CPU threads against 167 ms; weighted F1 0.856 against 0.860 on 3,000 validation complaints; 3.3% of predictions change.
- **API:** FastAPI with `/predict`, `/triage` (rate-limited), `/health`, `/ready`, `/model`, and the web page.

## The agent
- LangGraph agent whose tools (classify, search similar complaints, flag for review) are called only through an MCP server.
- Similar-complaint search: all-MiniLM-L6-v2 embeddings of 50,000 training complaints in Postgres + pgvector, cosine similarity.
- A label is suggested only if at least 2 of the 5 neighbours share it, so planted text cannot create one.
- The LLM (`openai/gpt-oss-120b` on Groq; any OpenAI-compatible endpoint works) only writes a note of at most 700 characters. Complaint text is treated as untrusted data. If the LLM fails, a template note is used and the case is still queued.
- The review queue is a SQLite table. There is no reviewer interface and no real reviewers.
- Langfuse tracing: one trace per run, with a span per tool call and the LLM call. Off without keys, with an opt-out checkbox on the page.
- A cold-start problem found by the first trace (an 18 s first search) is fixed by loading models in the background at start-up.

## The LLM experiment
- Setup: 150 random validation complaints the classifier was unsure about (confidence below 0.8, the hardest 26%, so every number here is low by design). The LLM chose among at most five candidate labels with the true label hidden.
- 76 of the 150 calls returned a usable answer. The other 74 failed (61 hit the free tier's rate limit, 13 replies could not be parsed); the LLM never declined.
- On those 76: LLM right 39% (30), classifier right 43% (33) on the same complaints. Where they disagreed: classifier right 12 times, LLM 9 (sign test p = 0.66).
- The LLM called itself high-confidence on 99% of answers, so its confidence says nothing about when to trust it.
- I did not rerun with retries, so this is a smaller test than planned, and the 13 unreadable replies could be biased toward harder complaints.

## Evaluation gate
- A candidate replaces the champion only if it passes, on the held-out test set (24 classes): the 95% lower bound of the paired weighted-F1 difference is at least -0.005; no class has recall under 0.10; a 40-case golden set still passes. `--require-improvement` adds a rule that the lower bound is above zero.
- Weak model (baseline on 5% of data): blocked, 0.776 to 0.757, 32 of 40 golden cases kept.
- LoRA vs baseline: promoted, +0.033 (+0.030 to +0.035), 40 of 40 golden cases kept, lowest class recall 0.129.
- Class offsets vs LoRA: +0.008 (+0.006 to +0.010) but blocked on the 24 classes by one small class ("Written notification about debt", recall 0.095 against the 0.10 floor). The offsets are what is served; the verdict is reported as it came out.
- Reproduce: `python -m complaintops.gate_merged`; the original promotion tool is `complaintops.promote`.

## Drift monitoring
- `python -m complaintops.drift` compares batches with a reference (70% of validation) using PSI, a text domain classifier (AUC) and Evidently. No labels are needed.
- Normal batch: OK (text AUC 0.51). Newest month of the test period: ALERT (AUC 0.75, confidence falls 0.864 to 0.831). Over-sampled rare products: ALERT (AUC 0.80, class-mix PSI 1.38).

## Hosting
- **Static copy (free, live):** https://hb-007-complaintops.static.hf.space/. Five files; the demo replays four real recorded agent runs and says so. Built by `deploy/build_static.py`.
- **Live container:** `deploy/build_space.py` builds a 310 MB image (API, page, int8 model, in-memory search index, no PyTorch or database). It starts in about 8 s and uses about 580 MB. Hugging Face now charges for Docker Spaces, so it is not hosted. See `deploy/README.md`.

## Tests and CI
- 110 tests locally. GitHub Actions (`.github/workflows/ci.yml`) runs on every push: install, then the agent behaviour checks as their own step, then the full suite.
- Agent checks (`tests/test_agent_rules.py`) run the real agent against a stub tool server and a scripted LLM, so they need no model, database or key. They check: confident complaints never call the LLM; unsure ones are flagged for a person; the LLM never sets the label, even with a planted instruction; a failing LLM falls back to a template note. Forcing the router to accept everything makes 10 of the 11 fail.
- On a fresh checkout 70 tests pass and 40 skip (they need the model, the search index or Postgres, none of which are in git). The gate's rules are tested on synthetic predictions; the gate itself runs locally.

## Run it
```
uv sync
uv run pytest
docker compose up -d                                  # Postgres + pgvector
uv run python -m complaintops.vector_store index      # embed 50,000 training complaints
uv run uvicorn complaintops.api:app --port 8000       # open http://127.0.0.1:8000
```
- Models and data are not in git (`data/` is ignored); `export_onnx` and the training scripts rebuild them.
- Keys go in a git-ignored `.env` (see `.env.example`). With no key the agent uses a mock LLM.
- `python -m complaintops.make_showcase` regenerates `web/showcase.json` from the result files (`--record` re-records the example runs).

## Known limits
- Held-out weighted F1 is 0.817; the small classes are the weak spot (macro-F1 0.533).
- The three merged credit-reporting issues cannot be told apart.
- Overconfidence in the middle: complaints scored 0.8 to 0.9 are right 85% of the time.
- About 26% of complaints go to a person; the pipeline does not resolve them.
- The champion fails the class-recall floor on the 24 classes by 0.005.
- The XGBoost model cannot be scored on the 24 classes (its vectorizer was not saved).
- The demo's LLM runs on a free quota and is rate-limited.
