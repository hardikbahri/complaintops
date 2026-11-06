---
title: ComplaintOps
emoji: ✅
colorFrom: blue
colorTo: blue
sdk: static
pinned: false
short_description: Complaint triage that knows when not to trust its model
---

# ComplaintOps

A fine-tuned DistilBERT sorts consumer financial complaints into 24 product and issue classes. When it is unsure, an
agent looks up similar past complaints, has an LLM write a short note, and sends the case to a human reviewer.

This public copy is a static page: the results, the evaluation gate, drift monitoring and serving numbers are the real
measurements, and the demo replays real recorded runs of the agent. The full project (API, agent, tests, evaluation
code) runs locally; the page you are viewing was generated from its result files.
