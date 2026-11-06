---
title: ComplaintOps
emoji: ✅
colorFrom: blue
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Complaint triage that knows when not to trust its model
---

# ComplaintOps

A fine-tuned DistilBERT sorts consumer financial complaints into 24 product and issue classes. When it is unsure, an
agent looks up similar past complaints, has an LLM write a short note, and sends the case to a human reviewer. The page
runs the real agent, and shows the held-out results, the evaluation gate, drift monitoring and serving speed.

Public demo on free hardware: please do not enter personal information. It may take a moment to wake up, and requests
are rate-limited. The data is the public CFPB Consumer Complaint Database (a third-party Hugging Face copy).
