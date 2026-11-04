"""The FastAPI service: health, predictions, and input validation."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.skipif(not Path("data/onnx/model_int8.onnx").exists(), reason="run export_onnx first")


@pytest.fixture(scope="module")
def client():
    from complaintops.api import app

    with TestClient(app) as c:  # the context manager runs the start-up that loads the model
        yield c


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json() == {"status": "ready", "agent": "ready"}
    assert client.get("/model").json()["n_classes"] == 24  # the merged taxonomy is the served default


def test_predict_returns_product_issue_confidence_and_alternatives(client):
    r = client.post("/predict", json={"texts": ["A collector keeps calling me about a debt that is not mine."],
                                      "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    p = body["predictions"][0]
    assert p["label"] == f"{p['product']} | {p['issue']}"
    assert 0 < p["confidence"] <= 1 and len(p["alternatives"]) == 3
    assert p["alternatives"][0]["label"] == p["label"]  # best first
    assert body["latency_ms"] > 0 and "model" in body


def test_predict_handles_several_texts_in_order(client):
    texts = ["There is a wrong account on my credit report.", "My mortgage servicer lost my payment."]
    preds = client.post("/predict", json={"texts": texts}).json()["predictions"]
    assert len(preds) == 2
    assert preds[0]["product"] == "Credit reporting" and preds[1]["product"] == "Mortgage"


def test_triage_runs_the_agent_and_returns_decision_trace_and_timing(client):
    r = client.post("/triage", json={"text": "A collector keeps calling me about a debt that is not mine."})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "live" and body["llm_provider"] == "mock" and body["total_ms"] > 0
    assert body["final"]["source"] in {"classifier", "llm", "human_review"}
    assert [t["step"] for t in body["trace"]][0] == "classify"
    assert body["classification"]["label"]


@pytest.mark.parametrize("payload", [{"text": ""}, {"text": "   "}, {"text": "x" * 2001}, {"texts": ["a"]}])
def test_triage_rejects_bad_input(client, payload):
    assert client.post("/triage", json=payload).status_code == 422


def test_the_rate_limiter_blocks_a_flood_but_not_normal_use():
    from fastapi import HTTPException

    from complaintops.api import RateLimiter

    rl = RateLimiter(per_minute=3, per_day=100)
    for _ in range(3):
        rl.check("a")
    with pytest.raises(HTTPException) as e:
        rl.check("a")
    assert e.value.status_code == 429
    rl.check("someone else")  # other clients are not affected by one client's flood
    day = RateLimiter(per_minute=100, per_day=2)
    day.check("a"), day.check("b")
    with pytest.raises(HTTPException):
        day.check("c")  # the global daily budget


@pytest.mark.parametrize("payload", [
    {"texts": []},                       # nothing to classify
    {"texts": ["   "]},                  # blank text
    {"texts": ["x"] * 33},               # too many
    {"texts": ["x" * 10_001]},           # too long
    {"texts": ["ok"], "top_k": 0},       # bad top_k
    {"text": "wrong field name"},        # wrong shape
])
def test_bad_requests_are_rejected_with_422(client, payload):
    assert client.post("/predict", json=payload).status_code == 422
