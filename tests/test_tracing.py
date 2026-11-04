"""Langfuse tracing: a no-op without keys, never breaks a run, and records the right steps (fake client, no network)."""
import asyncio
from pathlib import Path

import pytest

from complaintops import tracing
from complaintops.agent import ComplaintAgent
from complaintops.llm import MockLLM

SURE = "A collector keeps calling me about a debt that is not mine."
UNSURE = "I was charged an overdraft fee even though I had enough money in my account."


class FakeObs:
    def __init__(self, log, name, as_type, fields, trace_id="trace-1"):
        self.log, self.name, self.as_type, self.fields, self.trace_id = log, name, as_type, fields, trace_id
        log.append(("start", name, as_type, fields))

    def start_observation(self, name, as_type="span", **fields):
        return FakeObs(self.log, name, as_type, fields, self.trace_id)

    def update(self, **fields):
        self.log.append(("update", self.name, fields))

    def end(self):
        self.log.append(("end", self.name))


class FakeClient:
    def __init__(self):
        self.log = []

    def start_observation(self, name, as_type="span", **fields):
        return FakeObs(self.log, name, as_type, fields)

    def get_trace_url(self, trace_id):
        return f"https://langfuse.test/trace/{trace_id}"

    def flush(self):
        self.log.append(("flush",))


@pytest.fixture
def fake(monkeypatch):
    client = FakeClient()
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_TRACING", "on")
    monkeypatch.setattr(tracing, "_client", client)
    return client


def test_without_keys_or_when_switched_off_tracing_is_a_harmless_no_op(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert not tracing.configured()
    t = tracing.Trace.start("triage", input={"text": "x"})
    assert not t.active and t.trace_id is None
    with t.step("tool:x", input={"a": 1}) as s:  # every call still works
        s.update(output="y")
    t.update(output="z"); t.end()
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test"); monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_TRACING", "off")
    assert not tracing.configured()


def test_a_failing_langfuse_never_breaks_the_caller(fake, monkeypatch):
    class Broken:
        def start_observation(self, *a, **k):
            raise RuntimeError("langfuse is down")

    monkeypatch.setattr(tracing, "_client", Broken())
    t = tracing.Trace.start("triage", input={"text": "x"})
    assert not t.active
    broken_child = tracing.Handle(FakeObs([], "x", "span", {}))
    broken_child.obs.update = lambda **k: (_ for _ in ()).throw(RuntimeError("boom"))
    broken_child.update(output=1)  # swallowed


def test_a_step_that_raises_is_marked_as_an_error_and_still_ended(fake):
    t = tracing.Trace.start("triage", input={"text": "x"})
    with pytest.raises(ValueError):
        with t.step("tool:x"):
            raise ValueError("bad")
    updates = [e for e in fake.log if e[0] == "update" and e[1] == "tool:x"]
    assert updates and updates[0][2]["level"] == "ERROR" and ("end", "tool:x") in fake.log


needs_model = pytest.mark.skipif(not Path("data/onnx/model_int8.onnx").exists(), reason="run export_onnx first")


def db_available():
    try:
        import psycopg

        from complaintops.vector_store import DB_URL

        with psycopg.connect(DB_URL, connect_timeout=2) as c:
            return c.execute("select count(*) from complaints").fetchone()[0] > 0
    except Exception:
        return False


def run_agent(text, tmp_path, monkeypatch, **kw):
    monkeypatch.setenv("REVIEW_DB", str(tmp_path / "q.db"))

    async def go():
        async with ComplaintAgent(llm=MockLLM()) as agent:
            return await agent.run(text, **kw)

    return asyncio.run(go())


@needs_model
def test_a_confident_run_is_one_trace_with_a_tool_step_and_a_link(fake, tmp_path, monkeypatch):
    r = run_agent(SURE, tmp_path, monkeypatch)
    started = [(e[1], e[2]) for e in fake.log if e[0] == "start"]
    assert started[0] == ("triage", "span") and ("tool:classify_complaint", "span") in started
    assert not any(n.startswith("llm") for n, _ in started)  # the confident path makes no LLM call
    assert r["trace_id"] == "trace-1" and r["trace_url"] == "https://langfuse.test/trace/trace-1"
    root_update = [e for e in fake.log if e[0] == "update" and e[1] == "triage"][-1][2]
    assert root_update["output"]["source"] == "classifier"
    assert ("end", "triage") in fake.log


@needs_model
def test_a_run_can_opt_out_of_tracing(fake, tmp_path, monkeypatch):
    r = run_agent(SURE, tmp_path, monkeypatch, trace=False)
    assert fake.log == [] and r["trace_id"] is None and r["trace_url"] is None


@needs_model
@pytest.mark.skipif(not db_available(), reason="start the database and run vector_store index")
def test_an_unsure_run_records_the_search_and_the_llm_as_a_generation(fake, tmp_path, monkeypatch):
    run_agent(UNSURE, tmp_path, monkeypatch)
    started = {(e[1], e[2]): e[3] for e in fake.log if e[0] == "start"}
    assert ("tool:search_similar_complaints", "span") in started
    gen = started[("llm:choose-label", "generation")]
    assert gen["model"] == "mock" and gen["input"][1]["role"] == "user" and UNSURE in gen["input"][1]["content"]
    assert [e for e in fake.log if e[0] == "update" and e[1] == "llm:choose-label"][0][2]["output"]  # the raw reply
