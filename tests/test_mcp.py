"""The tools, called directly and through the real MCP server over stdio."""
import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
from mcp import Client, StdioServerParameters

from complaintops import tools

needs_model = pytest.mark.skipif(not Path("data/onnx/model_int8.onnx").exists(), reason="run export_onnx first")


def db_available():
    try:
        import psycopg

        from complaintops.vector_store import DB_URL

        with psycopg.connect(DB_URL, connect_timeout=2) as c:
            return c.execute("select count(*) from complaints").fetchone()[0] > 0
    except Exception:
        return False


needs_db = pytest.mark.skipif(not db_available(), reason="start the database and run vector_store index")


def payload(result):
    """The tool's returned dict, whichever way this MCP version delivers it."""
    structured = getattr(result, "structured_content", None) or getattr(result, "structuredContent", None)
    if structured:
        return structured.get("result", structured) if set(structured) == {"result"} else structured
    return json.loads(result.content[0].text)


def server_params(review_db):
    return StdioServerParameters(command=sys.executable, args=["-m", "complaintops.mcp_server"],
                                 env={**os.environ, "REVIEW_DB": str(review_db)})


# ---- the functions themselves ----

def test_flag_for_review_queues_cases_and_validates_input(tmp_path, monkeypatch):
    monkeypatch.setenv("REVIEW_DB", str(tmp_path / "q.db"))
    a = tools.flag_for_review("My overdraft fee was wrong.", "low confidence", "Other", 0.5)
    b = tools.flag_for_review("Another complaint.", "similar complaints disagree")
    assert a["status"] == "queued" and (a["queue_length"], b["queue_length"]) == (1, 2)
    queue = tools.list_review_queue()
    assert [q["case_id"] for q in queue] == [b["case_id"], a["case_id"]]  # newest first
    with pytest.raises(ValueError):
        tools.flag_for_review("   ", "reason")
    with pytest.raises(ValueError):
        tools.flag_for_review("text", "")


@needs_model
def test_classify_complaint_function():
    r = tools.classify_complaint("A collector keeps calling me about a debt that is not mine.")
    assert r["label"] == f"{r['product']} | {r['issue']}" and 0 < r["confidence"] <= 1
    assert len(r["alternatives"]) == 3
    with pytest.raises(ValueError):
        tools.classify_complaint("")


# ---- through MCP ----

def test_mcp_server_lists_exactly_the_three_tools(tmp_path):
    async def run():
        async with Client(server_params(tmp_path / "q.db")) as c:
            return {t.name: t for t in (await c.list_tools()).tools}

    found = asyncio.run(run())
    assert set(found) == {"classify_complaint", "search_similar_complaints", "flag_for_review"}
    description = " ".join(found["classify_complaint"].description.split())  # the LLM reads these
    assert "Always call this first" in description


@needs_model
def test_classify_through_mcp(tmp_path):
    async def run():
        async with Client(server_params(tmp_path / "q.db")) as c:
            return await c.call_tool("classify_complaint",
                                     {"text": "There is a wrong account on my credit report."})

    r = payload(asyncio.run(run()))
    assert r["product"] == "Credit reporting" and 0 < r["confidence"] <= 1


def test_flag_for_review_through_mcp_writes_to_the_queue(tmp_path):
    async def run():
        async with Client(server_params(tmp_path / "q.db")) as c:
            return await c.call_tool("flag_for_review", {"text": "Unclear complaint", "reason": "test"})

    r = payload(asyncio.run(run()))
    assert r["status"] == "queued" and r["queue_length"] == 1
    assert (tmp_path / "q.db").exists()  # the server process wrote it


@needs_model
@needs_db
def test_search_through_mcp(tmp_path):
    async def run():
        async with Client(server_params(tmp_path / "q.db")) as c:
            return await c.call_tool("search_similar_complaints",
                                     {"text": "I was charged an overdraft fee I should not have been.", "k": 3})

    results = payload(asyncio.run(run()))["results"]
    assert len(results) == 3 and results[0]["similarity"] >= results[-1]["similarity"]
    assert all(r["label"] and r["snippet"] for r in results)
