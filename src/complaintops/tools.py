"""The three tools the agent can use. Milestone 2, step 3.

Plain Python functions so they can be tested directly; mcp_server.py exposes exactly these over MCP and the
agent calls them through that server. Inputs are validated because an LLM, not a programmer, supplies them.
"""
import json
import os
import sqlite3
import time
from pathlib import Path

MAX_CHARS = 10_000
MAX_K = 10

_classifier = None


def _clf():
    global _classifier
    if _classifier is None:  # load the model on first use, so the MCP server starts instantly
        from complaintops.serve_onnx import Classifier

        _classifier = Classifier()
    return _classifier


def warm_up():
    """Load the classifier and the embedding model now, so the first real request is not slow (a trace showed the
    search step taking 18 s on a cold start). Failures are ignored: a tool still loads what it needs on first use."""
    try:
        _clf().predict(["warm up"])
    except Exception:
        pass
    try:
        if os.environ.get("SEARCH_BACKEND", "pgvector") == "memory":
            from complaintops.search_index import embed_query, get_index

            get_index()
            embed_query("warm up")
        else:
            from complaintops.vector_store import embed

            embed(["warm up"])
    except Exception:
        pass


def _check_text(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    if len(text) > MAX_CHARS:
        raise ValueError(f"text must be at most {MAX_CHARS} characters")
    return text


def classify_complaint(text: str) -> dict:
    """Classify one complaint into a product and an issue, with the model's confidence and runner-up classes."""
    r = _clf().predict([_check_text(text)], top_k=3)[0]
    product, _, issue = r["label"].partition(" | ")
    return {
        "label": r["label"], "product": product, "issue": issue or "(no specific issue)",
        "confidence": round(r["confidence"], 4),
        "alternatives": [{"label": a["label"], "confidence": round(a["confidence"], 4)}
                         for a in r["alternatives"]],
    }


def search_similar_complaints(text: str, k: int = 5, product: str | None = None) -> dict:
    """Find the most similar past complaints (with their labels), optionally within one product."""
    if os.environ.get("SEARCH_BACKEND", "pgvector") == "memory":  # hosts without Postgres
        from complaintops.search_index import search
    else:
        from complaintops.vector_store import search

    k = max(1, min(int(k), MAX_K))
    return {"results": search(_check_text(text), k=k, product=product or None)}


def _review_db():
    path = Path(os.environ.get("REVIEW_DB", "data/review_queue.db"))
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE IF NOT EXISTS review_queue (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, status TEXT, text TEXT, reason TEXT,
        predicted_label TEXT, confidence REAL)""")
    have = {row[1] for row in conn.execute("PRAGMA table_info(review_queue)")}
    for column in ("summary", "suggestions"):  # added later; older queue files are upgraded in place
        if column not in have:
            conn.execute(f"ALTER TABLE review_queue ADD COLUMN {column} TEXT")
    return conn


def flag_for_review(text: str, reason: str, predicted_label: str | None = None,
                    confidence: float | None = None, summary: str | None = None,
                    suggestions: list | None = None) -> dict:
    """Put a complaint in the human review queue, with the reason it could not be decided automatically, an optional
    short note for the reviewer and the labels worth considering."""
    _check_text(text)
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string")
    conn = _review_db()
    with conn:
        cur = conn.execute(
            "INSERT INTO review_queue (created_at, status, text, reason, predicted_label, confidence, summary, "
            "suggestions) VALUES (?, 'queued', ?, ?, ?, ?, ?, ?)",
            (time.strftime("%Y-%m-%dT%H:%M:%S"), text, reason.strip()[:1000], predicted_label, confidence,
             (summary or "").strip()[:1000] or None, json.dumps(suggestions[:10]) if suggestions else None))
        case_id = cur.lastrowid
    queued = conn.execute("SELECT COUNT(*) FROM review_queue WHERE status = 'queued'").fetchone()[0]
    conn.close()
    return {"case_id": case_id, "status": "queued", "queue_length": queued}


def list_review_queue(limit: int = 50) -> list[dict]:
    """The queued cases, newest first (for a reviewer UI; not exposed to the agent)."""
    conn = _review_db()
    rows = conn.execute("SELECT id, created_at, status, text, reason, predicted_label, confidence, summary, suggestions "
                        "FROM review_queue ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    keys = ["case_id", "created_at", "status", "text", "reason", "predicted_label", "confidence", "summary", "suggestions"]
    out = [dict(zip(keys, r)) for r in rows]
    for row in out:
        row["suggestions"] = json.loads(row["suggestions"]) if row["suggestions"] else []
    return out
