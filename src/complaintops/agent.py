"""LangGraph agent: trust the classifier when it is sure; otherwise look up similar complaints, have an LLM write a
short note, and queue the case for a human reviewer. Milestone 2, step 4.

The LLM never chooses the label. An experiment (README, evaluate_agent.py) found that when it chose among candidate
labels it was not better than the classifier's own guess, so its job is a note for the reviewer, with the
classifier's top guesses and the similar past complaints beside it. If the LLM is unavailable a plain template note
is used, so the review path never depends on it.

Every tool call goes through the MCP server (mcp_server.py), so the agent only knows tool names and arguments.
Flow:  classify -> (confidence >= THRESHOLD ? accept
                    : search similar -> LLM writes a note -> flag for human review with suggestions)
Each step is recorded in `trace`.

    async with ComplaintAgent() as agent:
        result = await agent.run("complaint text")
"""
import asyncio
import contextvars
import json
import os
import sys
import time
from collections import Counter
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from mcp import Client, StdioServerParameters

from complaintops.llm import SUMMARY_SYSTEM, build_summary_prompt, get_llm, summarise_case
from complaintops.tracing import Trace, flush

_TRACE = contextvars.ContextVar("trace", default=Trace())  # the current run's Langfuse trace (a no-op by default)
THRESHOLD = 0.8  # chosen from validation: the confident part is answered at high accuracy, the rest goes to a person


class State(TypedDict, total=False):
    text: str
    classification: dict
    similar: list
    suggestions: list
    note: dict
    final: dict
    trace: list


def _payload(result):
    structured = getattr(result, "structured_content", None) or getattr(result, "structuredContent", None)
    if structured:
        return structured.get("result", structured) if set(structured) == {"result"} else structured
    return json.loads(result.content[0].text)


class ComplaintAgent:
    def __init__(self, llm=None, threshold=THRESHOLD, server_module="complaintops.mcp_server"):
        self.llm = llm or get_llm()
        self.threshold = threshold
        self.server_module = server_module  # tests point this at a stub tool server
        self.graph = self._build()

    async def __aenter__(self):
        params = StdioServerParameters(command=sys.executable, args=["-m", self.server_module],
                                       env={**os.environ})
        self._client_cm = Client(params)
        self.client = await self._client_cm.__aenter__()
        return self

    async def __aexit__(self, *exc):
        await self._client_cm.__aexit__(*exc)

    async def _tool(self, name, args):
        with _TRACE.get().step(f"tool:{name}", input=args, metadata={"via": "MCP"}) as span:
            out = _payload(await self.client.call_tool(name, args))
            span.update(output=out)
            return out

    # ---- graph nodes: each returns the parts of the state it changes ----
    async def classify(self, s: State):
        t0 = time.perf_counter()
        c = await self._tool("classify_complaint", {"text": s["text"]})
        return {"classification": c, "trace": s["trace"] + [{
            "step": "classify", "tool": "classify_complaint", "model": "DistilBERT+LoRA (ONNX int8)",
            "summary": f"{c['label']} at confidence {c['confidence']:.2f}", "ms": _ms(t0)}]}

    def route(self, s: State):
        return "accept_classifier" if s["classification"]["confidence"] >= self.threshold else "search"

    async def accept_classifier(self, s: State):
        c = s["classification"]
        return {"final": self._final(c, "classifier", c["label"],
                                     f"confidence {c['confidence']:.2f} is at or above {self.threshold}"),
                "trace": s["trace"] + [{"step": "accept", "summary": "classifier answer accepted", "ms": 0}]}

    async def search(self, s: State):
        t0 = time.perf_counter()
        similar = (await self._tool("search_similar_complaints", {"text": s["text"], "k": 5}))["results"]
        suggestions = build_suggestions(s["classification"]["alternatives"], similar)
        return {"similar": similar, "suggestions": suggestions, "trace": s["trace"] + [{
            "step": "search", "tool": "search_similar_complaints",
            "summary": f"{len(similar)} similar complaints; labels worth considering: "
                       f"{[x['label'] for x in suggestions]}", "ms": _ms(t0)}]}

    async def write_note(self, s: State):
        """The LLM's only job: a short note for the reviewer. It cannot set a label, and a template covers a failure."""
        t0 = time.perf_counter()
        prompt = build_summary_prompt(s["text"], s["suggestions"], s["similar"])
        with _TRACE.get().step("llm:summarise-case", as_type="generation", model=self.llm.name,
                               input=[{"role": "system", "content": SUMMARY_SYSTEM}, {"role": "user", "content": prompt}],
                               model_parameters={"temperature": 0}) as gen:
            d = await asyncio.to_thread(summarise_case, self.llm, s["text"], s["suggestions"], s["similar"])
            gen.update(output=d.pop("reply", None), metadata={"ok": d["ok"], "error": d["error"]})
        if d["ok"]:
            note = {"text": d["summary"], "source": "llm", "model": d["model"]}
        else:
            note = {"text": template_note(s["classification"], s["suggestions"], s["similar"], self.threshold),
                    "source": "template", "model": None}
        return {"note": note, "trace": s["trace"] + [{
            "step": "llm", "model": note["model"] or "template (the LLM did not answer)",
            "summary": (f"wrote a {len(note['text'])}-character note for the reviewer" if note["source"] == "llm"
                        else "used a template note because the LLM did not answer"), "ms": _ms(t0)}]}

    async def flag(self, s: State):
        t0 = time.perf_counter()
        c, note = s["classification"], s["note"]
        reason = f"classifier confidence {c['confidence']:.2f} is below {self.threshold}"
        case = await self._tool("flag_for_review", {
            "text": s["text"], "reason": reason, "predicted_label": c["label"], "confidence": c["confidence"],
            "summary": note["text"], "suggestions": s["suggestions"]})
        final = self._final(c, "human_review", None, reason)
        final.update(case_id=case["case_id"], summary=note["text"], summary_source=note["source"],
                     suggestions=s["suggestions"])
        return {"final": final, "trace": s["trace"] + [{
            "step": "flag", "tool": "flag_for_review",
            "summary": f"queued as case {case['case_id']} (queue length {case['queue_length']})", "ms": _ms(t0)}]}

    @staticmethod
    def _final(c, source, label, reason):
        product, _, issue = (label or "").partition(" | ")
        return {"source": source, "label": label, "product": product or None, "issue": issue or None,
                "classifier_label": c["label"], "classifier_confidence": c["confidence"], "reason": reason}

    def _build(self):
        g = StateGraph(State)
        for name in ("classify", "accept_classifier", "search", "write_note", "flag"):
            g.add_node(name, getattr(self, name))
        g.add_edge(START, "classify")
        g.add_conditional_edges("classify", self.route, {"accept_classifier": "accept_classifier", "search": "search"})
        g.add_edge("search", "write_note")
        g.add_edge("write_note", "flag")
        g.add_edge("accept_classifier", END)
        g.add_edge("flag", END)
        return g.compile()

    async def run(self, text: str, trace: bool = True) -> dict:
        """Run the agent on one complaint. trace=False keeps this run out of Langfuse even if tracing is configured."""
        tr = Trace.start("triage", input={"text": text}, enabled=trace,
                         metadata={"threshold": self.threshold, "llm": self.llm.name})
        token = _TRACE.set(tr)
        try:
            state = await self.graph.ainvoke({"text": text, "trace": []})
        except BaseException as e:
            tr.update(level="ERROR", status_message=f"{type(e).__name__}: {e}"[:300])
            tr.end()
            raise
        finally:
            _TRACE.reset(token)
        out = {"final": state["final"], "trace": state["trace"], "classification": state["classification"],
               "similar": state.get("similar", []), "suggestions": state.get("suggestions", []),
               "note": state.get("note"), "trace_id": None, "trace_url": None}
        if tr.active:
            tr.update(output=state["final"], metadata={"source": state["final"]["source"], "threshold": self.threshold,
                                                       "llm": self.llm.name})
            tr.end()
            out["trace_id"], out["trace_url"] = tr.trace_id, await asyncio.to_thread(tr.url)
            asyncio.get_running_loop().run_in_executor(None, flush)  # deliver in the background; the response is not held up
        return out


def build_suggestions(alternatives, similar):
    """The labels worth a reviewer's attention: the classifier's top three, plus any label that at least two of the five
    similar complaints share (at most five). A single stray neighbour, which text planted in the complaint could
    cause, cannot add a suggestion."""
    votes = Counter(x["label"] for x in similar)
    out = [{"label": a["label"], "classifier_confidence": a["confidence"], "neighbour_votes": votes.get(a["label"], 0)}
           for a in alternatives]
    seen = {s["label"] for s in out}
    for label, n in votes.most_common():
        if n >= 2 and label not in seen:
            out.append({"label": label, "classifier_confidence": None, "neighbour_votes": n})
    return out[:5]


def build_candidates(alternatives, similar):
    """Just the labels of build_suggestions (kept for evaluate_agent.py, which measures an LLM choosing among them)."""
    return [s["label"] for s in build_suggestions(alternatives, similar)]


def template_note(classification, suggestions, similar, threshold):
    """A note written without any model, for when the LLM is unavailable."""
    parts = [f"The classifier's best guess is {classification['label']} at {classification['confidence']:.0%}, "
             f"below the {threshold:.0%} threshold for acting alone."]
    top = [s for s in suggestions if s["classifier_confidence"] is not None][1:3]
    if top:
        parts.append("Close alternatives: " + "; ".join(f"{s['label']} ({s['classifier_confidence']:.0%})" for s in top) + ".")
    votes = Counter(x["label"] for x in similar)
    if votes:
        label, n = votes.most_common(1)[0]
        parts.append(f"{n} of the {len(similar)} most similar past complaints were labelled {label}.")
    return " ".join(parts)


def _ms(t0):
    return round((time.perf_counter() - t0) * 1000)
