"""LangGraph agent: trust the classifier when it is sure, otherwise look up similar complaints and ask an LLM,
and send anything still unclear to a human. Milestone 2, step 4.

Every tool call goes through the MCP server (mcp_server.py), so the agent only knows tool names and arguments.
Flow:  classify -> (confidence >= THRESHOLD ? accept : search similar -> LLM chooses among candidates)
       -> (LLM sure ? accept : flag for human review).   Each step is recorded in `trace`.

    async with ComplaintAgent() as agent:
        result = await agent.run("complaint text")
"""
import asyncio
import json
import os
import contextvars
import sys
import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from mcp import Client, StdioServerParameters

from complaintops.llm import SYSTEM, build_prompt, decide_label, get_llm
from complaintops.tracing import Trace, flush

_TRACE = contextvars.ContextVar("trace", default=Trace())  # the current run's Langfuse trace (a no-op by default)
THRESHOLD = 0.8  # from validation: at >= 0.8 the classifier is right 85% of the time and handles 43% of complaints


class State(TypedDict, total=False):
    text: str
    classification: dict
    similar: list
    candidates: list
    llm: dict
    final: dict
    trace: list


def _payload(result):
    structured = getattr(result, "structured_content", None) or getattr(result, "structuredContent", None)
    if structured:
        return structured.get("result", structured) if set(structured) == {"result"} else structured
    return json.loads(result.content[0].text)


class ComplaintAgent:
    def __init__(self, llm=None, threshold=THRESHOLD):
        self.llm = llm or get_llm()
        self.threshold = threshold
        self.graph = self._build()

    async def __aenter__(self):
        params = StdioServerParameters(command=sys.executable, args=["-m", "complaintops.mcp_server"],
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
        candidates = build_candidates(s["classification"]["alternatives"], similar)
        return {"similar": similar, "candidates": candidates, "trace": s["trace"] + [{
            "step": "search", "tool": "search_similar_complaints",
            "summary": f"{len(similar)} similar complaints; candidates: {candidates[:5]}", "ms": _ms(t0)}]}

    async def ask_llm(self, s: State):
        t0 = time.perf_counter()
        prompt = build_prompt(s["text"], s["candidates"], s["similar"])
        with _TRACE.get().step("llm:choose-label", as_type="generation", model=self.llm.name,
                               input=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                               model_parameters={"temperature": 0}) as gen:
            d = await asyncio.to_thread(decide_label, self.llm, s["text"], s["candidates"], s["similar"])
            gen.update(output=d.pop("reply", None), metadata={"chosen": d["label"], "accepted": d["ok"],
                                                              "confidence": d["confidence"], "reason": d["reason"]})
        return {"llm": d, "trace": s["trace"] + [{
            "step": "llm", "model": d["model"],
            "summary": f"chose {d['label']!r} ({d['confidence']}): {d['reason']}", "ms": _ms(t0)}]}

    def route_llm(self, s: State):
        return "accept_llm" if s["llm"]["ok"] else "flag"

    async def accept_llm(self, s: State):
        d = s["llm"]
        final = self._final(s["classification"], "llm", d["label"], d["reason"])
        final["llm_confidence"] = d["confidence"]
        return {"final": final, "trace": s["trace"] + [{"step": "accept", "summary": "LLM answer accepted", "ms": 0}]}

    async def flag(self, s: State):
        t0 = time.perf_counter()
        c, d = s["classification"], s.get("llm", {})
        reason = (f"classifier confidence {c['confidence']:.2f} (below {self.threshold}); "
                  f"LLM: {d.get('reason') or 'no answer'}")
        case = await self._tool("flag_for_review", {"text": s["text"], "reason": reason,
                                                    "predicted_label": c["label"], "confidence": c["confidence"]})
        final = self._final(c, "human_review", None, reason)
        final["case_id"] = case["case_id"]
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
        for name in ("classify", "accept_classifier", "search", "ask_llm", "accept_llm", "flag"):
            g.add_node(name, getattr(self, name))
        g.add_edge(START, "classify")
        g.add_conditional_edges("classify", self.route, {"accept_classifier": "accept_classifier", "search": "search"})
        g.add_edge("search", "ask_llm")
        g.add_conditional_edges("ask_llm", self.route_llm, {"accept_llm": "accept_llm", "flag": "flag"})
        for end in ("accept_classifier", "accept_llm", "flag"):
            g.add_edge(end, END)
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
               "similar": state.get("similar", []), "candidates": state.get("candidates", []),
               "llm": state.get("llm"), "trace_id": None, "trace_url": None}
        if tr.active:
            tr.update(output=state["final"], metadata={"source": state["final"]["source"], "threshold": self.threshold,
                                                       "llm": self.llm.name})
            tr.end()
            out["trace_id"], out["trace_url"] = tr.trace_id, await asyncio.to_thread(tr.url)
            asyncio.get_running_loop().run_in_executor(None, flush)  # deliver in the background; the response is not held up
        return out


def build_candidates(alternatives, similar):
    """The labels the LLM may choose from: the classifier's top 3, plus any label that at least 2 of the similar
    complaints agree on (at most 5). A single stray neighbour, which text planted in the complaint could cause,
    cannot add a candidate."""
    votes = {}
    for x in similar:
        votes[x["label"]] = votes.get(x["label"], 0) + 1
    candidates = [a["label"] for a in alternatives]
    candidates += [lab for lab, n in sorted(votes.items(), key=lambda kv: -kv[1]) if n >= 2 and lab not in candidates]
    return candidates[:5]


def _ms(t0):
    return round((time.perf_counter() - t0) * 1000)
