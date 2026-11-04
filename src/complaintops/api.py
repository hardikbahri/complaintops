"""FastAPI service: the classifier, the triage agent and the web page. Milestone 2.

Run:  python -m uv run uvicorn complaintops.api:app --port 8000
      then open http://127.0.0.1:8000  (the page)  or  /docs  (the API)
Endpoints: GET /health, GET /ready, GET /model, POST /predict (classifier only), POST /triage (the agent).
"""
import asyncio
import collections
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from complaintops.config import load_env
from complaintops.serve_onnx import Classifier

MODEL_NAME = os.environ.get("MODEL_NAME", "complaint-classifier v4 (DistilBERT + LoRA, ONNX int8)")
MAX_TEXTS = 32
MAX_CHARS = 10_000
TRIAGE_MAX_CHARS = 2_000
WEB_DIR = Path(os.environ.get("WEB_DIR", "web"))


class PredictRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=MAX_TEXTS,
                             description=f"1 to {MAX_TEXTS} complaint texts")
    top_k: int = Field(3, ge=1, le=5, description="how many alternative classes to return per text")

    @field_validator("texts")
    @classmethod
    def texts_must_be_reasonable(cls, texts):
        for t in texts:
            if not t.strip():
                raise ValueError("texts must not be empty")
            if len(t) > MAX_CHARS:
                raise ValueError(f"each text must be at most {MAX_CHARS} characters")
        return texts


class Choice(BaseModel):
    label: str
    confidence: float


class Prediction(BaseModel):
    label: str
    product: str
    issue: str
    confidence: float
    alternatives: list[Choice]


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    model: str
    latency_ms: float


class TriageRequest(BaseModel):
    text: str = Field(min_length=1, max_length=TRIAGE_MAX_CHARS, description="one complaint")

    @field_validator("text")
    @classmethod
    def not_blank(cls, t):
        if not t.strip():
            raise ValueError("text must not be empty")
        return t


class RateLimiter:
    """Per-client and global limits, so a public demo cannot burn through a free LLM quota."""

    def __init__(self, per_minute=8, per_day=400):
        self.per_minute, self.per_day = per_minute, per_day
        self.by_client = collections.defaultdict(collections.deque)
        self.day = collections.deque()

    def check(self, client):
        now = time.time()
        q = self.by_client[client]
        while q and now - q[0] > 60:
            q.popleft()
        while self.day and now - self.day[0] > 86_400:
            self.day.popleft()
        if len(q) >= self.per_minute:
            raise HTTPException(429, f"Slow down: at most {self.per_minute} requests per minute.")
        if len(self.day) >= self.per_day:
            raise HTTPException(429, "The demo's daily LLM budget is used up; try the recorded examples.")
        q.append(now)
        self.day.append(now)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_env()  # a local .env supplies the LLM key; real environment variables always win
    app.state.clf = Classifier()  # fails loudly at start-up if data/onnx/ is missing
    app.state.agent, app.state.agent_error = None, None
    app.state.limiter = RateLimiter(int(os.environ.get("RATE_PER_MINUTE", 8)), int(os.environ.get("RATE_PER_DAY", 400)))
    app.state.slots = asyncio.Semaphore(2)  # at most two agent runs at once
    try:
        from complaintops.agent import ComplaintAgent

        agent = ComplaintAgent()
        await agent.__aenter__()  # starts the MCP server the agent talks to
        app.state.agent = agent
    except Exception as e:  # the classifier endpoints keep working without the agent
        app.state.agent_error = f"{type(e).__name__}: {e}"
    yield
    if app.state.agent is not None:
        await app.state.agent.__aexit__(None, None, None)


app = FastAPI(title="ComplaintOps", version="0.2.0",
              description="Classifies consumer financial complaints into product and issue, and triages the "
                          "uncertain ones with an agent.", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
                   allow_methods=["GET", "POST"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready(request: Request):
    s = request.app.state
    return {"status": "ready" if getattr(s, "clf", None) else "loading",
            "agent": "ready" if getattr(s, "agent", None) else "unavailable"}


@app.get("/model")
def model_info(request: Request):
    clf = request.app.state.clf
    agent = request.app.state.agent
    return {"model": MODEL_NAME, "n_classes": len(clf.classes), "classes": clf.classes,
            "llm": agent.llm.name if agent else None, "agent_threshold": agent.threshold if agent else None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request):
    t0 = time.perf_counter()
    results = request.app.state.clf.predict(req.texts, top_k=max(req.top_k, 2))
    preds = []
    for r in results:
        product, _, issue = r["label"].partition(" | ")
        preds.append(Prediction(label=r["label"], product=product, issue=issue or "(no specific issue)",
                                confidence=r["confidence"], alternatives=r["alternatives"][:req.top_k]))
    return PredictResponse(predictions=preds, model=MODEL_NAME,
                           latency_ms=round((time.perf_counter() - t0) * 1000, 1))


@app.post("/triage")
async def triage(req: TriageRequest, request: Request):
    """Run the agent on one complaint: classifier, then (if unsure) similar complaints and an LLM, then a human."""
    s = request.app.state
    if s.agent is None:
        raise HTTPException(503, f"The agent is not available ({s.agent_error or 'still starting'}).")
    fwd = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    s.limiter.check(fwd or (request.client.host if request.client else "unknown"))
    t0 = time.perf_counter()
    async with s.slots:
        result = await s.agent.run(req.text)
    return {"mode": "live", "llm_provider": s.agent.llm.name, "threshold": s.agent.threshold,
            "total_ms": round((time.perf_counter() - t0) * 1000), **result}


if WEB_DIR.is_dir():  # the page; mounted last so the API routes above take precedence
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
