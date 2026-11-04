"""FastAPI service around the ONNX classifier. Milestone 2, step 1.

Run:  python -m uv run uvicorn complaintops.api:app --port 8000      then open http://127.0.0.1:8000/docs
Endpoints: GET /health (is the process up), GET /ready (is the model loaded), GET /model, POST /predict.
"""
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from complaintops.serve_onnx import Classifier

MODEL_NAME = os.environ.get("MODEL_NAME", "complaint-classifier v4 (DistilBERT + LoRA, ONNX int8)")
MAX_TEXTS = 32
MAX_CHARS = 10_000


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.clf = Classifier()  # fails loudly at start-up if data/onnx/ is missing
    yield


app = FastAPI(title="ComplaintOps", version="0.1.0",
              description="Classifies consumer financial complaints into product and issue.", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
                   allow_methods=["GET", "POST"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready(request: Request):
    return {"status": "ready" if getattr(request.app.state, "clf", None) else "loading"}


@app.get("/model")
def model_info(request: Request):
    clf = request.app.state.clf
    return {"model": MODEL_NAME, "n_classes": len(clf.classes), "classes": clf.classes}


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
