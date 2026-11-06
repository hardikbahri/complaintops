# Hugging Face Space (Docker) image: the API, the page, the int8 model and the in-memory search index.
# It listens on 7860 and runs as a non-root user, as Spaces require.
FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface \
    FASTEMBED_CACHE_PATH=/home/user/.cache/fastembed
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Fetch the query-embedding model while building, so the first visitor does not wait for it.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')"

COPY --chown=user src ./src
COPY --chown=user web ./web
COPY --chown=user data ./data

# Defaults. Set LLM_PROVIDER=groq and GROQ_API_KEY as Space secrets to use a real LLM for the reviewer notes;
# leave the LANGFUSE_* variables unset so visitors' text is never sent to a tracing service.
ENV PYTHONPATH=/home/user/app/src \
    SEARCH_BACKEND=memory \
    TAXONOMY=merged \
    WEB_DIR=web \
    LLM_PROVIDER=mock \
    REVIEW_DB=/tmp/review_queue.db \
    RATE_PER_MINUTE=6 \
    RATE_PER_DAY=300

EXPOSE 7860
CMD ["uvicorn", "complaintops.api:app", "--host", "0.0.0.0", "--port", "7860"]
