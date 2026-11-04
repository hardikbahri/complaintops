"""Similar-complaint search with Postgres + pgvector. Milestone 2, step 2.

Each past complaint (from the TRAIN split, so it has a known label) is turned into a 384-number vector by a
small sentence-embedding model. Similar complaints end up close together, and Postgres finds the nearest
ones with an HNSW index. The agent uses this to show "here are similar past complaints and how they were labelled".

    docker compose up -d                                   # start the database
    python -m complaintops.vector_store index --n 50000    # embed and store 50,000 train complaints
    python -m complaintops.vector_store evaluate           # sanity check on validation complaints
    python -m complaintops.vector_store search "my bank charged me twice"
"""
import argparse
import os

import numpy as np
import pandas as pd
import psycopg
from pgvector.psycopg import register_vector

from complaintops.taxonomy import MERGE_MAP, merge_label

DB_URL = os.environ.get("DATABASE_URL", "postgresql://complaintops:complaintops@localhost:5433/complaintops")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DIM = 384
SNIPPET = 300

_embedder = None


def embed(texts, batch_size=128):
    """Unit-length vectors (so cosine similarity is a dot product), one row per text."""
    global _embedder
    if _embedder is None:
        import torch
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(EMBED_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    return _embedder.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                            show_progress_bar=False).astype("float32")


def connect():
    conn = psycopg.connect(DB_URL, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def init_db(conn, drop=False):
    if drop:
        conn.execute("DROP TABLE IF EXISTS complaints")
    conn.execute(f"""CREATE TABLE IF NOT EXISTS complaints (
        id BIGSERIAL PRIMARY KEY, source_row INTEGER, date DATE, product TEXT, label TEXT,
        narrative TEXT, embedding vector({DIM}))""")


def index_complaints(n=50_000, seed=42, chunk=2000):
    """Embed and store a random sample of train complaints (drops and rebuilds the table)."""
    conn = connect()
    init_db(conn, drop=True)
    df = pd.read_parquet("data/processed/train.parquet")
    df = df.sample(min(n, len(df)), random_state=seed)
    for i in range(0, len(df), chunk):
        part = df.iloc[i:i + chunk]
        vecs = embed(part["narrative"].tolist())
        rows = [(int(idx), d.date(), p, lab, text[:2000], v)
                for idx, d, p, lab, text, v in zip(part.index, part["date"], part["product"], part["label"],
                                                   part["narrative"], vecs)]
        with conn.cursor() as cur:
            cur.executemany("INSERT INTO complaints (source_row, date, product, label, narrative, embedding) "
                            "VALUES (%s, %s, %s, %s, %s, %s)", rows)
        print(f"  indexed {min(i + chunk, len(df))}/{len(df)}", flush=True)
    conn.execute("CREATE INDEX complaints_embedding_idx ON complaints USING hnsw (embedding vector_cosine_ops)")
    conn.execute("ANALYZE complaints")
    return len(df)


def search(text, k=5, label=None, product=None, conn=None):
    """The k most similar stored complaints: [{id, similarity, label, product, date, snippet}]."""
    own = conn is None
    conn = conn or connect()
    try:
        vec = embed([text])[0]
        where, params = [], []
        if label:  # the stored labels are the original 26; a merged label matches all of its members
            where.append("label = ANY(%s)")
            params.append([k for k, v in MERGE_MAP.items() if v == label] or [label])
        if product:
            where.append("product = %s")
            params.append(product)
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        conn.execute("SET hnsw.ef_search = 100")  # look at more candidates: better recall, still milliseconds
        rows = conn.execute(
            f"SELECT id, 1 - (embedding <=> %s) AS sim, label, product, date, narrative FROM complaints "
            f"{clause} ORDER BY embedding <=> %s LIMIT %s", [vec, *params, vec, k]).fetchall()
        return [dict(id=r[0], similarity=float(r[1]), label=merge_label(r[2]), product=r[3], date=str(r[4]),
                     snippet=r[5][:SNIPPET]) for r in rows]
    finally:
        if own:
            conn.close()


def evaluate(n=500, k=5, seed=7):
    """Do neighbours of VALIDATION complaints (never stored) carry the right label? A sanity check, not a model."""
    conn = connect()
    va = pd.read_parquet("data/processed/val.parquet").sample(n, random_state=seed)
    vecs = embed(va["narrative"].tolist())
    top1_label = top1_product = vote_label = 0
    for (_, row), v in zip(va.iterrows(), vecs):
        conn.execute("SET hnsw.ef_search = 100")
        hits = conn.execute("SELECT label, product FROM complaints ORDER BY embedding <=> %s LIMIT %s",
                            [v, k]).fetchall()
        labels = [h[0] for h in hits]
        top1_label += labels[0] == row["label"]
        top1_product += hits[0][1] == row["product"]
        vote_label += max(set(labels), key=labels.count) == row["label"]
    conn.close()
    return dict(n=n, k=k, top1_same_label=top1_label / n, top1_same_product=top1_product / n,
                majority_of_k_same_label=vote_label / n)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("index")
    p.add_argument("--n", type=int, default=50_000)
    sub.add_parser("evaluate")
    p = sub.add_parser("search")
    p.add_argument("text")
    p.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    if args.cmd == "index":
        print(f"indexed {index_complaints(args.n)} complaints")
    elif args.cmd == "evaluate":
        print(evaluate())
    else:
        for h in search(args.text, args.k):
            print(f"{h['similarity']:.3f}  {h['label']}  ({h['date']})\n      {h['snippet'][:160]}")


if __name__ == "__main__":
    main()
