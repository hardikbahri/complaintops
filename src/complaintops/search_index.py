"""In-memory similar-complaint search, for hosts without Postgres. Milestone 2 deployment.

The same 50,000 stored complaints as the pgvector table, exported to two small files and searched with one matrix
product (50,000 x 384 floats takes milliseconds). Needs only NumPy to search, and fastembed (ONNX, no PyTorch) to embed
the query: its vectors match the sentence-transformers ones the index was built with (cosine 1.0 in a check).
    python -m complaintops.search_index export        # from the running pgvector database
    SEARCH_BACKEND=memory                              # make the agent's search tool use this index
"""
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

from complaintops.taxonomy import MERGE_MAP, merge_label

INDEX_DIR = Path(os.environ.get("SEARCH_INDEX_DIR", "data/search_index"))
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SNIPPET = 300

_embedder = None
_index = None


def embed_query(text):
    """A unit-length 384-number vector for one text, without PyTorch."""
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding

        _embedder = TextEmbedding(EMBED_MODEL)
    v = np.asarray(next(iter(_embedder.embed([text]))), dtype="float32")
    return v / np.linalg.norm(v)


class MemoryIndex:
    def __init__(self, vectors, meta):
        self.vectors = np.asarray(vectors)  # float16 as stored: half the memory; scored in float32 chunks
        self.meta = meta  # dict of equal-length lists: id, label, product, date, snippet
        self.products = np.array(meta["product"])
        self.labels = np.array(meta["label"])

    @classmethod
    def load(cls, directory=INDEX_DIR):
        directory = Path(directory)
        vectors = np.load(directory / "vectors.npy")  # float16: halves the file and the memory
        with gzip.open(directory / "meta.json.gz", "rt", encoding="utf-8") as f:
            return cls(vectors, json.load(f))

    def search(self, vec, k=5, label=None, product=None):
        mask = np.ones(len(self.vectors), dtype=bool)
        if product:
            mask &= self.products == product
        if label:  # stored labels are the original 26; a merged label matches all of its members
            mask &= np.isin(self.labels, [m for m, v in MERGE_MAP.items() if v == label] or [label])
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            return []
        q = np.asarray(vec, dtype="float32")
        scores = np.empty(len(idx), dtype="float32")
        for i in range(0, len(idx), 8192):  # convert one chunk at a time, never the whole matrix
            scores[i:i + 8192] = self.vectors[idx[i:i + 8192]].astype("float32") @ q
        order = np.argsort(-scores)[:k]
        top, sims = idx[order], scores[order]
        return [dict(id=int(self.meta["id"][i]), similarity=float(s), label=merge_label(self.meta["label"][i]),
                     product=self.meta["product"][i], date=self.meta["date"][i], snippet=self.meta["snippet"][i])
                for i, s in zip(top, sims)]


def get_index():
    global _index
    if _index is None:
        _index = MemoryIndex.load()
    return _index


def search(text, k=5, label=None, product=None, conn=None):
    """Same call shape as vector_store.search, so the tools can use either backend. `conn` is ignored."""
    return get_index().search(embed_query(text), k=k, label=label, product=product)


def export_index(directory=INDEX_DIR):
    """Dump the pgvector table (the source of truth) to vectors.npy (float16) and meta.json.gz."""
    from complaintops.vector_store import connect

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    conn = connect()
    rows = conn.execute("SELECT id, label, product, date, narrative, embedding FROM complaints ORDER BY id").fetchall()
    conn.close()
    meta = {"id": [r[0] for r in rows], "label": [r[1] for r in rows], "product": [r[2] for r in rows],
            "date": [str(r[3]) for r in rows], "snippet": [r[4][:SNIPPET] for r in rows]}
    vectors = np.stack([r[5].to_numpy() for r in rows]).astype("float16")  # pgvector's Vector type -> NumPy
    np.save(directory / "vectors.npy", vectors)
    with gzip.open(directory / "meta.json.gz", "wt", encoding="utf-8") as f:
        json.dump(meta, f)
    return len(rows), vectors.nbytes / 1e6, (directory / "meta.json.gz").stat().st_size / 1e6


if __name__ == "__main__":
    if sys.argv[1:] == ["export"]:
        n, v_mb, m_mb = export_index()
        print(f"exported {n} complaints: vectors {v_mb:.0f} MB, metadata {m_mb:.0f} MB -> {INDEX_DIR}")
    else:
        print(__doc__)
