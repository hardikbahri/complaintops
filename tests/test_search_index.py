"""The in-memory similar-complaint search: the same answers as pgvector, with no database."""
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from complaintops import search_index, tools
from complaintops.search_index import MemoryIndex
from complaintops.taxonomy import MERGE_MAP, MERGED_LABEL

TRIO = list(MERGE_MAP)


def make_index():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(6, 8)).astype("float32")
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    meta = {"id": [10, 11, 12, 13, 14, 15],
            "label": [TRIO[0], TRIO[1], "Mortgage | Struggling to pay mortgage", TRIO[2], "Other", "Other"],
            "product": ["Credit reporting", "Credit reporting", "Mortgage", "Credit reporting", "Other", "Other"],
            "date": ["2023-01-01"] * 6, "snippet": [f"snippet {i}" for i in range(6)]}
    return MemoryIndex(v, meta), v


def test_a_complaints_nearest_neighbour_is_itself_and_results_are_ordered():
    idx, v = make_index()
    hits = idx.search(v[2], k=4)
    assert hits[0]["id"] == 12 and hits[0]["similarity"] == pytest.approx(1.0, abs=1e-5)
    assert [h["similarity"] for h in hits] == sorted((h["similarity"] for h in hits), reverse=True)
    assert hits[0]["snippet"] == "snippet 2" and len(hits) == 4


def test_the_product_filter_only_returns_that_product():
    idx, v = make_index()
    assert {h["product"] for h in idx.search(v[0], k=10, product="Credit reporting")} == {"Credit reporting"}
    assert idx.search(v[0], k=10, product="Student loan") == []  # nothing matches: an empty list, not an error


def test_labels_come_back_merged_and_a_merged_label_filter_matches_all_its_members():
    idx, v = make_index()
    hits = idx.search(v[0], k=10)
    assert {h["label"] for h in hits if h["id"] in (10, 11, 13)} == {MERGED_LABEL}  # the three originals are one class
    by_merged = idx.search(v[0], k=10, label=MERGED_LABEL)
    assert sorted(h["id"] for h in by_merged) == [10, 11, 13]


def test_k_larger_than_the_index_just_returns_everything():
    idx, v = make_index()
    assert len(idx.search(v[0], k=50)) == 6


def test_an_index_written_as_float16_loads_and_finds_the_same_top_hit(tmp_path):
    idx, v = make_index()
    np.save(tmp_path / "vectors.npy", v.astype("float16"))
    with gzip.open(tmp_path / "meta.json.gz", "wt", encoding="utf-8") as f:
        json.dump(idx.meta, f)
    loaded = MemoryIndex.load(tmp_path)
    assert [loaded.search(v[i], k=1)[0]["id"] for i in range(6)] == [10, 11, 12, 13, 14, 15]


def test_the_search_tool_uses_the_memory_backend_when_asked(monkeypatch):
    idx, v = make_index()
    monkeypatch.setenv("SEARCH_BACKEND", "memory")
    monkeypatch.setattr(search_index, "_index", idx)
    monkeypatch.setattr(search_index, "embed_query", lambda text: v[2])
    out = tools.search_similar_complaints("anything", k=3)
    assert out["results"][0]["id"] == 12 and len(out["results"]) == 3


def parity_possible():
    try:
        import fastembed  # noqa: F401
        import psycopg

        from complaintops.vector_store import DB_URL

        with psycopg.connect(DB_URL, connect_timeout=2) as c:
            has_rows = c.execute("select count(*) from complaints").fetchone()[0] > 0
        return has_rows and Path("data/search_index/vectors.npy").exists()
    except Exception:
        return False


@pytest.mark.skipif(not parity_possible(), reason="needs the database, the exported index and fastembed")
def test_the_exported_index_returns_the_same_neighbours_as_pgvector():
    from complaintops import vector_store

    same = total = 0
    for q in ["I was charged an overdraft fee even though I had enough money in my account.",
              "A collector keeps calling me about a debt that is not mine.",
              "My mortgage servicer lost my payment and charged late fees."]:
        a = {h["id"] for h in vector_store.search(q, k=5)}
        b = {h["id"] for h in search_index.search(q, k=5)}
        same += len(a & b)
        total += 5
    assert same / total >= 0.9  # float16 storage and a different embedding runtime may reorder a near-tie
